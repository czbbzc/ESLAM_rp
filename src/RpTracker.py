import torch
import copy
import os
import time
import cv2

from colorama import Fore, Style
from torch.utils.data import DataLoader
from tqdm import tqdm

from src import Tracker
from src.common import (matrix_to_cam_pose, cam_pose_to_matrix, get_samples, get_samples_rp, get_samples_gt)
from src.utils.rp_datasets import get_dataset
from src.utils.Frame_Visualizer import Frame_Visualizer


class RpTracker(Tracker.Tracker):
    def __init__(self, cfg, args, eslam):
        
        super(RpTracker,self).__init__(cfg, args, eslam)
        
        self.rp = cfg['tracking']['rp']
        self.rp_num = cfg['tracking']['rp_num']
        self.rp_type = cfg['tracking']['rp_type']
        self.rp_file = cfg['tracking']['rp_file']
        self.w_rp_loss = cfg['tracking']['w_rp_loss']
        
        self.frame_reader = get_dataset(cfg, args, self.scale, device=self.device, rp=self.rp, rp_file=self.rp_file)
        self.n_img = len(self.frame_reader)
        self.frame_loader = DataLoader(self.frame_reader, batch_size=1, shuffle=False,
                                       num_workers=1, pin_memory=False, prefetch_factor=2)

        self.visualizer = Frame_Visualizer(freq=cfg['tracking']['vis_freq'], inside_freq=cfg['tracking']['vis_inside_freq'],
                                           vis_dir=os.path.join(self.output, 'tracking_vis'), renderer=self.renderer,
                                           truncation=self.truncation, verbose=self.verbose, device=self.device)

        self.H, self.W, self.fx, self.fy, self.cx, self.cy = eslam.H, eslam.W, eslam.fx, eslam.fy, eslam.cx, eslam.cy

        self.decoders = copy.deepcopy(self.shared_decoders)

        self.planes_xy = copy.deepcopy(self.shared_planes_xy)
        self.planes_xz = copy.deepcopy(self.shared_planes_xz)
        self.planes_yz = copy.deepcopy(self.shared_planes_yz)

        self.c_planes_xy = copy.deepcopy(self.shared_c_planes_xy)
        self.c_planes_xz = copy.deepcopy(self.shared_c_planes_xz)
        self.c_planes_yz = copy.deepcopy(self.shared_c_planes_yz)

        for p in self.decoders.parameters():
            p.requires_grad_(False)
            
        self.K = torch.tensor([[self.fx, 0, self.cx],
                        [0, self.fy, self.cy],
                        [0, 0, 1]]).to(self.device).double()
        self.trans = torch.eye(4).to(self.device).double()
        self.trans[2,2] = -1
        self.trans[1,1] = -1
        self.H_final = self.H - 2 * self.ignore_edge_H
        self.W_final = self.W - 2 * self.ignore_edge_W

    def optimize_tracking(self, cam_pose, gt_color, gt_depth, gt_cpd, batch_size, optimizer, idx):
        """
        Do one iteration of camera tracking. Sample pixels, render depth/color, calculate loss and backpropagation.

        Args:
            cam_pose (tensor): camera pose.
            gt_color (tensor): ground truth color image of the current frame.
            gt_depth (tensor): ground truth depth image of the current frame.
            batch_size (int): batch size, number of sampling rays.
            optimizer (torch.optim): camera optimizer.

        Returns:
            loss (float): The value of loss.
        """
        all_planes = (self.planes_xy, self.planes_xz, self.planes_yz, self.c_planes_xy, self.c_planes_xz, self.c_planes_yz)
        device = self.device
        H, W, fx, fy, cx, cy = self.H, self.W, self.fx, self.fy, self.cx, self.cy

        c2w = cam_pose_to_matrix(cam_pose)
        
        self.gt_color = gt_color
        c2w_gt = self.gt_c2w.to(device)
        
        # batch_rays_o, batch_rays_d, batch_gt_depth, batch_gt_color = get_samples(self.ignore_edge_H, H-self.ignore_edge_H,
        #                                                                          self.ignore_edge_W, W-self.ignore_edge_W,
        #                                                                          batch_size, H, W, fx, fy, cx, cy, c2w,
        #                                                                          gt_depth, gt_color, device)
        batch_rays_o, batch_rays_d, batch_gt_depth, batch_gt_color, self.indices, batch_rays_o_gt, batch_rays_d_gt = get_samples_gt(self.ignore_edge_H, H-self.ignore_edge_H,
                                                                                 self.ignore_edge_W, W-self.ignore_edge_W,
                                                                                 batch_size, H, W, fx, fy, cx, cy, c2w, c2w_gt, 
                                                                                 gt_depth, gt_color, device)

        # should pre-filter those out of bounding box depth value
        # with torch.no_grad():
        #     det_rays_o = batch_rays_o.clone().detach().unsqueeze(-1)  # (N, 3, 1)
        #     det_rays_d = batch_rays_d.clone().detach().unsqueeze(-1)  # (N, 3, 1)
        #     t = (self.bound.unsqueeze(0).to(
        #         device) - det_rays_o) / det_rays_d
        #     t, _ = torch.min(torch.max(t, dim=2)[0], dim=1)
        #     inside_mask = t >= batch_gt_depth
        #     inside_mask = inside_mask & (batch_gt_depth > 0)

        # batch_rays_d = batch_rays_d[inside_mask]
        # batch_rays_o = batch_rays_o[inside_mask]
        # batch_gt_depth = batch_gt_depth[inside_mask]
        # batch_gt_color = batch_gt_color[inside_mask]

        depth, color, sdf, z_vals = self.renderer.render_batch_ray(all_planes, self.decoders, batch_rays_d, batch_rays_o,
                                                                   self.device, self.truncation, gt_depth=batch_gt_depth)

        ## Filtering the rays for which the rendered depth error is greater than 10 times of the median depth error
        depth_error = (batch_gt_depth - depth.detach()).abs()
        error_median = depth_error.median()
        depth_mask = (depth_error < 10 * error_median)

        ## SDF losses
        loss = self.sdf_losses(sdf[depth_mask], z_vals[depth_mask], batch_gt_depth[depth_mask])
        sdf_loss = loss
        
        ## Color Loss
        color_loss = torch.square(batch_gt_color - color)[depth_mask].mean()
        loss = loss + self.w_color * color_loss

        ### Depth loss
        depth_loss = torch.square(batch_gt_depth[depth_mask] - depth[depth_mask]).mean()
        loss = loss + self.w_depth * depth_loss

        if self.rp:
            # rp_loss = self.reprojection_loss(depth, batch_rays_o, batch_rays_d, batch_gt_cpd, idx)
            rp_loss = self.reprojection_loss_gt(depth, batch_rays_o, batch_rays_d, batch_rays_o_gt, batch_rays_d_gt, batch_gt_depth, idx)
            loss += self.w_rp_loss * rp_loss
            
            # print(sdf_loss.item(), depth_loss.item(),color_loss.item(),rp_loss.item(),depth.mean().item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()
    
    def reprojection_loss_gt(self, depth, rays_o, rays_d, rays_o_gt, rays_d_gt, gt_depth, idx):
        num_not_for_rp = self.tracking_pixels - self.rp_num
        rays_o = rays_o[num_not_for_rp:]
        rays_d = rays_d[num_not_for_rp:]
        rays_o_gt = rays_o_gt[num_not_for_rp:]
        rays_d_gt = rays_d_gt[num_not_for_rp:]
        depth = depth[num_not_for_rp:]
        gt_depth = gt_depth[num_not_for_rp:]
        points = rays_o + rays_d * depth[...,None]
        points_gt = rays_o_gt + rays_d_gt * gt_depth[...,None]
        if self.rp_type == "pre_1":
            # print("pre frame index: ",self.idx[0])
            pose = self.estimate_c2w_list[idx-1]
            pose_gt = self.gt_c2w_list[idx-1]
        # print(f"reprojection loss {pose.shape} and {intr.shape} points shape {points.shape}")
        repro_idx = self.reprojection(points,pose)  ### B len(rp_ray_idx) 2
        repro_idx_gt = self.reprojection(points_gt,pose_gt)  ### B len(rp_ray_idx) 2
        
        self.indices = self.indices[num_not_for_rp:]
        h = self.indices // self.W_final + self.ignore_edge_H
        w = self.indices % self.W_final + self.ignore_edge_W
        # print(self.indices.shape,h[0],w[0],repro_idx_gt[0],repro_idx[0])
        self.gt_color = self.gt_color.cpu().numpy() * 255
        pre1 = self.pre_color.clone().cpu().numpy() * 255
        pre2 = self.pre_color.clone().cpu().numpy() * 255
        
        gt_color1 = self.gt_color.squeeze(0)
        pre11 = pre1.squeeze(0)
        pre22 = pre2.squeeze(0)
        
        for nd in range(3):
            cv2.circle(gt_color1, tuple([int(w[nd]),int(h[nd])]), 2, (0, 255, 0), -1)
            # cv2.circle(depth1, tuple([int(point_a[0]),int(point_a[1])]), 3, (0, 255, 0), -1)
            
            cv2.circle(pre11, tuple([int(repro_idx_gt[nd][0]),int(repro_idx_gt[nd][1])]), 2, (0, 255, 0), -1)
            cv2.circle(pre22, tuple([int(repro_idx[nd][0]),int(repro_idx[nd][1])]), 2, (0, 255, 0), -1)

        cv2.imwrite(f'imgs/image_a_with_point4.png', gt_color1)
        cv2.imwrite(f'imgs/image_b_with_point4.png', pre11)
        cv2.imwrite(f'imgs/image_c_with_point4.png', pre22)
        
        mask = (repro_idx_gt[...,0]>self.ignore_edge_H) & (repro_idx_gt[...,0]<self.H_final) & (repro_idx_gt[...,1]>self.ignore_edge_W) & (repro_idx_gt[...,1]<self.W_final)
        return self.L1_loss(repro_idx_gt[mask],repro_idx[mask])
    
    
    def reprojection_loss(self, depth, rays_o, rays_d, gt_cpd, idx):
        # print(depth.shape,rays_o.shape,rays_d.shape, gt_cpd.shape)
        num_not_for_rp = self.tracking_pixels - self.rp_num
        rays_o = rays_o[num_not_for_rp:]
        rays_d = rays_d[num_not_for_rp:]
        depth = depth[num_not_for_rp:]
        points = rays_o + rays_d * depth[...,None]

        if self.rp_type == "pre_1":
            # print("pre frame index: ",self.idx[0])
            pose = self.estimate_c2w_list[idx-1]
        # print(f"reprojection loss {pose.shape} and {intr.shape} points shape {points.shape}")
        repro_idx = self.reprojection(points,pose)  ### B len(rp_ray_idx) 2
        
        rp_conf = gt_cpd[...,-1]
        rp_cpd = gt_cpd[...,2:4]

        rp_cpd[...,0] = rp_cpd[...,0]/self.W_final
        repro_idx[...,0] = repro_idx[...,0]/self.W_final
        rp_cpd[...,1] = rp_cpd[...,1]/self.H_final
        repro_idx[...,1] = repro_idx[...,1]/self.H_final

        # print(f"before L1 loss {rp_cpd.shape} {repro_idx.shape} {rp_conf.shape} {depth.shape}")
        # print(f"before L1 loss {repro_idx_gt[14,::5]} {repro_idx[14,::5]} {rp_conf[14,::5]} {depth[14,::5]}")
        # print(repro_idx.shape,repro_idx_gt.shape,rp_conf.shape,depth_var.shape,rgb_error.shape)  
        return self.L1_loss(rp_cpd,repro_idx,rp_conf[...,None])
    
    def reprojection(self, point, pose):
        
        pose_inv = self.invert(pose[:3])
        point = torch.cat([point,torch.ones_like(point[...,:1])],dim=-1).to(self.device).double()
        # print(self.K.device,self.trans[:3,:3].device,pose_inv.device,point.device)
        # print(self.K.dtype,self.trans[:3,:3].dtype,pose_inv.dtype,point.dtype)
        p_uv = self.K @ self.trans[:3,:3] @ pose_inv @ point[...,None]
        
        p_uv = p_uv.squeeze()
        # print(p_uv.shape)
        p_uv = p_uv / p_uv[...,2][...,None]

        # print(f"p_uv shape {p_uv.shape} sample {p_uv[0]}")

        return p_uv[...,:2]
    
    def invert(self,pose,use_inverse=False):
        # invert a camera pose
        R,t = pose[...,:3],pose[...,3:]
        R_inv = R.inverse() if use_inverse else R.transpose(-1,-2)
        t_inv = (-R_inv@t)[...,0]
        R_inv = R_inv.double()
        t_inv = t_inv.double()
        # print(R_inv.shape,t_inv.shape)
        pose_inv = torch.cat([R_inv,t_inv[...,None]],dim=-1).to(self.device) # [...,3,4]
        assert(pose_inv.shape[-2:]==(3,4))
        return pose_inv
    

    def L1_loss(self, pred, label=0, conf=None, depth_var=None, rgb_error=None):
        loss = (pred.contiguous()-label).abs()
        
        if conf is not None:
            loss = loss * conf
        return loss.mean()
    
    def update_params_from_mapping(self):
        """
        Update the parameters of scene representation from the mapping thread.
        """
        if self.mapping_idx[0] != self.prev_mapping_idx:
            if self.verbose:
                print('Tracking: update the parameters from mapping')

            self.decoders.load_state_dict(self.shared_decoders.state_dict())

            for planes, self_planes in zip(
                    [self.shared_planes_xy, self.shared_planes_xz, self.shared_planes_yz],
                    [self.planes_xy, self.planes_xz, self.planes_yz]):
                for i, plane in enumerate(planes):
                    self_planes[i] = plane.detach()

            for c_planes, self_c_planes in zip(
                    [self.shared_c_planes_xy, self.shared_c_planes_xz, self.shared_c_planes_yz],
                    [self.c_planes_xy, self.c_planes_xz, self.c_planes_yz]):
                for i, c_plane in enumerate(c_planes):
                    self_c_planes[i] = c_plane.detach()

            self.prev_mapping_idx = self.mapping_idx[0].clone()
            
    def run(self):
        """
            Runs the tracking thread for the input RGB-D frames.

            Args:
                None

            Returns:
                None
        """
        device = self.device
        all_planes = (self.planes_xy, self.planes_xz, self.planes_yz, self.c_planes_xy, self.c_planes_xz, self.c_planes_yz)

        if self.verbose:
            pbar = self.frame_loader
        else:
            pbar = tqdm(self.frame_loader, smoothing=0.05)

        for idx, gt_color, gt_depth, gt_c2w, gt_cpd in pbar:
            gt_color = gt_color.to(device, non_blocking=True)
            gt_depth = gt_depth.to(device, non_blocking=True)
            gt_c2w = gt_c2w.to(device, non_blocking=True)
            
            gt_cpd = gt_cpd[0]
            self.gt_c2w = gt_c2w

            if not self.verbose:
                pbar.set_description(f"Tracking Frame {idx[0]}")
            idx = idx[0]

            # initiate mapping every self.every_frame frames
            if idx > 0 and (idx % self.every_frame == 1 or self.every_frame == 1):
                while self.mapping_idx[0] != idx - 1:
                    time.sleep(0.001)
                pre_c2w = self.estimate_c2w_list[idx - 1].unsqueeze(0).to(device)

            self.update_params_from_mapping()

            if self.verbose:
                print(Fore.MAGENTA)
                print("Tracking Frame ",  idx.item())
                print(Style.RESET_ALL)

            if idx == 0 or self.gt_camera:
                c2w = gt_c2w
                if not self.no_vis_on_first_frame:
                    self.visualizer.save_imgs(idx, 0, gt_depth, gt_color, c2w.squeeze(), all_planes, self.decoders)

            else:
                if self.const_speed_assumption and idx - 2 >= 0:
                    ## Linear prediction for initialization
                    pre_poses = torch.stack([self.estimate_c2w_list[idx - 2], pre_c2w.squeeze(0)], dim=0)
                    pre_poses = matrix_to_cam_pose(pre_poses)
                    cam_pose = 2 * pre_poses[1:] - pre_poses[0:1]
                else:
                    ## Initialize with the last known pose
                    cam_pose = matrix_to_cam_pose(pre_c2w)

                T = torch.nn.Parameter(cam_pose[:, -3:].clone())
                R = torch.nn.Parameter(cam_pose[:,:4].clone())
                cam_para_list_T = [T]
                cam_para_list_R = [R]
                optimizer_camera = torch.optim.Adam([{'params': cam_para_list_T, 'lr': self.cam_lr_T, 'betas':(0.5, 0.999)},
                                                     {'params': cam_para_list_R, 'lr': self.cam_lr_R, 'betas':(0.5, 0.999)}])

                current_min_loss = torch.tensor(float('inf')).float().to(device)
                for cam_iter in range(self.num_cam_iters):
                    cam_pose = torch.cat([R, T], -1)

                    self.visualizer.save_imgs(idx, cam_iter, gt_depth, gt_color, cam_pose, all_planes, self.decoders)

                    loss = self.optimize_tracking(cam_pose, gt_color, gt_depth, gt_cpd, self.tracking_pixels, optimizer_camera, idx)
                    if loss < current_min_loss:
                        current_min_loss = loss
                        candidate_cam_pose = cam_pose.clone().detach()

                c2w = cam_pose_to_matrix(candidate_cam_pose)

            self.estimate_c2w_list[idx] = c2w.squeeze(0).clone()
            self.gt_c2w_list[idx] = gt_c2w.squeeze(0).clone()
            pre_c2w = c2w.clone()
            
            self.pre_color = gt_color.clone()
            self.pre_depth = gt_depth.clone()
            
            self.idx[0] = idx