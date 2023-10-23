import os
import time

import numpy as np
import torch
import torch.multiprocessing
import torch.multiprocessing as mp

from src import ESLAM
from src import config
from src.RpMapper import RpMapper
from src.RpTracker import RpTracker
from src.utils.datasets import get_dataset
from src.utils.Logger import Logger
from src.utils.Mesher import Mesher
from src.utils.Renderer import Renderer

torch.multiprocessing.set_sharing_strategy('file_system')


class ESLAM_rp(ESLAM.ESLAM):
    """
    ESLAM_rp main class.
    Mainly allocate shared resources, and dispatch mapping and tracking process.
    """

    def __init__(self, cfg, args):
        
        super(ESLAM_rp, self).__init__(cfg, args)
        
        self.renderer = Renderer(cfg, self)
        self.mesher = Mesher(cfg, args, self)
        self.logger = Logger(self)
        self.mapper = RpMapper(cfg, args, self)
        self.tracker = RpTracker(cfg, args, self)
        
        
        
        

# This part is required by torch.multiprocessing
if __name__ == '__main__':
    pass
