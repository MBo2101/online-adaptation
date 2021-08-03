# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 10:30:55 2021

@author: MBo
"""

from RTArrays.RTArray import RTArray
import numpy as np

class Structure(RTArray):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.base_value = 0.
    
    @property
    def volume(self):
        if self.ndarray is None:
            self.load_file()
        voxels = np.count_nonzero(self.ndarray==1)
        return voxels * self.voxel_volume
