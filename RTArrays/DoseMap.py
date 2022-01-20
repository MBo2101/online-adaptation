# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 10:30:55 2021

@author: MBo
"""

from RTArrays.RTArray import RTArray

class DoseMap(RTArray):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.base_value = 0.

    # Methods
    
    def get_array_in_mask(self, mask):
        '''
        Returns 1D dose array within input mask (Structure class).
        Use if a mask was applied to the Dij matrix.
        Returns self.array_1D if mask is None.
        '''
        if mask is not None:
            array_in_mask = self.array_1D[mask.voxel_indices]
            return array_in_mask
        else:
            return self.array_1D
