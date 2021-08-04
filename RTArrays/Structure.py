# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 10:30:55 2021

@author: MBo
"""

from RTArrays.RTArray import RTArray
import numpy as np

class Structure(RTArray):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.base_value = 0.
    
    @property
    def volume(self):
        if self.ndarray is None:
            self.load_file()
        voxels = np.count_nonzero(self.ndarray==1)
        return voxels * self.voxel_volume

    @property
    def voxel_indices(self):
        return self.array_1D.nonzero()[0]
    
    def get_intersection_indices(self, structure):
        '''
        Returns intersection (voxel indices) with other structure (Structure class).
        '''
        return np.intersect1d(self.voxel_indices, structure.voxel_indices)
    
    def get_union_indices(self, structure):
        '''
        Returns union (voxel indices) with other structure (Structure class).
        '''
        return np.union1d(self.voxel_indices, structure.voxel_indices)
    
    def exclude_structure_indices(self, structure):
        '''
        Returns voxel indices excluding all voxels that are shared with other structure (Structure class).
        Equivalent to logical operator minus/except.
        '''
        return np.setdiff1d(self.voxel_indices, structure.voxel_indices)
    
    def get_indices_in_mask(self, mask):
        '''
        Returns structure's voxel indices within input mask (Structure class).
        Use if a mask was applied to the Dij matrix.
        '''
        array_in_mask = self.array_1D[mask.voxel_indices]
        return np.nonzero(array_in_mask)[0]


        