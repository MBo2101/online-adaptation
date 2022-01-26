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
    
    # Properties

    @property
    def n_voxels(self):
        if self.array_1D is not None:
            n_voxels = np.count_nonzero(self.array_1D==1)
        elif self.ndarray is not None:
            n_voxels = np.count_nonzero(self.ndarray==1)
        return n_voxels
    
    @property
    def volume(self):
        return self.n_voxels * self.voxel_volume

    @property
    def voxel_indices(self):
        return self.array_1D.nonzero()[0]
    
    # Methods
    
    def get_indices_of_intersection(self, structure):
        '''
        Returns voxel indices of intersection with other structure (Structure class).
        '''
        return np.intersect1d(self.voxel_indices, structure.voxel_indices)
    
    def get_indices_of_union(self, structure):
        '''
        Returns voxel indices of union with other structure (Structure class).
        '''
        return np.union1d(self.voxel_indices, structure.voxel_indices)
    
    def get_indices_of_exclusion(self, structure):
        '''
        Returns voxel indices after excluding all voxels that are shared with other structure (Structure class).
        Equivalent to logical operator minus/except.
        '''
        return np.setdiff1d(self.voxel_indices, structure.voxel_indices)

    def get_intersection(self, structure):
        '''
        Returns instance of Structure class of intersection with other structure (Structure class).
        '''
        indices = self.get_indices_of_intersection(structure)
        return Structure(array_1D = self.indices_to_array_1D(indices, self.n_voxels_total), header = self.header)
    
    def get_union(self, structure):
        '''
        Returns instance of Structure class of union with other structure (Structure class).
        '''
        indices = self.get_indices_of_union(structure)
        return Structure(array_1D = self.indices_to_array_1D(indices, self.n_voxels_total), header = self.header)
    
    def get_exclusion(self, structure):
        '''
        Returns instance of Structure class after excluding all voxels that are shared with other structure (Structure class).
        Equivalent to logical operator minus/except.
        '''
        indices = self.get_indices_of_exclusion(structure)
        return Structure(array_1D = self.indices_to_array_1D(indices, self.n_voxels_total), header = self.header)
    
    def get_indices_in_mask(self, mask):
        '''
        Returns structure's voxel indices within input mask (Structure class).
        Use if a mask was applied to the Dij matrix.
        Returns self.voxel_indices if mask is None.
        '''
        if mask is not None:
            array_in_mask = self.array_1D[mask.voxel_indices]
            return np.nonzero(array_in_mask)[0]
        else:
            return self.voxel_indices

    @staticmethod
    def indices_to_array_1D(voxel_indices, n_voxels_total):
        arr = np.zeros(n_voxels_total, dtype='uint8')
        arr[voxel_indices] = 1
        return arr
        
        