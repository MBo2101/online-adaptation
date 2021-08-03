# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 10:30:55 2021

@author: MBo
"""

from RTArrays.RTArray import RTArray

class VectorField(RTArray):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.base_value = 0.

class TranslationVF(VectorField):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    @property
    def shift_x(self):
        return self.get_shifts()[0]
    @property
    def shift_y(self):
        return self.get_shifts()[1]
    @property
    def shift_z(self):
        return self.get_shifts()[2]
        
    def get_shifts(self):
        if self.ndarray is None:
            self.load_file()
        # Assuming each voxel receives the same shift:
        voxel = self.ndarray[0][0][0]
                
        return voxel[0], voxel[1], voxel[2]

class RigidVF(VectorField):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class BSplineVF(VectorField):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

