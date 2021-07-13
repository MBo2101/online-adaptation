# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 10:30:55 2021

@author: MBo
"""

import numpy as np
from medpy.io import load

class Image(object):
    
    def __init__(self, path):
        self.__path = path
        self.__array_1D = None

    # Properties
    
    @property
    def path(self, path = ''):
        if path != '':
            self.__path = path
        return self.__path
    @property
    def array_3D(self):
        return self.__array_3D
    @property
    def array_1D(self):
        if self.__array_1D is None:
            self.__array_1D = self.array_3D_to_1D(self.__array_3D)
        return self.__array_1D
    @property
    def n_voxels(self):
        return self.__array_3D.shape[0] * self.__array_3D.shape[1] * self.__array_3D.shape[2]
    @property
    def data_type(self):
        return self.__array_3D.dtype
    @property
    def origin_x(self):
        return self.__header.offset[0]
    @property
    def origin_y(self):
        return self.__header.offset[1]
    @property
    def origin_z(self):
        return self.__header.offset[2]
    @property
    def size_x(self):
        return self.__array_3D.shape[0]
    @property
    def size_y(self):
        return self.__array_3D.shape[1]
    @property
    def size_z(self):
        return self.__array_3D.shape[2]
    @property
    def spacing_x(self):
        return self.__header.spacing[0]
    @property
    def spacing_y(self):
        return self.__header.spacing[1]
    @property
    def spacing_z(self):
        return self.__header.spacing[2]
    @property
    def direction_x(self):
        return self.__header.direction[0]
    @property
    def direction_y(self):
        return self.__header.direction[1]
    @property
    def direction_z(self):
        return self.__header.direction[2]

    # Methods
    
    def load_image_medpy(self):
        self.__array_3D, self.__header = load(self.__path)

    def print_properties(self):
        props = [p for p in dir(Image) if isinstance(getattr(Image,p), property) and hasattr(self,p)]
        for p in props:
            print(p + ' = ' +str(getattr(self, p)))
    
    # Static methods
    
    @staticmethod
    def array_3D_to_1D(array_3D):
        array_temp = np.flip(array_3D, 2)
        array_temp = np.flipud(array_temp)
        array_temp = np.rot90(array_temp, 3, (0,2))
        array_1D = np.reshape(array_temp, -1)
        return array_1D
