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

    @property
    def path(self, path = ''):
        if path != '':
            self.__path = path
        return self.__path
    
    @property
    def header(self):
        return self.__header
    @property
    def data_type(self):
        return self.__data_type
    @property
    def origin_x(self):
        return self.__origin_x
    @property
    def origin_y(self):
        return self.__origin_y
    @property
    def origin_z(self):
        return self.__origin_z
    @property
    def size_x(self):
        return self.__size_x
    @property
    def size_y(self):
        return self.__size_y
    @property
    def size_z(self):
        return self.__size_z
    @property
    def spacing_x(self):
        return self.__spacing_x
    @property
    def spacing_y(self):
        return self.__spacing_y
    @property
    def spacing_z(self):
        return self.__spacing_z
    @property
    def array_1D(self):
        return self.__array_1D
    @property
    def array_3D(self):
        return self.__array_3D
        
    def load_image_medpy(self):
        self.__array_3D, self.__header = load(self.__path)
        self.__data_type = self.__array_3D.dtype
        self.__origin_x = self.__header.offset[0]
        self.__origin_y = self.__header.offset[1]
        self.__origin_z = self.__header.offset[2]
        self.__size_x = self.__array_3D.shape[0]
        self.__size_y = self.__array_3D.shape[1]
        self.__size_z = self.__array_3D.shape[2]
        self.__spacing_x = self.__header.spacing[0]
        self.__spacing_y = self.__header.spacing[1]
        self.__spacing_z = self.__header.spacing[2]
        
    def array_3D_to_1D(self):
        array_temp = np.flip(self.__array_3D, 2)
        array_temp = np.flipud(array_temp)
        array_temp = np.rot90(array_temp, 3, (0,2))
        self.__array_1D = np.reshape(array_temp, -1)
        return self.__array_1D
