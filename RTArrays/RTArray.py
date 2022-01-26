# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 10:30:55 2021

@author: MBo
"""

import os
import numpy as np
from medpy.io import load, save
# Consider using SimpleITK instead of MedPy

'''
Superclass
'''

class RTArray(object):
    '''
    Keywords arguments:
        path --> file path (str)
        header --> image header (medpy.io.header.Header)
        ndarray --> 3D or 4D array (numpy.ndarray)
        array_1D --> 1D array (numpy.ndarray)
        skip_load --> option to skip load when 'path' is provided (bool)
        header_only --> option to not save array into memory (bool)
    '''
    def __init__(self, arg=None, **kwargs):
        
        self.__path = kwargs.get('path')
        self.__header = kwargs.get('header')
        self.__ndarray = kwargs.get('ndarray')
        self.__array_1D = kwargs.get('array_1D')
        self.__skip_load = kwargs.get('skip_load')
        self.__header_only = kwargs.get('header_only')
        
        if type(arg) is str:
            if self.__path is None:
                self.__path = arg
        elif type(arg) is np.ndarray:
            if np.ndim(arg) == 1:
                self.__array_1D = arg
            else:
                self.__ndarray = arg
                    
        if self.__path is not None:
            self.__filename = os.path.basename(self.__path)
            if self.__skip_load is not True:
                if self.__header_only is True:
                    self.load_header()
                else:
                    self.load_file()
        
        if self.__array_1D is None:
            if self.__ndarray is not None:
                if np.ndim(self.__ndarray) == 3:
                    self.__array_1D = self.array_3D_to_1D(self.__ndarray)

    def __check_file(self):
        if not os.path.isfile(self.__path):
            raise Exception('File does not exist.')
    
    # Properties
    
    @property
    def path(self, path = ''):
        if path != '':
            self.__path = path
        return self.__path
    @property
    def filename(self):
        if self.__filename is not None:
            return self.__filename
    @property
    def name(self):
        if self.__filename is not None:
            self.__name =  os.path.splitext(self.__filename)[0]
        return self.__name
    @property
    def header(self):
        return self.__header
    @property
    def ndarray(self):
        return self.__ndarray
    @property
    def array_1D(self):
        return self.__array_1D
    @property
    def n_dim(self):
        if self.__ndarray is not None:
            return self.__ndarray.ndim
        else:
            return self.__ndim
    @property
    def n_voxels_total(self):
        if self.__array_1D is not None:
            return self.__array_1D.size
        elif self.__ndarray is not None:
            if self.__ndarray.ndim == 3:
                return self.__ndarray.size
            elif self.__ndarray.ndim == 4:
                return self.__ndarray.size/3
        else:
            return self.__n_voxels_total
    @property
    def data_type(self):
        if self.__array_1D is not None:
            return self.__array_1D.dtype
        elif self.__ndarray is not None:
            return self.__ndarray.dtype
        else:
            return self.__data_type
    @property
    def size_x(self):
        if self.__ndarray is not None:
            return self.__ndarray.shape[0]
        else:
            return self.__size_x
    @property
    def size_y(self):
        if self.__ndarray is not None:
            return self.__ndarray.shape[1]
        else:
            return self.__size_y
    @property
    def size_z(self):
        if self.__ndarray is not None:
            return self.__ndarray.shape[2]
        else:
            return self.__size_z
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
    @property
    def base_value(self):
        return self.__base_value
    @property
    def voxel_volume(self):
        # In cm^3
        return self.__header.spacing[0] * self.__header.spacing[1] * self.__header.spacing[2] / 10**3
    
    # Setters
    
    @path.setter
    def path(self, var):
        self.__path = var

    @header.setter
    def header(self, var):
        self.__header = var

    @ndarray.setter
    def ndarray(self, var):
        self.__ndarray = var
    
    @array_1D.setter
    def array_1D(self, var):
        self.__array_1D = var
    
    @base_value.setter
    def base_value(self, var):
        self.__base_value = var

    # Methods
    
    def load_file(self):
        self.__check_file()
        self.__ndarray, self.__header = load(self.__path)
            
    def load_header(self):
        '''
        Does not store array --> use to save memory
        '''     
        self.__check_file()
        temp, self.__header = load(self.__path)
        self.__ndim = np.ndim(temp)
        self.__n_voxels_total = temp.shape[0] * temp.shape[1] * temp.shape[2]
        self.__data_type = temp.dtype
        self.__size_x = temp.shape[0]
        self.__size_y = temp.shape[1]
        self.__size_z = temp.shape[2]

    def save_file(self, path):
        save(self.__ndarray, path, hdr=self.__header)

    def print_properties(self):
        c = self.__class__
        props = [p for p in dir(c) if isinstance(getattr(c,p), property) and hasattr(self,p)]
        for p in props:
            print(p + ' = ' +str(getattr(self, p)))
    
    # Static methods

    @staticmethod
    def array_1D_to_3D(array_1D, nx, ny, nz):
        arr = array_1D.reshape(nz, ny, nx)
        arr = np.swapaxes(arr, 0, 2)
        return arr
        
    @staticmethod
    def array_3D_to_1D(array_3D):
        arr = np.swapaxes(array_3D, 0, 2)
        arr = arr.ravel()
        return arr

