# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 10:30:55 2021

@author: MBo
"""

import numpy as np
from medpy.io import load

'''
Superclass
'''

class RTArray(object):
    
    def __init__(self, path):
        self.__path = path
        self.__ndarray = None
        self.__array_1D = None
        self.__base_value = None

    # Properties
    
    @property
    def path(self, path = ''):
        if path != '':
            self.__path = path
        return self.__path
    @property
    def ndarray(self):
        return self.__ndarray
    @property
    def array_1D(self):
        if self.__array_1D is None:
            if self.__ndarray is not None:
                if np.ndim(self.__ndarray) == 3:
                    return self.array_3D_to_1D(self.__ndarray)
    @property
    def n_voxels(self):
        if self.__ndarray is not None:
            return self.__ndarray.shape[0] * self.__ndarray.shape[1] * self.__ndarray.shape[2]
        else:    
            return self.__n_voxels
    @property
    def n_dim(self):
        if self.__ndarray is not None:
            return np.ndim(self.__ndarray)
        else:
            return self.__ndim
    @property
    def data_type(self):
        if self.__ndarray is not None:
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
    
    # Setters
    
    @path.setter
    def path(self, var):
        self.__path = var
        
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
        self.__ndarray, self.__header = load(self.__path)
            
    def load_header(self):
        '''
        Does not store array --> use to save memory
        Seems to be faster without explicitly deleting 'temp'?
        '''     
        temp, self.__header = load(self.__path)
        self.__n_voxels = temp.shape[0] * temp.shape[1] * temp.shape[2]
        self.__ndim = np.ndim(temp)
        self.__data_type = temp.dtype
        self.__size_x = temp.shape[0]
        self.__size_y = temp.shape[1]
        self.__size_z = temp.shape[2]

    def print_properties(self):
        props = [p for p in dir(RTArray) if isinstance(getattr(RTArray,p), property) and hasattr(self,p)]
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

'''
Subclasses
'''

class PatientImage(RTArray):
    def __init__(self, path):
        super().__init__(path)

class ImageCT(PatientImage):
    def __init__(self, path):
        super().__init__(path)
        self.base_value = -1000.

class ImageCBCT(PatientImage):
    def __init__(self, path):
        super().__init__(path)
        self.base_value = -1000.

class ImageMRI(PatientImage):
    def __init__(self, path):
        super().__init__(path)

class DoseMap(RTArray):
    def __init__(self, path):
        super().__init__(path)
        self.base_value = 0.

class Structure(RTArray):
    def __init__(self, path):
        super().__init__(path)
        self.base_value = 0.

class VectorField(RTArray):
    def __init__(self, path):
        super().__init__(path)
        self.base_value = 0.

class TranslationVF(VectorField):
    def __init__(self, path):
        super().__init__(path)

class RigidVF(VectorField):
    def __init__(self, path):
        super().__init__(path)

class BSplineVF(VectorField):
    def __init__(self, path):
        super().__init__(path)

