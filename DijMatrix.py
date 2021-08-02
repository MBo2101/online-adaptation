# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 10:28:25 2021

@author: MBo
"""

import numpy as np
from scipy import sparse

class DijMatrix(object):

    def __init__(self, path):
        self.__path = path
        self.__csr_matrix = None
        self.__n_beamlets = None
        self.__n_voxels = None
        self.__dose_array_1D = None
        self.__dose_array_3D = None

    # Properties
        
    @property
    def path(self, path = ''):
        if path != '':
            self.__path = path
        return self.__path
    
    @property
    def csr_matrix(self):
        return self.__csr_matrix
    
    @property
    def n_beamlets(self):
        if self.__csr_matrix is not None:
            return np.shape(self.__csr_matrix)[0]
    
    @property
    def n_voxels(self):
        if self.__csr_matrix is not None:
            return np.shape(self.__csr_matrix)[1]
        
    @property
    def dose_array_1D(self):
        return self.__dose_array_1D
    
    @property
    def dose_array_3D(self):
        return self.__dose_array_3D
    
    # Methods
    
    def load_npz(self):
        self.__csr_matrix = sparse.load_npz(self.__path)

    def get_dose_array_1D(self):
        if self.__csr_matrix is None:
            self.load_npz()
        self.__dose_array_1D = self.__csr_matrix.sum(axis=0).A1
        return self.__dose_array_1D
    
    def get_dose_array_3D(self, nx, ny, nz):
        if self.__dose_array_1D is None:
            self.get_dose_array_1D()
        self.__dose_array_3D = self.dose_array_1D.reshape(nz, ny, nx)
        return self.__dose_array_3D
        
    def print_properties(self):
        props = [p for p in dir(DijMatrix) if isinstance(getattr(DijMatrix,p), property) and hasattr(self,p)]
        for p in props:
            print(p + ' = ' +str(getattr(self, p)))
            
            
