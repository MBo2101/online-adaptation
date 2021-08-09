# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 10:28:25 2021

@author: MBo
"""

import numpy as np
from scipy import sparse
from RTArrays import RTArray

class DijMatrix(object):

    def __init__(self, **kwargs):
        
        self.__path = kwargs.get('path')
        self.__csr_matrix = kwargs.get('csr_matrix')
        self.__scale = 1 if kwargs.get('scale') is None else kwargs.get('scale')

        if self.__csr_matrix is None and self.__path is not None:
            self.__csr_matrix = self.get_csr_matrix_from_npz()
            if self.__scale != 1:
                self.__csr_matrix = self.__csr_matrix * self.__scale

        if self.__csr_matrix is not None:
            self.__n_beamlets = np.shape(self.__csr_matrix)[0]
            self.__n_voxels = np.shape(self.__csr_matrix)[1]
            self.__nnz_elements = self.__csr_matrix.nnz
            self.__nnz_beamlets, self.__nnz_voxels = self.__csr_matrix.nonzero()
            self.__nnz_values = self.__csr_matrix.data

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
    def scale(self):
        return self.__scale
    @property
    def n_beamlets(self):
        return self.__n_beamlets
    @property
    def n_voxels(self):
        return self.__n_voxels
    @property
    def nnz_elements(self):
        return self.__nnz_elements
    @property
    def nnz_voxels(self):
        return self.__nnz_voxels
    @property
    def nnz_beamlets(self):
        return self.__nnz_beamlets
    @property
    def nnz_values(self):
        return self.__nnz_values

    # Methods
    
    def get_csr_matrix_from_npz(self):
        return sparse.load_npz(self.__path)

    def get_dense_dij_matrix(self):
        if self.__csr_matrix is None:
            self.load_npz()
        return self.__csr_matrix.toarray()

    def get_dose_array_1D(self):
        if self.__csr_matrix is None:
            self.load_npz()
        return self.__csr_matrix.sum(axis=0).A1
    
    def get_dose_array_3D(self, nx, ny, nz):
        temp = self.get_dose_array_1D()
        return RTArray.array_1D_to_3D(temp, nx, ny, nz)
        
    def print_properties(self):
        c = self.__class__
        props = [p for p in dir(c) if isinstance(getattr(c,p), property) and hasattr(self,p)]
        for p in props:
            print(p + ' = ' +str(getattr(self, p)))
            