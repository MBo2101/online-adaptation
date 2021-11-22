# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 10:28:25 2021

@author: MBo
"""

import numpy as np
import os
from scipy import sparse
from RTArrays import RTArray

class DijMatrix(object):
    '''
    Keywords arguments:
        csr_matrix --> CSR matrix (scipy.sparse.csr.csr_matrix)
        npz_file --> path to npz file (string)
        npz_dir --> path to directory containing npz files (string)
        scale --> scale to apply to Dij matrix (int / float)
    '''
    def __init__(self, **kwargs):
        self.__csr_matrix = kwargs.get('csr_matrix')
        self.__npz_file = kwargs.get('npz_file')
        self.__npz_dir = kwargs.get('npz_dir')
        self.__scale = kwargs.get('scale')

        if self.__npz_file is not None:
            self.__csr_matrix = self.get_csr_matrix_from_file()
        if self.__npz_dir is not None:
            self.__csr_matrix = self.get_csr_matrix_from_dir()
        if self.__scale is not None:
            self.__csr_matrix = self.__csr_matrix * self.__scale
        if self.__csr_matrix is not None:
            self.__n_beamlets = np.shape(self.__csr_matrix)[0]
            self.__n_voxels = np.shape(self.__csr_matrix)[1]
            self.__nnz_elements = self.__csr_matrix.nnz
            self.__nnz_beamlets, self.__nnz_voxels = self.__csr_matrix.nonzero()
            self.__nnz_values = self.__csr_matrix.data

    # Properties
        
    @property
    def npz_file(self, npz_file = ''):
        if npz_file != '':
            self.__npz_file = npz_file
        return self.__npz_file
    @property
    def npz_dir(self, npz_dir = ''):
        if npz_dir != '':
            self.__npz_dir = npz_dir
        return self.__npz_dir
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
    
    def get_csr_matrix_from_file(self):
        '''
        Returns CSR matrix from single .npz file.
        '''
        return sparse.load_npz(self.__npz_file)
    
    def get_csr_matrix_from_dir(self):
        '''
        Returns CSR matrix (stacked) for all .npz files in directory.
        '''
        npz_filenames = [i for i in os.listdir(self.__npz_dir) if '.npz' in i]
        npz_filepaths = [os.path.join(self.__npz_dir, i) for i in npz_filenames]
        csr_matrices = [sparse.load_npz(i) for i in npz_filepaths]
        csr_matrix = sparse.vstack(csr_matrices)
        return csr_matrix

    def get_dense_dij_matrix(self):
        if self.__csr_matrix is None:
            self.load_npz()
        return self.__csr_matrix.toarray()

    def get_dose_array_1D(self):
        if self.__csr_matrix is None:
            self.load_npz()
        return self.__csr_matrix.sum(axis=0).A1
    
    def get_dose_array_3D(self, nx, ny, nz):
        arr = self.get_dose_array_1D()
        return RTArray.array_1D_to_3D(arr, nx, ny, nz)
        
    def print_properties(self):
        c = self.__class__
        props = [p for p in dir(c) if isinstance(getattr(c,p), property) and hasattr(self,p)]
        for p in props:
            print(p + ' = ' +str(getattr(self, p)))
            