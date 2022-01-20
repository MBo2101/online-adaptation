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
        factor --> factor to apply to Dij matrix (int / float)
        mask_indices --> indices of mask to apply to Dij matrix (numpy.array)
    '''
    def __init__(self, **kwargs):
        self.csr_matrix = kwargs.get('csr_matrix')
        self.npz_file = kwargs.get('npz_file')
        self.npz_dir = kwargs.get('npz_dir')
        self.factor = kwargs.get('factor')
        self.mask_indices = kwargs.get('mask_indices')
        if self.npz_file is not None: self.csr_matrix = self.get_csr_matrix_from_file()
        if self.npz_dir is not None: self.csr_matrix = self.get_csr_matrix_from_dir()
        if self.factor is not None: self.csr_matrix = self.csr_matrix * self.factor
        if self.mask_indices is not None: self.csr_matrix = self.csr_matrix[:,self.mask_indices]
        if self.csr_matrix is not None:
            self.csr_matrix.eliminate_zeros()
            self.n_beamlets = np.shape(self.csr_matrix)[0]
            self.n_voxels = np.shape(self.csr_matrix)[1]
            self.nnz_elements = self.csr_matrix.nnz
            self.nnz_beamlets, self.nnz_voxels = self.csr_matrix.nonzero()
            self.nnz_values = self.csr_matrix.data
    
    def get_csr_matrix_from_file(self):
        '''
        Returns CSR matrix from single npz file.
        '''
        return sparse.load_npz(self.npz_file)
    
    def get_csr_matrix_from_dir(self):
        '''
        Returns CSR matrix (stacked) for all npz files in directory.
        '''
        npz_filenames = [i for i in os.listdir(self.npz_dir) if '.npz' in i]
        npz_filepaths = [os.path.join(self.npz_dir, i) for i in npz_filenames]
        csr_matrices = [sparse.load_npz(i) for i in npz_filepaths]
        csr_matrix = sparse.vstack(csr_matrices, format='csr')
        return csr_matrix
    
    def get_dense_dij_matrix(self):
        '''
        Returns dense Dij matrix.
        '''
        return self.csr_matrix.toarray()

    def get_dose_array_1D(self):
        '''
        Returns dose array (1D) corresponding to the Dij matrix.
        '''
        return self.csr_matrix.sum(axis=0).A1
    
    def get_dose_array_3D(self, nx, ny, nz):
        '''
        Returns dose array (3D) corresponding to the Dij matrix.
        '''
        arr = self.get_dose_array_1D()
        return RTArray.array_1D_to_3D(arr, nx, ny, nz)
    
    def apply_beamlet_scales(self, beamlet_scales):
        '''
        Returns CSR matrix after applying scales to beamlets.
        '''
        arr = np.array(beamlet_scales)
        coo_matrix = self.csr_matrix.multiply(arr.reshape(len(beamlet_scales), 1))
        return coo_matrix.tocsr()
        