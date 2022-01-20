# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 10:28:25 2021

@author: MBo
"""

import numpy as np
import struct
import subprocess
from time import time

class NymphManager(object):

    def __init__(self, dij_matrix, preexisting_dose=None):
        self.dir_path = '/shared/build/nymph'
        self.exe_path = '/shared/build/nymph/nymph'
        self.dij_path = '/shared/build/nymph/dij.bin'
        self.functions_path = '/shared/build/nymph/functions.bin'
        self.output_path = '/shared/build/nymph/x.bin'
        self.functions_list = []
        self.functions_string = ''
        self.log = ''
        self.opt_time = 0
        self.n_voxels = dij_matrix.n_voxels
        self.n_beamlets = dij_matrix.n_beamlets
        self.nnz_elements = dij_matrix.nnz_elements
        self.nnz_voxels = dij_matrix.nnz_voxels
        self.nnz_beamlets = dij_matrix.nnz_beamlets
        self.nnz_values = dij_matrix.nnz_values
        self.preexisting_dose = preexisting_dose
        self.beamlet_scales = None
        self.set_dij_list()

    # Methods: general
    
    def print_help(self):
        get_help = subprocess.Popen([self.exe_path, '-h'], stdout=subprocess.PIPE)
        get_help.wait()
        print(get_help.stdout.read().decode('utf-8'))
    
    def print_functions(self):
        # TODO: finish function to document used objectives/constraints
        pass
    
    def set_dij_list(self):
        self.dij_list = []
        self.dij_list.append(np.array([0], dtype='u8'))
        self.dij_list.append(np.array([0], dtype='u8'))
        self.dij_list.append(np.array([self.n_voxels], dtype='u8'))
        self.dij_list.append(np.array([self.n_beamlets], dtype='u8'))
        self.dij_list.append(np.array([self.nnz_elements], dtype='u8'))
        self.dij_list.append(self.nnz_voxels.astype('u8'))
        self.dij_list.append(self.nnz_beamlets.astype('u8'))
        self.dij_list.append(self.nnz_values.astype('f'))
        if self.preexisting_dose is not None:
            if np.any(self.preexisting_dose):
                self.dij_list.append(np.array([0], dtype='u8'))
                self.dij_list.append(np.array([10], dtype='u8'))
                self.dij_list.append(np.array([len(self.preexisting_dose)], dtype='u8'))
                self.dij_list.append(self.preexisting_dose.astype('d'))
    
    def write_dij_input(self, file_path=None):
        '''
        Writes Dij matrix to binary file for Nymph input.
        '''
        file_path = self.dij_path if file_path is None else file_path
        with open(file_path, 'bw') as f:
            for i in self.dij_list:
                f.write(i.tobytes())
    
    def write_functions_input(self, file_path=None):
        '''
        Writes optimization functions to binary file for Nymph input.
        '''
        file_path = self.functions_path if file_path is None else file_path
        with open(file_path, 'bw') as f:
            for i in self.functions_list:
                f.write(i.tobytes())
    
    def run_optimization(self):
        print('Writing Nymph files')
        self.write_dij_input()
        self.write_functions_input()
        print('Starting Nymph optimization:')
        start_opt = time()
        run = subprocess.Popen([self.exe_path,
                                '-dij', self.dij_path,
                                '-f', self.functions_path,
                                '-o', self.output_path,
                                '-s'], cwd=self.dir_path, stdout=subprocess.PIPE)
        while True:
            output_line = run.stdout.readline().decode('utf-8')
            if run.poll() is not None: break
            if output_line:
                self.log += output_line
                print(output_line.strip())
        end_opt = time()
        self.opt_time += end_opt - start_opt
        if run.poll() == 0:
            self.beamlet_scales = self.read_output()
            return True
        else:
            raise Exception('Nymph error')
        
    def read_output(self, file_path=None):
        '''
        Reads Nymph's output file and returns optimized beamlet scales.
        '''
        file_path = self.output_path if file_path is None else file_path
        with open(file_path, 'rb') as f:
            file_bytes = f.read()
            n_beamlets = (struct.unpack('Q', file_bytes[0:8]))[0] # 1st 8 bytes (uint64) = number of elements
            scaling_factors = struct.unpack('d'*n_beamlets, file_bytes[8:])
        return np.array(scaling_factors, dtype='d')

    # Methods: defining functions
    
    def clear_functions(self):
        self.functions_list = []
        self.functions_string = ''
    
    def add_dose_function(self, obj_type, obj_or_con, weight, parameter, scenario, voxel_indices):
        if obj_type not in [0,1,2,3,4,5,6,7,20,21,22]: raise Exception('Unknown objective type.')
        if obj_type == 20: raise Exception('Use "minimize_weighted_sum" for this objective type.')
        if obj_type == 21: raise Exception('Use "minimize_worst_case" for this objective type.')
        if obj_type == 22: raise Exception('Use "minimize_functions_CVaR" for this objective type.')
        n_voxels = len(voxel_indices)
        self.functions_list.append(np.array([obj_type], dtype='B'))
        self.functions_list.append(np.array([obj_or_con], dtype='B'))
        self.functions_list.append(np.array([weight], dtype='d'))
        if obj_type>3: self.functions_list.append(np.array([parameter], dtype='d'))
        self.functions_list.append(np.array([scenario], dtype='Q'))
        self.functions_list.append(np.array([n_voxels], dtype='Q'))
        self.functions_list.append(voxel_indices.astype('Q'))

    def minimize_weighted_sum(self, weight, obj_indices, obj_weights):
        if not len(obj_indices) == len(obj_weights):
            raise Exception('Array sizes do not match.')
        obj_type = 20
        obj_or_con = 0
        n_objectives = len(obj_indices)
        self.functions_list.append(np.array([obj_type], dtype='B'))
        self.functions_list.append(np.array([obj_or_con], dtype='B'))
        self.functions_list.append(np.array([weight], dtype='d'))
        self.functions_list.append(np.array([n_objectives], dtype='Q'))
        self.functions_list.append(obj_indices.astype('Q'))
        self.functions_list.append(obj_weights.astype('d'))

    def minimize_worst_case(self, weight, obj_indices, obj_weights, obj_offsets):
        if not len(obj_indices) == len(obj_weights) == len(obj_offsets):
            raise Exception('Array sizes do not match.')
        obj_type = 21
        obj_or_con = 0
        n_objectives = len(obj_indices)
        self.functions_list.append(np.array([obj_type], dtype='B'))
        self.functions_list.append(np.array([obj_or_con], dtype='B'))
        self.functions_list.append(np.array([weight], dtype='d'))
        self.functions_list.append(np.array([n_objectives], dtype='Q'))
        self.functions_list.append(obj_indices.astype('Q'))
        self.functions_list.append(obj_weights.astype('d'))
        self.functions_list.append(obj_offsets.astype('d'))
        
    def minimize_functions_CVaR(self, weight, alpha, obj_indices):
        obj_type = 22
        obj_or_con = 0
        n_objectives = len(obj_indices)        
        self.functions_list.append(np.array([obj_type], dtype='B'))
        self.functions_list.append(np.array([obj_or_con], dtype='B'))
        self.functions_list.append(np.array([weight], dtype='d'))
        self.functions_list.append(np.array([alpha], dtype='d'))
        self.functions_list.append(np.array([n_objectives], dtype='Q'))
        self.functions_list.append(obj_indices.astype('Q'))

    def mimic_dose(self, dose_array, weight=1, scenario=0, voxel_indices=None):
        '''
        Sets objective to mimic input dose distribution.
        '''
        if voxel_indices is None:
            voxel_indices = np.arange(0, len(dose_array))
        if not len(voxel_indices) == len(dose_array):
            raise Exception('Array sizes do not match.')
        self.add_dose_function(4, 0, weight, -123.456, scenario, voxel_indices)
        self.functions_list.append(dose_array.astype('d'))
        self.add_dose_function(5, 0, weight, -123.456, scenario, voxel_indices)
        self.functions_list.append(dose_array.astype('d'))
        
    def set_mean_dose(self, voxel_indices, dose, weight=1, scenario=0):
        '''
        Sets objective to get a homogeneous dose distribution for input voxel indices.
        '''
        self.add_dose_function(4, 0, weight, dose, scenario, voxel_indices)
        self.add_dose_function(5, 0, weight, dose, scenario, voxel_indices)

#%%
# Nymph instructions
'''
Usage: ./nymph -dij dij.bin -f functions.bin -o x.bin

dij.bin is a binary file of concatenated sparse D matrices, (optionally)
followed by preexisting dose vectors. If no preexisting dose vectors are
supplied, it is assumed that the patient has not received any dose yet.
Each D matrix is defined as:
    scenario (uint64)
    0 (uint64)
    n_voxels (uint64)
    n_beamlets (uint64)
    nnz (number of nonzero elements) (uint64)
    i*nnz (uint64)
    j*nnz (uint64)
    value*nnz (single precision float)
The file first contains all i values, then all j values, and then
all d(i,j) values. The nonzero values must be sorted on (i,j).
Each D matrix should have the same number of beamlets, but every
scenario can have its own set of voxels.
For preexisting dose, the syntax is:
    scenario (uint64)
    10 (uint64)
    n_voxels (uint64)
    preexisting_dose*n_voxels (double)

functions.bin is a binary file of concatenated objectives,
where each objective is defined as:
type (byte):  0 maximize minimum dose
              1 minimize maximum dose
              2 maximize mean dose
              3 minimize mean dose
              4 minimize mean underdose (parameter: dose threshold)
              5 minimize mean overdose (parameter: dose threshold)
              6 maximize lower CVaR (parameter: alpha in [0,1])
              7 minimize upper CVaR (parameter: alpha in [0,1])
             20 minimize weighted sum of other functions (parameter: weights)
             21 minimize worst case of other functions (parameters: weights and offsets)
             22 minimize CVaR of other functions (parameter: alpha in [0,1])
objective (byte): 0 for objective, 1 for constraint
objective weight or constraint bound (double)
[optional parameter (double)]
For dose-based objectives, the remaining fields are:
  scenario (uint64): indicates which D matrix it applies to
  number of voxels (uint64)
  voxel indices, starting at 0 for each scenario (uint64)
For function-based objectives, the remaining fields are:
  number of objectives it applies to (uint64)
  indices of objectives, starting at 0 (uint64)
  [optional weights of objectives (double)]
  [optional offsets of objectives (double), the weight is not applied to the offset]

x.bin is a binary file with the following elements:
n (number of elements) (uint64)
x_j*n (the n elements of x) (double)
'''

# CVaR: for alpha = 1% parameter is 0.01

# Dose mimicking:
# -123.456 as parameter unlocks the hidden feature - use together with objective types 4 and 5

# Robust optimization approaches: worst case (minimax) OR probabilistic/stocastic

# Worst case approaches: composite worst case (20 in each scenario and 21 overall function) 
#                OR objective-wise worst case (21 for each objective including all scenarios and 20 as overall function) 

# BRAM: both approaches should work fine, for objective-wise worst case might not need 20 as overall function

