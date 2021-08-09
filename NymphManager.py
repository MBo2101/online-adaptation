# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 10:28:25 2021

@author: MBo
"""

import numpy as np

class NymphManager(object):

    def __init__(self):
        self.__functions_list = []
        self.__functions_string = ''
    
    @property
    def functions_list(self):
        return self.__functions_list
    
    @property
    def functions_string(self):
        return self.__functions_string
    
    def clear_functions(self):
        self.__functions_list = []
        self.__functions_string = ''
    
    def add_dose_function(self, obj_type, obj_or_con, weight, parameter, scenario, voxel_indices):
        if obj_type not in [0,1,2,3,4,5,6,7,20,21,22]:
            raise Exception('Unknown objective type.')
        if obj_type == 20:
            raise Exception('Use "minimize_weighted_sum" for this objective type.')
        if obj_type == 21:
            raise Exception('Use "minimize_worst_case" for this objective type.')
        if obj_type == 22:
            raise Exception('Use "minimize_functions_CVaR" for this objective type.')

        parameter = parameter if obj_type >3 else None
        n_voxels = len(voxel_indices)
        
        lst = [np.array([obj_type, obj_or_con]).astype('B'),
               np.array([weight, parameter])[np.array([weight, parameter])!=None].astype('d'),
               np.array([scenario, n_voxels]).astype('Q'),
               np.array(voxel_indices).astype('Q')]
        
        for i in lst: self.__functions_list.append(i)

    def minimize_weighted_sum(self, weight, obj_indices, obj_weights):
        if not len(obj_indices) == len(obj_weights):
            raise Exception('Array sizes do not match.')
        
        obj_type = 20
        obj_or_con = 0

        n_objectives = len(obj_indices)

        lst = [np.array([obj_type, obj_or_con]).astype('B'),
               np.array([weight]).astype('d'),
               np.array([n_objectives]).astype('Q'),
               np.array(obj_indices).astype('Q'),
               np.array(obj_weights).astype('d')]
        
        for i in lst: self.__functions_list.append(i)

    def minimize_worst_case(self, weight, obj_indices, obj_weights, obj_offsets):
        if not len(obj_indices) == len(obj_weights) == len(obj_offsets):
            raise Exception('Array sizes do not match.')

        obj_type = 21
        obj_or_con = 0

        n_objectives = len(obj_indices)

        lst = [np.array([obj_type, obj_or_con]).astype('B'),
               np.array([weight]).astype('d'),
               np.array([n_objectives]).astype('Q'),
               np.array(obj_indices).astype('Q'),
               np.array(obj_weights).astype('d'),
               np.array(obj_offsets).astype('d')]
        
        for i in lst: self.__functions_list.append(i)
        
    def minimize_functions_CVaR(self, weight, alpha, obj_indices):
        
        obj_type = 22
        obj_or_con = 0

        n_objectives = len(obj_indices)

        lst = [np.array([obj_type, obj_or_con]).astype('B'),
               np.array([weight, alpha]).astype('d'),
               np.array([n_objectives]).astype('Q'),
               np.array(obj_indices).astype('Q')]
        
        for i in lst: self.__functions_list.append(i)

    def mimic_dose(self, weight, scenario, voxel_indices, dose_map):
        
        if not len(voxel_indices) == len(dose_map):
            raise Exception('Array sizes do not match.')
        
        self.add_dose_function(4, 0, weight, -123.456, scenario, voxel_indices)
        self.__functions_list.append(dose_map.astype('d'))
        self.add_dose_function(5, 0, weight, -123.456, scenario, voxel_indices)
        self.__functions_list.append(dose_map.astype('d'))
        
    def print_functions(self):
        pass
        
    def write_nymph_dij_file(self, file_path):
        pass
    
    def write_nymph_functions_file(self, file_path):
        pass

    def read_nymph_output(self, path):
        pass
    
