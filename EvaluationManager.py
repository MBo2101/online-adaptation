# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 10:30:55 2021

@author: MBo
"""

import os
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from RTArrays import DoseMap, Structure

class EvaluationManager(object):
    '''
    Keywords arguments:
        structures --> dictionary containing instances of the "Structure" class (dict)
        structures_dir --> path to directory containing structure (mha) files (str)
        structure_files --> paths to individual structure files (list / tuple)
        dose_array --> 1D dose array (numpy.ndarray)
        dose_file --> path to dose (mha) file (str)
    '''
    def __init__(self, **kwargs):
        self.structures = kwargs.get('structures')
        if self.structures is None:
            self.structures_dir = kwargs.get('structures_dir')
            self.structure_files = kwargs.get('structure_files')
            self.load_structures()
        self.dose_array = kwargs.get('dose_array')
        if self.dose_array is None:
            self.dose_file = kwargs.get('dose_file')
        self.DVH_dose_label = 'Dose_Gy'

    def set_dose(self, dose_file):
        self.dose = DoseMap(dose_file)
        self.dose_array = self.dose.array_1D
        self.max_dose = np.max(self.dose_array)

    def load_structures(self):
        if self.structure_files is None:
            if self.structures_dir is not None:
                mha_filenames = [i for i in os.listdir(self.structures_dir) if '.mha' in i]
                mha_filenames.sort()
                self.structure_files = [os.path.join(self.structures_dir, i) for i in mha_filenames]
        if self.structure_files in [None, []]:
            raise Exception('No structures found.')
        structures = [Structure(i) for i in self.structure_files]
        names = [i.name for i in structures]
        self.structures = dict(zip(names, structures))
    
    def get_structure_dose(self, structure_name):
        if self.dose.n_voxels != self.structures[structure_name].n_voxels:
            raise Exception('Dose and structure array sizes do not match.')
        structure_voxels = self.structures[structure_name].voxel_indices
        return self.dose_array[structure_voxels]
    
    def get_mean_dose(self, structure_name):
        return np.average(self.get_structure_dose(structure_name))
    
    def get_max_dose(self, structure_name):
        return np.max(self.get_structure_dose(structure_name))
    
    def get_min_dose(self, structure_name):
        return np.min(self.get_structure_dose(structure_name))
    
    def get_integral_dose(self, structure_name):
        # Unit: Gy*Liter
        # Volume loaded in cm^3
        structure_volume = self.structures[structure_name].volume
        return self.get_mean_dose(structure_name) * structure_volume / 1000
    
    def calculate_DVH(self, num_bins=1000, structures_list=None, save_path=None):
        structures_list = self.structures.keys() if structures_list is None else structures_list
        self.DVH_df = pd.DataFrame()
        for structure_name in structures_list:
            structure_dose = self.get_structure_dose(structure_name)
            hist_range = (0, round(self.max_dose))
            hist_bins  = round(self.max_dose) if num_bins is None else num_bins-1
            differential_DVH, dose_values = np.histogram(structure_dose, hist_bins, hist_range)
            cumulative_DVH = np.zeros(len(differential_DVH)+1)
            index = 0
            cumulative_voxels = 0
            for i in differential_DVH[::-1]:
                cumulative_voxels += i/2
                np.put(cumulative_DVH, index, cumulative_voxels)
                cumulative_voxels += i/2
                index += 1
            np.put(cumulative_DVH, index, cumulative_voxels)
            cumulative_DVH = cumulative_DVH[::-1]/cumulative_voxels
            self.DVH_df[structure_name] = cumulative_DVH
        self.DVH_df.insert(0, self.DVH_dose_label, dose_values)
        if save_path is not None:
            self.DVH_df.to_csv(save_path)
    
    def evaluate_V(self, dose, structure_name):
        DVH_dose = self.DVH_df[self.DVH_dose_label]
        DVH_volume = self.DVH_df[structure_name]
        f = interp1d(DVH_dose, DVH_volume)
        return float(f(dose))

    def evaluate_D(self, volume, structure_name):
        DVH_dose = self.DVH_df[self.DVH_dose_label]
        DVH_volume = self.DVH_df[structure_name]
        f = interp1d(DVH_volume, DVH_dose)
        return float(f(volume))
    