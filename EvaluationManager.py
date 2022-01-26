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
    
    def add_structure(self, structure, name=None):
        '''
        Manually adds structure to structures dictionary.
        '''
        name = structure.name if name is None else name
        self.structures[name] = structure
    
    def get_structure_dose(self, structure_name):
        if self.dose.n_voxels_total != self.structures[structure_name].n_voxels_total:
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
        # Unit: Gy * Liter
        # Volume loaded in cm^3
        structure_volume = self.structures[structure_name].volume
        return self.get_mean_dose(structure_name) * structure_volume / 1000
    
    def calculate_DVH(self, num_bins=1000, structures_list=None, save_path=None):
        structures_list = self.structures.keys() if structures_list is None else structures_list
        self.DVH_data_frame = pd.DataFrame()
        for structure_name in structures_list:
            structure_dose = self.get_structure_dose(structure_name)
            hist_range = (0, round(self.max_dose))
            # If num_bins ist not given --> set bins to steps of approx. 0.1 Gy
            hist_bins  = round(self.max_dose*10)-2 if num_bins is None else num_bins-1
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
            self.DVH_data_frame[structure_name] = cumulative_DVH
        self.DVH_data_frame.insert(0, self.DVH_dose_label, dose_values)
        if save_path is not None:
            if not os.path.exists(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path))
            self.DVH_data_frame.to_csv(save_path)
    
    def load_DVH(self, DVH_csv_file):
        self.DVH_data_frame = pd.read_csv(DVH_csv_file)
    
    def evaluate_V(self, dose, structure_name, prescription=None):
        '''
        Units: Volume [%] / Dose [Gy]
        For given prescription: Dose [%]
        '''
        DVH_dose = self.DVH_data_frame[self.DVH_dose_label]
        DVH_volume = self.DVH_data_frame[structure_name]
        f = interp1d(DVH_dose, DVH_volume)
        if prescription is None:
            return float(f(dose))*100
        else:
            return float(f(dose*prescription/100))*100

    def evaluate_D(self, volume, structure_name, prescription=None):
        '''
        Units: Volume [%] / Dose [Gy]
        For given prescription: Dose [%]
        '''
        DVH_dose = self.DVH_data_frame[self.DVH_dose_label]
        DVH_volume = self.DVH_data_frame[structure_name]
        f = interp1d(DVH_volume, DVH_dose)
        if prescription is None:
            return float(f(volume/100))
        else:
            return float(f(volume/100))/prescription*100
        
    def evaluate_D1cc(self, structure_name):
        '''
        Minimum dose deposited to the most irradiated 1 cubic centimeter of the structure.
        '''
        cc_in_percent = 1/self.structures[structure_name].volume*100
        return self.evaluate_D(cc_in_percent, structure_name)

