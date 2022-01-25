# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 22:04:00 2021

@author: MBo
"""

import os
from MoquiManager import MoquiManager
from NymphManager import NymphManager
from DijMatrix import DijMatrix
from RTArrays import DoseMap, Structure
from time import time

class PlanAdaptation(object):
    '''
    Initialize by passing an instance of the "DicomPlan" class as the argument.
    '''
    def __init__(self, plan):
        
        self.plan = plan
        self.rtplan_file = plan.rtplan_file
        self.time_dij   = 0
        self.time_nymph = 0
        self.time_dose  = 0
        self.time_total = 0
        
    def run(self, image_file, output_dir, mask_file=None, dose_file=None, structures=None):
        '''
        Runs plan adaptation.
        Provide "dose_file" to mimic dose distribution.
        Provide "structures" to apply objectives based on contours.        
        
        Args:
            image_file --> path to input image file (str)
            output_dir --> path to output dicom directory (str)
            mask_file --> path to input Dij mask file (str)
            dose_file --> path to input dose file to mimic (str)
            structures --> dictionary containing structures with the corresponding file paths (dict)
                           dictionary format: {structure_name : file_path}
        '''
        t1 = time()
        
        # Load arrays
        mask = Structure(mask_file) if mask_file is not None else None
        dose = DoseMap(dose_file) if dose_file is not None else None
        
        # Run Dij calculation
        moqui_dij = MoquiManager(self.plan.machine_name)
        moqui_dij.run_simulation(output_dir, self.rtplan_file, 'dij',
                                 image_file = image_file,
                                 masks = [mask_file])
                                 # masks = None)
        
        # Load Dij matrix
        mask_indices = mask.voxel_indices if mask is not None else None
        dij_total = DijMatrix(npz_dir = output_dir, mask_indices = mask_indices)
        
        # Split Dij matrix based on beamlet subset
        self.plan.set_beamlet_subset(0.33, 1)
        self.plan.exclude_zero_weighted_beamlets()
        included_beamlets = self.plan.optimized_beamlets
        excluded_beamlets = self.plan.fixed_beamlets
        dij_included = DijMatrix(csr_matrix = dij_total.csr_matrix[included_beamlets])
        dij_excluded = DijMatrix(csr_matrix = dij_total.csr_matrix[excluded_beamlets])
        preexisting_dose = dij_excluded.get_dose_array_1D()
        
        # Set optimization parameters --> separate function
        self.nymph = NymphManager(dij_included, preexisting_dose)
        self.nymph_set_objectives(mask, dose, structures)
        self.nymph.run_optimization()
        
        # Adapt beamlets weights
        self.beamlet_scales = self.nymph.beamlet_scales
        self.plan.adapt_weights(self.beamlet_scales)
        self.plan.write_tramps(output_dir)
        
        # Run dose calculation
        moqui_dose = MoquiManager(self.plan.machine_name)
        moqui_dose.run_simulation(output_dir, self.rtplan_file, 'dose',
                                  image_file = image_file,
                                  tramps_dir = output_dir,
                                  random_seed = 500)
        t2 = time()
        
        self.time_dij   = moqui_dij.sim_time
        self.time_nymph = self.nymph.opt_time
        self.time_dose  = moqui_dose.sim_time
        self.time_total = t2-t1
        self.log_dij  = moqui_dij.log
        self.log_opt  = self.nymph.log
        self.log_dose = moqui_dose.log
        self.write_log_file(output_dir)
    
    def nymph_set_objectives(self, mask, dose, structures):
        '''
        Separate function to set Nymph's objectives.
        '''
        if dose is not None:
            self.nymph.mimic_dose(dose.get_array_in_mask(mask), weight = 1)
        
        if structures is not None:
            HR_CTV       = Structure(structures['HR_CTV']).get_indices_in_mask(mask)
            LR_CTV       = Structure(structures['LR_CTV_uniform']).get_indices_in_mask(mask)
            # larynx       = Structure(structures['Larynx']).get_indices_in_mask(mask)
            # parotid_L    = Structure(structures['Parotid_L']).get_indices_in_mask(mask)
            # parotid_R    = Structure(structures['Parotid_R']).get_indices_in_mask(mask)
            # constrictors = Structure(structures['Constrictors']).get_indices_in_mask(mask)
            # spinal_cord  = Structure(structures['SpinalCord']).get_indices_in_mask(mask)
            # brainstem    = Structure(structures['Brainstem']).get_indices_in_mask(mask)
            # shell        = Structure(structures['conformality_shell']).get_indices_in_mask(mask)
            # 
            # Adding objectives
            self.nymph.set_mean_dose(HR_CTV, 70, label = 'HR_CTV', weight = 1/2)
            self.nymph.set_mean_dose(LR_CTV, 57, label = 'LR_CTV', weight = 1/2)
            # self.nymph.add_dose_function(3, 0, 0.01, None, 0, shell, label = 'conformality_shell')
            # self.nymph.add_dose_function(1, 0, 0.01, None, 0, shell, label = 'conformality_shell')
        
    def write_log_file(self, output_dir):
        separator = '-'*200
        self.log = 'Plan adaptation overview'
        self.log += '\ntime total = {} sec ≈ {} min'.format(self.time_total, round(self.time_total/60, 2))
        self.log += self.plan.beamlet_subset_info
        self.log += self.plan.adaptation_info
        self.log += '\n\n'+separator+'\n'
        self.log += '\n(1) Moqui dij calculation'
        self.log += '\ntime = {} sec ≈ {} min\n'.format(self.time_dij, round(self.time_dij/60, 2))
        self.log += self.log_dij
        self.log += '\n'+separator+'\n'
        self.log += '\n(2) Nymph optimization'
        self.log += '\ntime = {} sec ≈ {} min\n'.format(self.time_nymph, round(self.time_nymph/60, 2))
        self.log += self.log_opt
        self.log += '\n'+separator+'\n'
        self.log += '\n(3) Moqui dose calculation'
        self.log += '\ntime = {} sec ≈ {} min\n'.format(self.time_dose, round(self.time_dose/60, 2))
        self.log += self.log_dose
        log_path = os.path.join(output_dir, 'log.txt')
        with open(log_path, 'w') as f:
            f.write(self.log)
            