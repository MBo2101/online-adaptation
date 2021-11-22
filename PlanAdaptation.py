# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 22:04:00 2021

@author: MBo
"""

from MoquiManager import MoquiManager
from NymphManager import NymphManager
from DijMatrix import DijMatrix
from SpotMap import SpotMap

class PlanAdaptation(object):

    def __init__(self,
                 image_file,
                 rtplan_file,
                 dose_map,
                 output_dir,
                 machine_name):
        
        self.image_file = image_file
        self.rtplan_file = rtplan_file
        self.dose_map = dose_map
        self.output_dir = output_dir
        self.machine_name = machine_name
        
        moqui = MoquiManager(machine_name)
        moqui.run_simulation(output_dir, rtplan_file, 'dij',
                             image_file = image_file)
        
        dij_total = DijMatrix(npz_dir = output_dir)
        
        # Split Dij matrix based on beamlet subset
        spots = SpotMap(rtplan_file = rtplan_file, machine_name = machine_name)
        included_beamlets, excluded_beamlets = spots.set_beamlet_subset(0.33, 0.1)
        dij_included = DijMatrix(csr_matrix = dij_total.csr_matrix[included_beamlets])
        dij_excluded = DijMatrix(csr_matrix = dij_total.csr_matrix[excluded_beamlets])
        preexisting_dose = dij_excluded.get_dose_array_1D()
        
        nymph = NymphManager(dij_included, preexisting_dose)
        nymph.mimic_dose(dose_map)
        nymph.run_optimization()
        
        spots.adapt_weights(nymph.beamlet_scales)
        spots.write_tramps(output_dir)
        
        moqui.run_simulation(output_dir, rtplan_file, 'dose',
                             image_file = image_file,
                             tramps_dir = output_dir)
        
        