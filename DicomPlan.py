# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 10:28:25 2021

@author: MBo
"""

import numpy as np
import os
import pydicom
from BeamModel import BeamModel

class DicomPlan(object):
    '''
    Loads IMPT plan from a DICOM directory or RT plan file.
    Use to load and modify the plan's spot map.
    '''
    def __init__(self, **kwargs):
        self.dicom_dir    = kwargs.get('dicom_dir')
        self.rtplan_file  = kwargs.get('rtplan_file')
        self.machine_name = kwargs.get('machine_name')
        self.load_pydicom_ds()
        self.load_dicom_spot_map()
        self.convert_spot_map()

    def load_pydicom_ds(self):
        if self.rtplan_file is not None: self.dicom_dir = os.path.dirname(self.rtplan_file)
        elif self.dicom_dir is not None:
            lst = []
            for file in os.listdir(self.dicom_dir):
                if '.dcm' in file:
                    file_path = os.path.join(self.dicom_dir, file)
                    ds = pydicom.dcmread(file_path)
                    if 'IonBeamSequence' in ds:
                        lst.append(file_path)
            if len(lst) == 0: raise Exception('No RT plan file found.')
            elif len(lst) > 1: raise Exception('Multiple RT plan files found. '\
                                               'Use argument "rtplan_file" instead.'\
                                               '\n\n{}'.format(lst))
            self.rtplan_file = lst[0]
        self.pydicom_ds = pydicom.dcmread(self.rtplan_file)
    
    def load_dicom_spot_map(self):
        energies      = np.array([], dtype='d')
        x_coordinates = np.array([], dtype='d')
        y_coordinates = np.array([], dtype='d')
        weights_mu    = np.array([], dtype='d')
        labels        = np.array([], dtype='U')
        for beam_index in range(len(self.pydicom_ds.IonBeamSequence)):
            beam_name = self.pydicom_ds.IonBeamSequence[beam_index].BeamName
            if beam_name == 'Setup': continue
            ICPS = self.pydicom_ds.IonBeamSequence[beam_index].IonControlPointSequence
            n_layers = int(len(ICPS)/2)
            for layer_index in range(n_layers):
                n_spots_in_layer = int(len(ICPS[layer_index*2].ScanSpotPositionMap)/2)
                for beamlet_index in range(n_spots_in_layer):
                    energy = ICPS[layer_index*2].NominalBeamEnergy
                    x_coordinate = ICPS[layer_index*2].ScanSpotPositionMap[beamlet_index*2]
                    y_coordinate = ICPS[layer_index*2].ScanSpotPositionMap[beamlet_index*2+1]
                    if n_spots_in_layer == 1: weight_mu = ICPS[layer_index*2].ScanSpotMetersetWeights
                    else: weight_mu = ICPS[layer_index*2].ScanSpotMetersetWeights[beamlet_index]
                    energies      = np.append(energies,      energy)
                    x_coordinates = np.append(x_coordinates, x_coordinate)
                    y_coordinates = np.append(y_coordinates, y_coordinate)
                    weights_mu    = np.append(weights_mu,    weight_mu)
                    labels        = np.append(labels,        beam_name)
        self.energies      = energies
        self.x_coordinates = x_coordinates
        self.y_coordinates = y_coordinates
        self.weights_MU    = weights_mu
        self.labels        = labels
    
    def convert_spot_map(self):
        self.n_fractions = int(self.pydicom_ds.FractionGroupSequence[0].NumberOfFractionsPlanned)
        # Converting MU to Gp
        if self.machine_name is None:
            self.machine_name = self.pydicom_ds.IonBeamSequence[0].get('TreatmentMachineName')
        if self.machine_name is None:
            print('No treatment machine name found. Please input the machine name:')
            self.machine_name = input()
        beam_model = BeamModel(self.machine_name)
        self.weights_Gp = beam_model.MU2Gp(self.energies) * self.weights_MU
        # Indexing
        u, ind = np.unique(self.labels, return_index=True)
        self.beam_names = u[np.argsort(ind)]
        self.n_beams    = len(self.beam_names)
        pb = [np.where(self.beam_names[beam_index] == self.labels)[0] for beam_index in range(self.n_beams)]
        # Per beam
        self.indices_per_beam    = tuple(pb)
        self.n_beamlets_per_beam = tuple(len(i) for i in pb)
        self.n_layers_per_beam   = tuple(len(np.unique(self.energies[i])) for i in pb)
        self.fluence_per_beam    = tuple([np.sum(self.weights_Gp[i]) for i in pb])
        # Total
        self.indices    = np.arange(len(self.labels), dtype='int64')
        self.n_beamlets = len(self.labels)
        self.n_layers   = np.sum([len(np.unique(self.energies[i])) for i in pb])
        self.fluence    = np.sum(self.weights_Gp)
        # Default: all beamlets considered for optimization
        self.optimized_beamlets = self.indices
        self.fixed_beamlets = np.setdiff1d(self.indices, self.optimized_beamlets)

    def write_tramps(self, tramps_dir):
        file_names = ['beam_{}_{}.tramp'.format(np.where(self.beam_names == i)[0][0], i) for i in self.beam_names]
        if not os.path.exists(tramps_dir):
            os.makedirs(tramps_dir)
        for file in file_names:
            beam_index = file_names.index(file)
            with open(os.path.join(tramps_dir, file), 'w') as f:
                f.write('# beam_name {}\n'.format(self.beam_names[beam_index]))
                f.write('# n_layers {}\n'.format(self.n_layers_per_beam[beam_index]))
                f.write('# n_beamlets {}\n'.format(self.n_beamlets_per_beam[beam_index]))
                f.write('# fluence_Gp {}\n'.format(self.fluence_per_beam[beam_index]))
                f.write('# E(MeV) X(mm) Y(mm) N(Gp)\n')
                for beamlet_index in self.indices_per_beam[beam_index]:
                    f.write('{:5.1f}'.format(self.energies[beamlet_index])+'\t')
                    f.write('{:20.16f}'.format(self.x_coordinates[beamlet_index])+'\t')
                    f.write('{:20.16f}'.format(self.y_coordinates[beamlet_index])+'\t')
                    f.write(str(self.weights_Gp[beamlet_index])+'\n')
        with open(os.path.join(tramps_dir, 'beamlet_scales.txt'), 'w') as f:
            for beamlet_scale in self.scaling_factors:
                f.write(str(beamlet_scale)+'\n')

    def get_tramp_matrix(self):
        arr = np.array([self.energies,
                        self.x_coordinates,
                        self.y_coordinates,
                        self.weights_Gp]).T
        return arr

    def set_beamlet_subset(self, w, b=None):
        '''
        Returns indices for the smallest subset of beamlets carrying w % of the total beam weight.
        The subset contains at least b % of the total number of beamlets.
        
        Args:
            w --> beamlet subsets weight percentage of the total beam weight (int / float)
            b --> beamlet subsets percentage of the total number of beamlets (int / float)
        '''
        indices_sorted_by_weight = np.argsort(self.weights_Gp)[::-1] # Sorting weights, starting from highest
        
        weights_sorted = self.weights_Gp[indices_sorted_by_weight]
        
        beamlet_counter    = 0
        accumulated_weight = 0
        weight_threshold   = self.fluence * w

        for weight in weights_sorted:
            
            beamlet_counter += 1
            accumulated_weight = np.sum(weights_sorted[0:beamlet_counter])
            if accumulated_weight >= weight_threshold:
                break

        if b is not None:
            n_beamlets = round(self.n_beamlets * b)
            if beamlet_counter < n_beamlets:
                beamlet_counter = n_beamlets
            
        subset_indices   = np.sort(indices_sorted_by_weight[0:beamlet_counter])
        excluded_indices = np.sort(indices_sorted_by_weight[beamlet_counter:])
        
        self.subset_n_beamlets = beamlet_counter
        self.subset_fluence = np.sum(self.weights_Gp[subset_indices])
        self.w_ratio = round(self.subset_fluence/self.fluence*100, 2)
        self.b_ratio = round(self.subset_n_beamlets/self.n_beamlets*100, 2)
        self.get_beamlet_subset_info()
        self.optimized_beamlets = subset_indices
        self.fixed_beamlets = excluded_indices
    
    def set_beamlet_subset_absolute(self, n):
        '''
        Returns indices for the most-weighted n number of beamlets.
        
        Args:
            n --> number of beamlets to add to subset (int)
        '''
        indices_sorted_by_weight = np.argsort(self.weights_Gp)[::-1] # Sorting weights, starting from highest
            
        subset_indices   = np.sort(indices_sorted_by_weight[0:n])
        excluded_indices = np.sort(indices_sorted_by_weight[n:])
        
        self.subset_n_beamlets = n
        self.subset_fluence = np.sum(self.weights_Gp[subset_indices])
        self.w_ratio = round(self.subset_fluence/self.fluence*100, 2)
        self.b_ratio = round(self.subset_n_beamlets/self.n_beamlets*100, 2)
        self.get_beamlet_subset_info()
        self.optimized_beamlets = subset_indices
        self.fixed_beamlets = excluded_indices
        
    def exclude_zero_weighted_beamlets(self):
        zero_indices = np.where(self.weights_Gp==0)
        self.optimized_beamlets = np.setdiff1d(self.optimized_beamlets, zero_indices)
        self.fixed_beamlets = np.setdiff1d(self.indices, self.optimized_beamlets)
    
    def adapt_weights(self, scaling_factors_subset):
        self.scaling_factors_subset = scaling_factors_subset
        if not len(self.scaling_factors_subset) == len(self.optimized_beamlets):
            raise Exception('Scaling factors do not fit beamlet subset.')
        # Save nominal plan data
        self.weights_Gp_nominal = self.weights_Gp
        self.fluence_per_beam_nominal = self.fluence_per_beam
        self.fluence_nominal = self.fluence
        # Modify plan weights
        self.weights_Gp[self.optimized_beamlets] *= self.scaling_factors_subset
        self.fluence_per_beam = tuple([np.sum(self.weights_Gp[i]) for i in self.indices_per_beam])
        self.fluence = np.sum(self.weights_Gp)
        print('Beamlet weights adapted.')
        # Calculate statistics
        self.fluence_ratio = round(self.fluence/self.fluence_nominal*100, 2)
        self.fluence_per_beam_ratios = tuple([round(i/j*100, 2) for i,j in zip(self.fluence_per_beam, self.fluence_per_beam_nominal)])
        self.scaling_factors_mean = np.mean(self.scaling_factors_subset)
        self.scaling_factors_max = np.max(self.scaling_factors_subset)
        self.scaling_factors_std = np.std(self.scaling_factors_subset)
        self.get_adaptation_info()
        # Get scales for all beamlets (set 1 for fixed beamlets)
        self.scaling_factors = np.ones(self.n_beamlets, dtype=self.scaling_factors_subset.dtype)
        self.scaling_factors[self.optimized_beamlets] = self.scaling_factors_subset
        
    def get_beamlet_subset_info(self, print_info=True):
        self.beamlet_subset_info = '\nTotal weight of beamlet subset: {} out of {} --> {}%'.format(self.subset_fluence,
                                                                                                   self.fluence,
                                                                                                   self.w_ratio)
        self.beamlet_subset_info += '\nNumber of beamlets in subset: {} out of {} --> {}%'.format(self.subset_n_beamlets,
                                                                                                  self.n_beamlets,
                                                                                                  self.b_ratio)
        if print_info is True : print(self.beamlet_subset_info)
        
    def get_adaptation_info(self, print_info=True):
        self.adaptation_info  = '\nNominal plan fluence = {} Gp'.format(self.fluence_nominal)
        self.adaptation_info += '\nAdapted plan fluence = {} Gp'.format(self.fluence)
        self.adaptation_info += '\nFluence ratio (adapted/nominal) = {}%'.format(self.fluence_ratio)
        self.adaptation_info += '\nNominal plan fluence per beam = {} Gp'.format(self.fluence_per_beam_nominal)
        self.adaptation_info += '\nAdapted plan fluence per beam = {} Gp'.format(self.fluence_per_beam)
        self.adaptation_info += '\nFluence ratios per beam (adapted/nominal) = {} %'.format(self.fluence_per_beam_ratios)
        self.adaptation_info += '\nBeamlet scales mean = {}'.format(self.scaling_factors_mean)
        self.adaptation_info += '\nBeamlet scales max  = {}'.format(self.scaling_factors_max)
        self.adaptation_info += '\nBeamlet scales std  = {}'.format(self.scaling_factors_std)
        if print_info is True : print(self.adaptation_info)
    
    @staticmethod
    def read_beamlet_scales(txt_file):
        with open(txt_file, 'r') as f:
            lst = f.readlines()
        return np.array(lst, 'd')
                
        