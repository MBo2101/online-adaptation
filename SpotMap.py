# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 10:28:25 2021

@author: MBo
"""

import numpy as np
import os
import pydicom

class SpotMap(object):

    def __init__(self, **kwargs):
        '''
        Load spot map either from tramp files, or from a DICOM RT plan file.
        In case of RT plan file: conversion function from MU to Gp needs to be provided.
        
        Keyword arguments:
            rtplan_file_path
            tramps_folder_path
            MU2Gp_function
        '''
        rtplan_file_path   = kwargs.get('rtplan_file_path')
        tramps_folder_path = kwargs.get('tramps_folder_path')
        MU2Gp_function     = kwargs.get('MU2Gp_function')
        
        if rtplan_file_path is not None:
            self.load_rtplan(rtplan_file_path)
            
        elif tramps_folder_path is not None:
            self.load_tramps(tramps_folder_path)
        
        # Converting MU to Gp
        if not hasattr(self, 'weights_Gp'):
            if MU2Gp_function is None:
                raise Exception('Please provide appropriate MU2Gp conversion function.')
            else:
                self.__weights_Gp = MU2Gp_function(self.__energies) * self.__weights_MU
        
        u, ind = np.unique(self.__labels, return_index=True)
        self.__beam_names = u[np.argsort(ind)]
        self.__n_beams    = len(self.__beam_names)
        pb = [np.where(self.__beam_names[beam_index] == self.__labels)[0] for beam_index in range(self.__n_beams)]
        
        # Per beam
        self.__indices_per_beam    = tuple(pb)
        self.__n_beamlets_per_beam = tuple(len(i) for i in pb)
        self.__n_layers_per_beam   = tuple(len(np.unique(self.__energies[i])) for i in pb)
        self.__fluence_per_beam    = tuple([np.sum(self.__weights_Gp[i]) for i in pb])
        
        # Total
        self.__indices    = np.arange(len(self.__labels), dtype='int64')
        self.__n_beamlets = len(self.__labels)
        self.__n_layers   = np.sum([len(np.unique(self.__energies[i])) for i in pb])
        self.__fluence    = np.sum(self.__weights_Gp)
        
        # Default: all beamlets considered for optimization
        self.__optimized_beamlets = self.__indices

    # Properties

    @property
    def indices(self):
        return self.__indices
    @property
    def indices_per_beam(self):
        return self.__indices_per_beam
    @property
    def beam_names(self):
        return self.__beam_names
    @property
    def n_beams(self):
        return self.__n_beams
    @property
    def n_beamlets(self):
        return self.__n_beamlets
    @property
    def n_beamlets_per_beam(self):
        return self.__n_beamlets_per_beam
    @property
    def n_layers(self):
        return self.__n_layers
    @property
    def n_layers_per_beam(self):
        return self.__n_layers_per_beam
    @property
    def energies(self):
        return self.__energies
    @property
    def x_coordinates(self):
        return self.__x_coordinates
    @property
    def y_coordinates(self):
        return self.__y_coordinates
    @property
    def weights_MU(self):
        return self.__weights_MU
    @property
    def weights_Gp(self):
        return self.__weights_Gp
    @property
    def fluence(self):
        return self.__fluence
    @property
    def fluence_per_beam(self):
        return self.__fluence_per_beam
    @property
    def labels(self):
        return self.__labels
    @property
    def optimized_beamlets(self):
        return self.__optimized_beamlets

    # Methods

    def print_properties(self):
        c = self.__class__
        props = [p for p in dir(c) if isinstance(getattr(c,p), property) and hasattr(self,p)]
        for p in props:
            print(p + ' = ' +str(getattr(self, p)))

    def get_tramp_matrix(self):
        arr = np.array([self.__energies,
                        self.__x_coordinates,
                        self.__y_coordinates,
                        self.__weights_Gp]).T
        return arr
    
    def load_rtplan(self, rtplan_file_path):
        
        energies      = np.array([], dtype='d')
        x_coordinates = np.array([], dtype='d')
        y_coordinates = np.array([], dtype='d')
        weights_mu    = np.array([], dtype='d')
        labels        = np.array([], dtype='U')
        ds = pydicom.dcmread(rtplan_file_path)
        
        for beam_index in range(len(ds.IonBeamSequence)):
            beam_name = ds.IonBeamSequence[beam_index].BeamName
            ICPS = ds.IonBeamSequence[beam_index].IonControlPointSequence
            n_layers = int(len(ICPS)/2)
            
            for layer_index in range(n_layers):
                n_spots_in_layer = int(len(ICPS[layer_index*2].ScanSpotPositionMap)/2)
                
                for beamlet_index in range(n_spots_in_layer):
                    energy = ICPS[layer_index*2].NominalBeamEnergy
                    x_coordinate = ICPS[layer_index*2].ScanSpotPositionMap[beamlet_index*2]
                    y_coordinate = ICPS[layer_index*2].ScanSpotPositionMap[beamlet_index*2+1]
                    
                    if n_spots_in_layer == 1:
                        weight_mu = ICPS[layer_index*2].ScanSpotMetersetWeights
                    else:
                        weight_mu = ICPS[layer_index*2].ScanSpotMetersetWeights[beamlet_index]
                        
                    energies      = np.append(energies,      energy)
                    x_coordinates = np.append(x_coordinates, x_coordinate)
                    y_coordinates = np.append(y_coordinates, y_coordinate)
                    weights_mu    = np.append(weights_mu,    weight_mu)
                    labels        = np.append(labels,        beam_name)
        
        self.__energies      = energies
        self.__x_coordinates = x_coordinates
        self.__y_coordinates = y_coordinates
        self.__weights_MU    = weights_mu
        self.__labels        = labels
    
    def load_tramps(self, tramps_folder_path):
        files_lst = os.listdir(tramps_folder_path)
        
        energies      = np.array([], dtype='d')
        x_coordinates = np.array([], dtype='d')
        y_coordinates = np.array([], dtype='d')
        weights_gp    = np.array([], dtype='d')
        labels        = np.array([], dtype='U')
        
        for file in files_lst:
            with open(os.path.join(tramps_folder_path, file), 'r') as f:
                lines = f.readlines()
                for line in lines[:lines.index('# E(MeV) X(mm) Y(mm) N(Gp)\n')+1]:
                    if 'beam_name' in line:
                        beam_name = line.split(' ')[-1].strip('\n')
                        break
                for line in lines[lines.index('# E(MeV) X(mm) Y(mm) N(Gp)\n')+1:]:
                    e,x,y,w = line.split('\t')
                    energies      = np.append(energies,      float(e))
                    x_coordinates = np.append(x_coordinates, float(x))
                    y_coordinates = np.append(y_coordinates, float(y))
                    weights_gp    = np.append(weights_gp,    float(w))
                    labels        = np.append(labels,        beam_name)
        self.__energies      = energies
        self.__x_coordinates = x_coordinates
        self.__y_coordinates = y_coordinates
        self.__weights_Gp    = weights_gp
        self.__labels        = labels
        
    def write_tramps(self, tramps_folder_path):
        file_names = ['beam_{}_{}.tramp'.format(np.where(self.__beam_names == i)[0][0], i) for i in self.__beam_names]
        if not os.path.exists(tramps_folder_path):
            os.makedirs(tramps_folder_path)
        for file in file_names:
            beam_index = file_names.index(file)
            with open(os.path.join(tramps_folder_path, file), 'w') as f:
                f.write('# beam_name {}\n'.format(self.__beam_names[beam_index]))
                f.write('# n_layers {}\n'.format(self.__n_layers_per_beam[beam_index]))
                f.write('# n_beamlets {}\n'.format(self.__n_beamlets_per_beam[beam_index]))
                f.write('# fluence_Gp {}\n'.format(self.__fluence_per_beam[beam_index]))
                f.write('# E(MeV) X(mm) Y(mm) N(Gp)\n')
                for beamlet_index in self.__indices_per_beam[beam_index]:
                    f.write(str(self.__energies[beamlet_index])+'\t')
                    f.write(str(self.__x_coordinates[beamlet_index])+'\t')
                    f.write(str(self.__y_coordinates[beamlet_index])+'\t')
                    f.write(str(self.__weights_Gp[beamlet_index])+'\n')
    
    def set_beamlet_subset(self, w, b=None):
        '''
        Returns indices for the smallest subset of beamlets carrying w % of the total beam weight.
        The subset contains at least b % of the total number of beamlets.
        
        Args:
            w --> beamlet subsets weight percentage of the total beam weight (int / float)
            b --> beamlet subsets percentage of the total number of beamlets (int / float)
        '''
        indices_sorted_by_weight = np.argsort(self.__weights_Gp)[::-1] # Sorting weights, starting from highest
        
        weights_sorted = self.__weights_Gp[indices_sorted_by_weight]
        
        beamlet_counter    = 0
        accumulated_weight = 0
        weight_threshold   = self.__fluence * w

        for weight in weights_sorted:
            
            beamlet_counter += 1
            accumulated_weight = np.sum(weights_sorted[0:beamlet_counter])
            if accumulated_weight >= weight_threshold:
                break

        if b is not None:
            n_beamlets = round(self.__n_beamlets * b)
            if beamlet_counter < n_beamlets:
                beamlet_counter = n_beamlets
            
        subset_indices   = np.sort(indices_sorted_by_weight[0:beamlet_counter])
        excluded_indices = np.sort(indices_sorted_by_weight[beamlet_counter:])
        
        subset_fluence = np.sum(self.__weights_Gp[subset_indices])
        
        w_ratio = round(subset_fluence/self.__fluence*100, 2)
        b_ratio = round(beamlet_counter/self.__n_beamlets*100, 2)

        print('\nTotal weight of beamlet subset: {} out of {} --> {}%'.format(subset_fluence, 
                                                                              self.__fluence, 
                                                                              w_ratio))
        print('Number of beamlets in subset: {} out of {} --> {}%'.format(beamlet_counter, 
                                                                          self.__n_beamlets, 
                                                                          b_ratio))
        self.__optimized_beamlets = subset_indices
        
        return subset_indices, excluded_indices
    
    def set_beamlet_subset_absolute(self, n):
        '''
        Returns indices for the most-weighted n number of beamlets.
        
        Args:
            n --> number of beamlets to add to subset (int)
        '''
        indices_sorted_by_weight = np.argsort(self.__weights_Gp)[::-1] # Sorting weights, starting from highest
            
        subset_indices   = np.sort(indices_sorted_by_weight[0:n])
        excluded_indices = np.sort(indices_sorted_by_weight[n:])
        
        subset_fluence = np.sum(self.__weights_Gp[subset_indices])
        
        w_ratio = round(subset_fluence/self.__fluence*100, 2)
        b_ratio = round(n/self.__n_beamlets*100, 2)

        print('\nTotal weight of beamlet subset: {} out of {} --> {}%'.format(subset_fluence, 
                                                                              self.__fluence, 
                                                                              w_ratio))
        print('Number of beamlets in subset: {} out of {} --> {}%'.format(n, 
                                                                          self.__n_beamlets,
                                                                          b_ratio))
        self.__optimized_beamlets = subset_indices
        
        return subset_indices, excluded_indices
    
    def adapt_weights(self, scaling_factors):
        
        if not len(scaling_factors) == len(self.__optimized_beamlets):
            raise Exception('Scaling factors do not fit beamlet subset.')
            
        self.__weights_Gp[self.__optimized_beamlets] *= scaling_factors
        self.__fluence_per_beam = tuple([np.sum(self.__weights_Gp[i]) for i in self.__indices_per_beam])
        self.__fluence = np.sum(self.__weights_Gp)
        
        print('\nBeamlet weights adapted.')
        