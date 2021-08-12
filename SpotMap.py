# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 10:28:25 2021

@author: MBo
"""

import numpy as np

class SpotMap(object):

    def __init__(self, **kwargs):
        
        self.__beamlet_energies    = np.array(kwargs.get('energies'),      dtype='d')
        self.__x_coordinates       = np.array(kwargs.get('x_coordinates'), dtype='d')
        self.__y_coordinates       = np.array(kwargs.get('y_coordinates'), dtype='d')
        self.__beamlet_weights_MU  = np.array(kwargs.get('weights_mu'),    dtype='d')
        self.__beamlet_weights_Gp  = np.array(kwargs.get('weights_gp'),    dtype='d')
        self.__beam_name           = np.array(kwargs.get('beam_name'),     dtype='U')
        
        self.__n_beams    = len(np.unique(self.__beam_name))
        self.__n_beamlets = len(self.__beamlet_energies)
        self.__n_layers   = len(np.unique(self.__beamlet_energies))
        self.__beamlet_indices = np.arange(self.__n_beamlets, dtype='int64')

    # Properties
    
    @property
    def n_beams(self):
        return self.__n_beams
    @property
    def n_beamlets(self):
        return self.__n_beamlets
    @property
    def n_layers(self):
        return self.__n_layers
    @property
    def energies(self):
        return self.__beamlet_energies
    @property
    def x_coordinates(self):
        return self.__x_coordinates
    @property
    def y_coordinates(self):
        return self.__y_coordinates
    @property
    def weights_MU(self):
        return self.__beamlet_weights_MU
    @property
    def weights_Gp(self):
        return self.__beamlet_weights_Gp
    @property
    def beam_name(self):
        return self.__beam_name

    # Methods

    def print_properties(self):
        c = self.__class__
        props = [p for p in dir(c) if isinstance(getattr(c,p), property) and hasattr(self,p)]
        for p in props:
            print(p + ' = ' +str(getattr(self, p)))

    def get_tramp_matrix(self):
        arr = np.array([self.__beamlet_energies,
                        self.__x_coordinates,
                        self.__y_coordinates,
                        self.__beamlet_weights_Gp]).T
        return arr
    
    def get_for_beam_index(self, beam_index):
        beam = np.unique(self.__beam_name)[beam_index]
        indices = np.where(beam == self.__beam_name)[0]
        return SpotMap(energies = self.__beamlet_energies[indices],
                       x_coordinates = self.__x_coordinates[indices],
                       y_coordinates = self.__y_coordinates[indices],
                       weights_mu = self.__beamlet_weights_MU[indices],
                       weights_gp = self.__beamlet_weights_Gp[indices],
                       beam_name = self.__beam_name[indices])
    
    def split_per_beam(self):
        if self.__n_beams > 1:
            lst = []
            for beam_index in np.arange(self.__n_beams):
                lst.append(self.get_for_beam_index(beam_index))
            return tuple(lst)
        elif self.__n_beams == 1:
            return self
    
    def load_from_rtplan(rtplan_file_path):
        pass
    
    def load_from_tramps(tramps_folder_path):
        pass
    
    