# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 10:28:25 2021

@author: MBo
"""

import numpy as np

class SpotMap(object):

    def __init__(self, **kwargs):
        
        self.__n_beamlets = len(kwargs.get('weights'))
        self.__n_layers   = len(np.unique(kwargs.get('energies')))
        self.__beamlet_indices = np.arange(self.__n_beamlets, dtype='int64')
        
        self.__beamlet_energies = np.array(kwargs.get('energies'),      dtype='d')
        self.__x_coordinates    = np.array(kwargs.get('x_coordinates'), dtype='d')
        self.__y_coordinates    = np.array(kwargs.get('y_coordinates'), dtype='d')
        self.__beamlet_weights  = np.array(kwargs.get('weights'),       dtype='d') # Weights in Gp
        
        self.__total_beam_weight = np.sum(self.__beamlet_weights)

    # Properties
    
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
    def weights(self):
        return self.__beamlet_weights
    @property
    def total_beam_weight(self):
        return self.__total_beam_weight

    # Methods

    def print_properties(self):
        c = self.__class__
        props = [p for p in dir(c) if isinstance(getattr(c,p), property) and hasattr(self,p)]
        for p in props:
            print(p + ' = ' +str(getattr(self, p)))


