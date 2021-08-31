# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 10:28:25 2021

@author: MBo
"""

import numpy as np
# from pydicom import dcmread
import pydicom

# TODO: decide where to put beam model name

class SpotMap(object):

    def __init__(self, **kwargs):
        '''
        Keyword arguments:
            rtplan_file_path
            tramps_folder_path
            MU2Gp_function
        '''
        rtplan_file_path   = kwargs.get('rtplan_file_path')
        tramps_folder_path = kwargs.get('tramps_folder_path')
        MU2Gp_function     = kwargs.get('MU2Gp_function')
        
        if rtplan_file_path is not None:
            self.load_from_rtplan(rtplan_file_path)
            
        elif tramps_folder_path is not None:
            self.load_from_tramps(tramps_folder_path)
        
        else:
            self.__energies      = np.array(kwargs.get('energies'),      dtype='d')
            self.__x_coordinates = np.array(kwargs.get('x_coordinates'), dtype='d')
            self.__y_coordinates = np.array(kwargs.get('y_coordinates'), dtype='d')
            self.__weights_MU    = np.array(kwargs.get('weights_mu'),    dtype='d')
            self.__weights_Gp    = np.array(kwargs.get('weights_gp'),    dtype='d')
            self.__labels        = np.array(kwargs.get('labels'),        dtype='U')
        
        u, ind = np.unique(self.__labels, return_index=True)
        
        self.__beam_names = u[np.argsort(ind)]
        self.__n_beams    = len(self.__beam_names)
        self.__n_beamlets = len(self.__energies)
        self.__n_layers   = len(np.unique(self.__energies))
        self.__indices    = np.arange(self.__n_beamlets, dtype='int64')
        
        if not hasattr(self, 'weights_Gp'):
            if MU2Gp_function is None:
                raise Exception('Please provide appropriate MU2Gp conversion function.')
            else:
                self.__weights_Gp = MU2Gp_function(self.__energies) * self.__weights_MU

    # Properties

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
    def n_layers(self):
        return self.__n_layers
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
    def labels(self):
        return self.__labels

    # Methods

    def print_properties(self):
        c = self.__class__
        props = [p for p in dir(c) if isinstance(getattr(c,p), property) and hasattr(self,p)]
        for p in props:
            print(p + ' = ' +str(getattr(self, p)))

    def get_indices_per_beam(self):
        lst = [np.where(self.__beam_names[beam_index] == self.__labels)[0] for beam_index in range(self.__n_beams)]
        return tuple(lst)

    def get_tramp_matrix(self):
        arr = np.array([self.__energies,
                        self.__x_coordinates,
                        self.__y_coordinates,
                        self.__weights_Gp]).T
        return arr
    
    def load_from_rtplan(self, rtplan_file_path):
        
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
    
        #     total_weight_MU = total_weight_MU + spot_weight_MU
        #     total_weight_Gp = total_weight_Gp + spot_weight_Gp

        #     spot_map_string = spot_map_string + "{}\t".format(round(beam_energy,3)) + "{}\t".format(round(spot_position_x,3)) + "{}\t".format(round(spot_position_y,3)) + "{}\n".format(round(spot_weight_Gp,9))
    
        #     MU_per_layer += spot_weight_MU
        #     Gp_per_layer += spot_weight_Gp
    
        # spots_per_layer_list.append(spots_per_layer)
        # MU_per_layer_list.append(round(MU_per_layer,2))
        # Gp_per_layer_list.append(round(Gp_per_layer,2))
    
    def load_from_tramps(tramps_folder_path):
        pass
    
    