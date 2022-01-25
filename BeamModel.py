# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 10:28:25 2021

@author: MBo
"""

import numpy as np
from scipy import interpolate

class BeamModel(object):

    def __init__(self, name):
        self.name = name
        self.MU2Gp = self.get_MU2Gp_factor
        self.Gp2MU = self.get_Gp2MU_factor

    def get_MU2Gp_factor(self, beamlet_energy):
    
        if self.name == 'RSL_IBA_DED':
            energies_MeV = np.array([70,      75,      80,      85,      90,      95,      100,     105,     110,     115,     120,     125,     130,     135,     140,     145,     150,     155,     160,     165,     170,     175,     180,     185,     190,     195,     200,     205,     210,     215,     220,     225,     226.7])
            ions_per_MU  = np.array([5.997E7, 6.440E7, 6.870E7, 7.250E7, 7.652E7, 8.018E7, 8.390E7, 8.676E7, 9.000E7, 9.339E7, 9.658E7, 9.937E7, 1.024E8, 1.053E8, 1.086E8, 1.115E8, 1.148E8, 1.174E8, 1.198E8, 1.226E8, 1.255E8, 1.283E8, 1.310E8, 1.332E8, 1.357E8, 1.385E8, 1.410E8, 1.432E8, 1.453E8, 1.473E8, 1.499E8, 1.516E8, 1.524E8])
            f = interpolate.interp1d(energies_MeV, ions_per_MU)
            # Additional calibration factor needed to determine Gp in GPMC vs Gp in RayStation
            # Calibration factor alpha with: Gp_GPMC = Gp_RS * alpha = Gp_RS * (m * E (MeV) + n)
            # Parameters provided by Patrick Wohlfahrt
            m = -0.00021028
            n = 0.921474
            return f(beamlet_energy) * (m*beamlet_energy+n) / 10**9
        
        elif self.name == 'R330_01':
            return (1.0-0.99/100)*(-4.54e-12*beamlet_energy**5+3.8144e-09*beamlet_energy**4-1.255e-06*beamlet_energy**3+0.00020231*beamlet_energy**2-0.015917*beamlet_energy+1.5402)/1e9

        elif self.name == 'r330_01r':
            return (1.0-0.99/100)*(-4.54e-12*beamlet_energy**5+3.8144e-09*beamlet_energy**4-1.255e-06*beamlet_energy**3+0.00020231*beamlet_energy**2-0.015917*beamlet_energy+1.5402)/1e9

    def get_Gp2MU_factor(self, beamlet_energy):
        return 1/self.get_MU2Gp_factor(beamlet_energy)