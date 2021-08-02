# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 10:30:55 2021

@author: MBo
"""

from RTArrays.RTArray import RTArray

class Structure(RTArray):
    def __init__(self, path):
        super().__init__(path)
        self.base_value = 0.
