# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 10:30:55 2021

@author: MBo
"""

from RTArrays.RTArray import RTArray

class DoseMap(RTArray):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.base_value = 0.
