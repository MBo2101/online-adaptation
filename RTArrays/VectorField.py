# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 10:30:55 2021

@author: MBo
"""

from RTArrays.RTArray import RTArray

class VectorField(RTArray):
    def __init__(self, path):
        super().__init__(path)
        self.base_value = 0.

class TranslationVF(VectorField):
    def __init__(self, path):
        super().__init__(path)

class RigidVF(VectorField):
    def __init__(self, path):
        super().__init__(path)

class BSplineVF(VectorField):
    def __init__(self, path):
        super().__init__(path)

