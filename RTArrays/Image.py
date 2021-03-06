# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 10:30:55 2021

@author: MBo
"""

from RTArrays.RTArray import RTArray

class Image(RTArray):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class ImageCT(Image):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.base_value = -1000.

class ImageCBCT(Image):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.base_value = -1000.

class ImageMRI(Image):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.base_value = 0.
