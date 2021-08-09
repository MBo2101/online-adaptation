# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 10:28:25 2021

@author: MBo
"""

import subprocess

class MoquiManager(object):

    def __init__(self):
        self.__exe_path = '/shared/build/moqui...'
    
    # Properties
    
    @property
    def exe_path(self):
        return self.__exe_path
    
    def print_properties(self):
        c = self.__class__
        props = [p for p in dir(c) if isinstance(getattr(c,p), property) and hasattr(self,p)]
        for p in props:
            print(p + ' = ' +str(getattr(self, p)))
            