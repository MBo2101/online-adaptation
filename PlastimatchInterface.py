# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 10:28:25 2021

@author: MBo
"""

import subprocess
# from RTArray import RTArray, ImageCT, ImageCBCT, ImageMRI, DoseMap, Structure, VectorField
from RTArray import *

class PlastimatchInterface(object):
    
    def __init__(self):
        pass
    
    # Static methods
    
    @staticmethod
    def print_stats(input_file:RTArray):

        '''
        Method returns image stats as a dictionary.
        Args:
            RTArray
        '''
        if not isinstance(input_file, RTArray):
            raise TypeError('Wrong input')
        
        get_stats = subprocess.Popen(['plastimatch','stats',
                                      input_file.path,
                                      ],stdout=subprocess.PIPE)
        get_stats.wait()
        
        stats = (get_stats.stdout.read()).decode('utf-8')
        
        print(stats)
        
        return stats
        
        # stats = stats.strip('\n')
        
        # keys = stats.split(' ')[::2]
        # values = [float(i) for i in stats.split(' ')[1::2]]
        
        
        
        # image_stats = dict(zip(keys, values))
        
        # return image_stats
    
