# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 10:28:25 2021

@author: MBo
"""

import os
import subprocess
from pympler import asizeof
from RTArray import RTArray, PatientImage, ImageCT, ImageCBCT, ImageMRI, DoseMap, Structure, VectorField, TranslationVF, RigidVF, BSplineVF 

# from RTArray import *

class PlastimatchInterface(object):
    
    def __init__(self):
        pass

    def __check_input(input_file, supported_cls):
        '''
        Method to support both RTArray (class) or path (string) as input.
        
        Args:
            input_file --> instance of RTArray class or file path
            supported_cls --> list or tuple of supported classes
        '''
        if isinstance(input_file, str):
            if os.path.exists(input_file):
                return input_file
            else:
                raise TypeError('Wrong input')
        else:
            if any([issubclass(input_file.__class__, i) for i in supported_cls]):
                return input_file.path
            else:
                raise TypeError('Wrong input')
    
    # Static methods
    
    @staticmethod
    def print_stats(input_file):
        '''
        Prints stats of input image, dose, structure or vector field.
        
        Args:
            input_file --> instance of RTArray class or file path
        '''
        supported_classes = (RTArray,)
        file_path = PlastimatchInterface.__get_path(input_file, supported_classes)
        
        get_stats = subprocess.Popen(['plastimatch','stats',
                                      file_path,
                                      ],stdout=subprocess.PIPE)
        get_stats.wait()

        print(get_stats.stdout.read().decode('utf-8'))
        
    @staticmethod
    def get_stats(input_file):
        '''
        Returns stats of input image, dose, or structure as a dictionary.
        
        Args:
            input_file --> instance of RTArray class (except VectorField) or file path
        '''
        #TODO: Make it work for VectorField
        
        supported_classes = (PatientImage, DoseMap, Structure)
        file_path = PlastimatchInterface.__get_path(input_file, supported_classes)
        
        get_stats = subprocess.Popen(['plastimatch','stats',
                                      file_path,
                                      ],stdout=subprocess.PIPE)
        get_stats.wait()
        
        stats  = (get_stats.stdout.read()).decode('utf-8')
        stats  = stats.strip('\n')
        keys   = stats.split(' ')[::2]
        values = [float(i) for i in stats.split(' ')[1::2]]
    
        return dict(zip(keys, values))
        
    ######################################################################################
    
    @staticmethod
    def extend_or_crop(input_file,
                       output_file_path,
                       x_lower=0,
                       x_upper=0,
                       y_lower=0,
                       y_upper=0,
                       z_lower=0,
                       z_upper=0,
                       unit='vox',
                       default_value=None):
        
        '''
        Method extends and/or crops RTArray (rectangular cuboid).
        Bound parameters are given in number of voxels or mm.
        Convention: (+) extends image, (-) crops image.
        For lower bound parameters != 0 --> origin will be shifted --> voxel resampling may be applied.
    
        Args:
            input_file --> instance of RTArray class or file path
            output_file_path --> path to output file (string)
            x_lower, x_upper, ... --> size of extension/crop at each image bound (int)
            unit --> specifies unit for size, "vox" or "mm" (str)
            default_value --> numeric value that is applied to the background (float)
        '''
        
        supported_classes = (RTArray,)
        file_path = PlastimatchInterface.__get_path(input_file, supported_classes)
        
        input_file.load_file_medpy()
        
        if unit == 'vox':
        
            x_lower_vox = x_lower
            x_upper_vox = x_upper
            y_lower_vox = y_lower
            y_upper_vox = y_upper
            z_lower_vox = z_lower
            z_upper_vox = z_upper
    
        elif unit == 'mm':
        
            x_lower_vox = int(x_lower / input_file.spacing_x)
            x_upper_vox = int(x_upper / input_file.spacing_x)
            y_lower_vox = int(y_lower / input_file.spacing_y)
            y_upper_vox = int(y_upper / input_file.spacing_y)
            z_lower_vox = int(z_lower / input_file.spacing_z)
            z_upper_vox = int(z_upper / input_file.spacing_z)
    
        origin_x_new = input_file.origin_x - x_lower_vox * input_file.spacing_x
        origin_y_new = input_file.origin_y - y_lower_vox * input_file.spacing_y
        origin_z_new = input_file.origin_z - z_lower_vox * input_file.spacing_z
        
        size_x_new = int(input_file.size_x + x_upper_vox + x_lower_vox)
        size_y_new = int(input_file.size_y + y_upper_vox + y_lower_vox)
        size_z_new = int(input_file.size_z + z_upper_vox + z_lower_vox)
    
        spacing_x_new = input_file.spacing_x
        spacing_y_new = input_file.spacing_y
        spacing_z_new = input_file.spacing_z
    
        extend_or_crop = subprocess.Popen(['plastimatch','resample',
                                           '--input', input_file.path,
                                           '--output', output_file_path,
                                           '--origin', '{} {} {}'.format(origin_x_new, origin_y_new, origin_z_new),
                                           '--dim', '{} {} {}'.format(size_x_new, size_y_new, size_z_new),
                                           '--spacing', '{} {} {}'.format(spacing_x_new, spacing_y_new, spacing_z_new),
                                           '--default-value', '-1000'
                                           ])
        extend_or_crop.wait()
        
        if unit == 'mm':
            print('\nExtend/crop input converted from [mm] to [vox]:'\
                  '\nx_lower={}\nx_upper={}\ny_lower={}\ny_upper={}\nz_lower={}\nz_upper={}\n'\
                  .format(x_lower_vox, x_upper_vox, y_lower_vox, y_upper_vox, z_lower_vox, z_upper_vox))
        
        extend_parameters = 'x_lower={}, x_upper={}, y_lower={}, y_upper={}, z_lower={}, z_upper={}, unit="{}"'\
                            .format(x_lower, x_upper, y_lower, y_upper, z_lower, z_upper, unit)
    
        return RTArray(output_file_path)
    
    ######################################################################################
