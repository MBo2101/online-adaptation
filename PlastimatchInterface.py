# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 10:28:25 2021

@author: MBo
"""

import os
import shutil
import subprocess
# from pympler import asizeof
from RTArray import RTArray, PatientImage, ImageCT, ImageCBCT, ImageMRI, DoseMap, Structure, VectorField, TranslationVF, RigidVF, BSplineVF 

# from RTArray import *

class PlastimatchInterface(object):
    
    def __init__(self):
        pass

    def __input_check(input_file, supported_cls):
        '''
        Method to check input
        
        Args:
            input_file --> instance of RTArray class
            supported_cls --> list or tuple of supported classes
        '''
        if not any([issubclass(input_file.__class__, i) for i in supported_cls]):
            raise TypeError('Wrong input')
        else:
            return True
    
    # Static methods
    
    @staticmethod
    def print_stats(input_file):
        '''
        Prints stats of input image, dose, structure or vector field.
        
        Args:
            input_file --> instance of RTArray class
        '''
        supported_cls = (RTArray,)
        PlastimatchInterface.__input_check(input_file, supported_cls)
        
        get_stats = subprocess.Popen(['plastimatch','stats',
                                      input_file.path,
                                      ],stdout=subprocess.PIPE)
        get_stats.wait()

        print(get_stats.stdout.read().decode('utf-8'))
        
    @staticmethod
    def get_stats(input_file):
        '''
        Returns stats of input image, dose, or structure as a dictionary.
        
        Args:
            input_file --> instance of RTArray class (except VectorField)
        '''
        #TODO: Make it work for VectorField

        supported_cls = (PatientImage, DoseMap, Structure)
        PlastimatchInterface.__input_check(input_file, supported_cls)
        
        get_stats = subprocess.Popen(['plastimatch','stats',
                                      input_file.path,
                                      ],stdout=subprocess.PIPE)
        get_stats.wait()
        
        stats  = (get_stats.stdout.read()).decode('utf-8')
        stats  = stats.strip('\n')
        keys   = stats.split(' ')[::2]
        values = [float(i) for i in stats.split(' ')[1::2]]
    
        return dict(zip(keys, values))
    
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
        Returns RTArray object for the output file (same class as input_file).
        Bound parameters are given in number of voxels or mm.
        Convention: (+) extends image, (-) crops image.
        For lower bound parameters != 0 --> origin will be shifted --> voxel resampling may be applied.
    
        Args:
            input_file --> instance of RTArray class
            output_file_path --> path to output file (string)
            x_lower, x_upper, ... --> size of extension/crop at each image bound (int)
            unit --> specifies unit for size, "vox" or "mm" (str)
            default_value --> numeric value that is applied to the background (float)
                          --> will be applied according to class if not specified
        '''
        supported_cls = (RTArray,)
        PlastimatchInterface.__input_check(input_file, supported_cls)
        
        input_file.load_header()

        default_value = input_file.base_value if default_value == None else default_value

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
                                            '--default-value', '{}'.format(default_value)
                                            ])
        extend_or_crop.wait()
        
        if unit == 'mm':
            print('\nExtend/crop input converted from [mm] to [vox]:'\
                  '\nx_lower={}\nx_upper={}\ny_lower={}\ny_upper={}\nz_lower={}\nz_upper={}\n'\
                  .format(x_lower_vox, x_upper_vox, y_lower_vox, y_upper_vox, z_lower_vox, z_upper_vox))
    
        return input_file.__class__(output_file_path)
    
    @staticmethod
    def merge_images(background, foreground, output_file_path, *masks):
        '''
        Merges two images (foreground on background).
        Masks specify voxels where the foreground is applied.
        Returns RTArray object for the output file (same class as background).
        
        Args:
            background --> instance of RTArray class for background image file
            foreground --> instance of RTArray class for foreground image file
            output_file_path --> path to output image file (string)
            masks --> paths to masks (list of strings)
        '''
        supported_cls = (RTArray,)
        PlastimatchInterface.__input_check(background, supported_cls)
        PlastimatchInterface.__input_check(foreground, supported_cls)
        
        dirpath = os.path.dirname(foreground)
        temp = Structure(os.path.join(dirpath, 'mask_temp.mha'))
        
        if len(masks) == 0:
            raise Exception('Need at least one mask file.')
            
        elif len(masks) == 1:
            shutil.copyfile(masks[0].path, temp.path)
            
        else:
            PlastimatchInterface.get_union(temp.path, *masks)
            
        mask_foreground = subprocess.Popen(['plastimatch','mask',
                                            '--input', foreground.path,
                                            '--mask', temp.path,
                                            '--mask-value', '0',
                                            '--output', 'foreground_temp.mha'
                                            ], cwd=dirpath)
        mask_foreground.wait()
    
        fill_background = subprocess.Popen(['plastimatch','fill',
                                            '--input', background.path,
                                            '--mask', temp.path,
                                            '--mask-value', '0',
                                            '--output', 'background_temp.mha'
                                            ], cwd=dirpath)
        fill_background.wait()
            
        get_final_image = subprocess.Popen(['plastimatch','add',
                                            'foreground_temp.mha',
                                            'background_temp.mha',
                                            '--output', output_file_path,
                                            ], cwd=dirpath)
        get_final_image.wait()
        
        os.remove(os.path.join(dirpath, 'foreground_temp.mha'))
        os.remove(os.path.join(dirpath, 'background_temp.mha'))
        os.remove(os.path.join(dirpath, 'mask_temp.mha'))
        
        return background.__class__(output_file_path)
    
    @staticmethod
    def scale_image_linear(input_file, output_file_path, *parameters):
        '''
        Applies piece-wise linear function to image.
        Returns RTArray object for the output file (same class as input_file).
        
        Args:
            input_file --> instance of PatientImage or DoseMap
            output_file_path --> path to output file (string)
            parameters --> pair-wise values for linear function (int or float)
        '''      
        # Example values used before: "-1024,-1024,261,58" (261 from CBCT, 58 from CT)
        
        supported_cls = (PatientImage, DoseMap)
        PlastimatchInterface.__input_check(input_file, supported_cls)
        
        transform_str = ''
        
        for val in parameters:
            transform_str = transform_str + '{},'.format(val)
        
        adjust = subprocess.Popen(['plastimatch','adjust',
                                   '--input', input_file.path,
                                   '--output', output_file_path,
                                   '--pw-linear', transform_str,
                                   ])
        adjust.wait()

        return input_file.__class__(output_file_path)

    @staticmethod
    def scale_image_factor(input_file, output_file_path, factor):
        '''
        Multiplies factor to image voxel by voxel.
        Slow implementation! Use "apply_image_weight" instead.
        Returns RTArray object for the output file (same class as input_file).
        
        Args:
            input_file --> instance of PatientImage or DoseMap
            output_file_path --> path to output file (string)
            factor --> multiplication factor (float or int)
        '''     
        supported_cls = (PatientImage, DoseMap)
        PlastimatchInterface.__input_check(input_file, supported_cls)

        stats = PlastimatchInterface.get_stats(input_file) 
        
        min_value = int(stats['MIN'])
        max_value = int(stats['MAX']+1)
        
        values_ini =   [i for i in range(min_value, max_value+1)]
        values_mod =   [i*factor for i in values_ini]
        values_input = [j for k in zip(values_ini, values_mod) for j in k]
        
        PlastimatchInterface.scale_image_linear(input_file, output_file_path, *values_input)

        return input_file.__class__(output_file_path)
    
    @staticmethod
    def apply_image_weight(input_file, output_file_path, weight):
        '''
        Applies weight to image.
        Faster implementation than "scale_image_factor".
        Returns RTArray object for the output file (same class as input_file).
        
        Args:
            input_file --> instance of PatientImage or DoseMap
            output_file_path --> path to output file (string)
            weight --> multiplication factor (float or int)
        '''     
        supported_cls = (PatientImage, DoseMap)
        PlastimatchInterface.__input_check(input_file, supported_cls)

        weight_image = subprocess.Popen(['plastimatch','add',
                                         input_file.path,
                                         '--weight', '{}'.format(weight),
                                         '--output', output_file_path,
                                         ])
        weight_image.wait()

        return input_file.__class__(output_file_path)

    @staticmethod
    def apply_manual_translation(input_file,
                                 output_file_path,
                                 x, y, z,
                                 unit='mm',
                                 frame='shift',
                                 discrete_voxels=False):
        '''
        Applies manual translation to input image, dose, structure or vector field.
        Returns RTArray object for the output file (same class as input_file).
        
        Args:
            input_file --> instance of RTArray class
            output_file_path --> path to output file (string)
            x, y, z --> parameters for 3D shift in mm (int or float)
            unit --> specifies unit for shift distance, "vox" or "mm" (str)
            frame --> specifies how image frame is handled, "shift" or "fix" (str)
            discrete_voxels --> option to apply shift in discrete number of voxels (bool)
        '''
        supported_cls = (RTArray,)
        PlastimatchInterface.__input_check(input_file, supported_cls)
        
        dirpath = os.path.dirname(input_file.path)
        temp_path = os.path.join(dirpath, 'ext_temp.mha')
        vf_file = VectorField(os.path.join(dirpath, 'translation_vf.mha'))
        
        input_file.load_header()
        
        if discrete_voxels == True:
            
            if unit == 'vox':
                
                x = round(x)
                y = round(y)
                z = round(z)
                
            elif unit == 'mm':
                
                x = x / input_file.spacing_x
                y = y / input_file.spacing_y
                z = z / input_file.spacing_z
                
                x = round(x)
                y = round(y)
                z = round(z)
                
                x = x * input_file.spacing_x
                y = y * input_file.spacing_y
                z = z * input_file.spacing_z
        
        # Convention
        x = -x
        y = -y
        z = -z
        
        x_lower, x_upper, y_lower, y_upper, z_lower, z_upper = 0, 0, 0, 0, 0, 0
        
        if x > 0:
            x_lower = abs(x)
        elif x < 0:
            x_upper = abs(x)
            
        if y > 0:
            y_lower = abs(y)
        elif y < 0:
            y_upper = abs(y)
            
        if z > 0:
            z_lower = abs(z)
        elif z < 0:
            z_upper = abs(z)
        
        PlastimatchInterface.extend_or_crop(input_file, temp_path, x_lower, x_upper, y_lower, y_upper, z_lower, z_upper, unit)
        
        if unit == 'vox':
            
            x = x * input_file.spacing_x
            y = y * input_file.spacing_y
            z = z * input_file.spacing_z
        
        translation_str = '"{} {} {}"'.format(x, y, z)
        
        create_vf = subprocess.Popen(['plastimatch synth-vf --fixed {} --xf-trans {} --output {}'
                                      .format(temp_path, translation_str, vf_file.path)
                                      ], cwd = dirpath, shell=True)
        create_vf.wait()
        
        PlastimatchInterface.warp_image(input_file,
                                        output_file_path,
                                        vf_file)
        
        os.remove(temp_path)
    
        if frame == 'shift':
            PlastimatchInterface.extend_or_crop(input_file.__class__(output_file_path),
                                                output_file_path,
                                                -x_upper, -x_lower,
                                                -y_upper, -y_lower,
                                                -z_upper, -z_lower,
                                                unit)
        
        elif frame == 'fix':
            PlastimatchInterface.extend_or_crop(input_file.__class__(output_file_path),
                                                output_file_path,
                                                -x_lower, -x_upper,
                                                -y_lower, -y_upper,
                                                -z_lower, -z_upper,
                                                unit)
        
        print('\nApplied translation:\n'\
              'x = {} mm | = {} voxels\n'\
              'y = {} mm | = {} voxels\n'\
              'z = {} mm | = {} voxels'\
              .format(-x, -x/input_file.spacing_x, -y, -y/input_file.spacing_y, -z, -z/input_file.spacing_z))
            
        return input_file.__class__(output_file_path)

    @staticmethod
    def warp_image(input_file, output_file_path, vf_file, default_value=None):
        '''
        Warps image using an input vector field.
        Returns RTArray object for the output file (same class as input_file).
        
        Args:
            input_file --> instance of RTArray class (except Structure)
            output_file_path --> path to output file (string)
            vf_file --> instance of VectorField class
            default_value --> numeric value that is applied to the background (float)
                          --> will be applied according to class if not specified
        '''
        supported_cls = (PatientImage, DoseMap, VectorField)
        PlastimatchInterface.__input_check(input_file, supported_cls)
        PlastimatchInterface.__input_check(vf_file, (VectorField,))
        
        default_value = input_file.base_value if default_value == None else default_value
        
        warp = subprocess.Popen(['plastimatch','convert',
                                 '--input', input_file.path,
                                 '--output-img', output_file_path,
                                 '--xf', vf_file.path,
                                 '--default-value', str(default_value),
                                 # '--algorithm', 'itk',
                                 # '--output-type','float',
                                 ])
        warp.wait()
        
        return input_file.__class__(output_file_path)

    # Structure methods

    @staticmethod
    def get_union(output_mask_path, *masks):
        '''
        Generates union of provided masks.
        Returns Structure object for the output mask file.
        
        Args:
            output_mask_path --> path to output mask file (string)
            masks --> Strucutre objects of input masks
        '''
        supported_cls = (Structure,)
        
        if len(masks) < 2:
            raise Exception('Need at least two mask files.')
            
        all(PlastimatchInterface.__input_check(mask, supported_cls) for mask in masks)
        
        shutil.copyfile(masks[0].path, output_mask_path)
        
        for mask in masks[1:]:
        
            union = subprocess.Popen(['plastimatch', 'union',
                                      output_mask_path,
                                      mask,
                                      '--output', output_mask_path,
                                      ])
            union.wait()
            
        return Structure(output_mask_path)
    
    
            
        
        