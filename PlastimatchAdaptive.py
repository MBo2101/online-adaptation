# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 10:28:25 2021

@author: MBo
"""

import numpy as np
import os
import shutil
import subprocess
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from RTArrays import RTArray, TranslationVF

'''
Plastimatch extension to support the adaptive proton therapy project.
'''

class PlastimatchAdaptive(object):
    
    def __init__(self):
        pass

    # def __input_check(input_file, supported_cls):
    #     '''
    #     Method to check input.
        
    #     Args:
    #         input_file --> instance of RTArray class
    #         supported_cls --> list or tuple of supported classes
    #     '''
    #     if not any([issubclass(input_file.__class__, i) for i in supported_cls]):
    #         raise TypeError('Wrong input')
    #     else:
    #         return True
    
    def get_default_value(input_file):
        '''
        Returns default value of input ITK file.
        Assuming CT or CBCT for dtype = 'f' --> -1001
        Assuming binary mask for dtype = 'uint8' --> 0
        
        Args:
            input_file --> path to input file (str)
        '''
        arr = RTArray(input_file)
        
        if arr.data_type is np.dtype('f'):
            return -1001
        
        elif arr.data_type is np.dtype('uint8'):
            return 0
        
    def run(command, *args, **kwargs):
        '''
        Use to run Plastimatch with subprocess.Popen module.
        For keyword arguments use '_' instead of '-' --> e.g. 'output_img'.
        Returns stdout as a string if Plastimatch excecutes succesfully.
        '''
        run_list = ['plastimatch', command]
        stdout_str = ''
        
        for a in args:
            if a != None:
                run_list.append(str(a))
        for k in kwargs:
            if kwargs[k] != None:
                option = '--' + k.replace('_','-')
                run_list.append(option)
                run_list.append(str(kwargs[k]))
        
        run = subprocess.Popen(run_list, stdout=subprocess.PIPE)
        
        while True:
            output_line = run.stdout.readline().decode('utf-8')
            if run.poll() is not None:
                break
            if output_line:
                stdout_str += output_line
                print(output_line.strip())
        
        if run.poll() == 0:
            return stdout_str
        else:
            raise Exception('Plastimatch error')
        
    def image_registration_global(**kwargs):
        '''
        Use to define global parameters of Plastimatch command file.
        '''
        known_keywords = ['fixed',
                          'moving',
                          'vf_out',
                          'img_out',
                          'fixed_mask',
                          'moving_mask',
                          'default_value']
        
        if all([k in known_keywords for k in kwargs]) != True:
            raise TypeError('Unknown keyword argument')
        
        command_file_string = '[GLOBAL]'
        for k in kwargs:
            if kwargs[k] != None:
                command_file_string = command_file_string + '\n{}={}'.format(k, kwargs[k])
        return command_file_string + '\n'
    
    def image_registration_stage(**kwargs):
        '''
        Use to define stages in Plastimatch command file.
        '''
        known_keywords = ['res',
                          'impl',
                          'optim',
                          'xform',
                          'metric',
                          'max_its',
                          'threading',
                          'grid_spac',
                          'regularization_lambda']
        
        if all([k in known_keywords for k in kwargs]) != True:
            raise TypeError('Unknown keyword argument')
        
        command_file_string = '\n[STAGE]'
        for k in kwargs:
            if kwargs[k] != None:
                command_file_string = command_file_string + '\n{}={}'.format(k, kwargs[k])
        return command_file_string + '\n'
    
    #%% General methods
    
    @staticmethod
    def print_stats(input_file):
        '''
        Prints stats of input image, dose, structure or vector field.
        
        Args:
            input_file --> path to input file (str)
        '''
        
        PlastimatchAdaptive.run('stats', input_file)
        
    @staticmethod
    def get_stats(input_file):
        '''
        Returns stats of input image, dose, or structure as a dictionary.
        
        Args:
            input_file --> path to input file (str)
        '''
        #TODO: Make it work for VectorField
        
        stats  = PlastimatchAdaptive.run('stats', input_file)
        stats  = stats.strip('\n')
        keys   = stats.split(' ')[::2]
        values = [float(i) for i in stats.split(' ')[1::2]]
    
        return dict(zip(keys, values))
    
    @staticmethod
    def extend_or_crop(input_file,
                       output_file,
                       x_lower=0,
                       x_upper=0,
                       y_lower=0,
                       y_upper=0,
                       z_lower=0,
                       z_upper=0,
                       unit='vox',
                       default_value=-1001):
        '''
        Method extends and/or crops image array (rectangular cuboid).
        Bound parameters are given in number of voxels or mm.
        Convention: (+) extends image, (-) crops image.
        For lower bound parameters != 0 --> origin will be shifted --> voxel resampling may be applied.
    
        Args:
            input_file --> path to input file (str)
            output_file --> path to output file (str)
            x_lower, x_upper, ... --> size of extension/crop at each image bound (int / float)
            unit --> specifies unit for size, "vox" or "mm" (str)
            default_value --> numeric value that is applied to the background (int / float)
        '''
        
        arr = RTArray(path = input_file)

        if unit == 'vox':
        
            x_lower_vox = x_lower
            x_upper_vox = x_upper
            y_lower_vox = y_lower
            y_upper_vox = y_upper
            z_lower_vox = z_lower
            z_upper_vox = z_upper
    
        elif unit == 'mm':
        
            x_lower_vox = int(x_lower / arr.spacing_x)
            x_upper_vox = int(x_upper / arr.spacing_x)
            y_lower_vox = int(y_lower / arr.spacing_y)
            y_upper_vox = int(y_upper / arr.spacing_y)
            z_lower_vox = int(z_lower / arr.spacing_z)
            z_upper_vox = int(z_upper / arr.spacing_z)
    
        origin_x_new = arr.origin_x - x_lower_vox * arr.spacing_x
        origin_y_new = arr.origin_y - y_lower_vox * arr.spacing_y
        origin_z_new = arr.origin_z - z_lower_vox * arr.spacing_z
        
        size_x_new = int(arr.size_x + x_upper_vox + x_lower_vox)
        size_y_new = int(arr.size_y + y_upper_vox + y_lower_vox)
        size_z_new = int(arr.size_z + z_upper_vox + z_lower_vox)
    
        spacing_x_new = arr.spacing_x
        spacing_y_new = arr.spacing_y
        spacing_z_new = arr.spacing_z
    
        origin_new = '{} {} {}'.format(origin_x_new, origin_y_new, origin_z_new)
        size_new = '{} {} {}'.format(size_x_new, size_y_new, size_z_new)
        spacing_new = '{} {} {}'.format(spacing_x_new, spacing_y_new, spacing_z_new)
    
        PlastimatchAdaptive.run('resample',
                                input = input_file,
                                output = output_file,
                                origin = origin_new,
                                dim = size_new,
                                spacing = spacing_new,
                                default_value = default_value)
        
        if unit == 'mm':
            print('\nExtend/crop input converted from [mm] to [vox]:'\
                  '\nx_lower={}\nx_upper={}\ny_lower={}\ny_upper={}\nz_lower={}\nz_upper={}\n'\
                  .format(x_lower_vox, x_upper_vox, y_lower_vox, y_upper_vox, z_lower_vox, z_upper_vox))
    
        parameters = 'x_lower={}, x_upper={}, y_lower={}, y_upper={}, z_lower={}, z_upper={}, unit="{}"'\
                     .format(x_lower, x_upper, y_lower, y_upper, z_lower, z_upper, unit)

        return parameters
    
    @staticmethod
    def merge_images(background, foreground, output_file, *masks):
        '''
        Merges two images (foreground on background).
        Masks specify voxels where the foreground is applied.
        
        Args:
            background --> path to input background image (str)
            foreground --> path to input foreground image (str)
            output_file --> path to output file (str)
            masks --> paths to input masks (str)
        '''
        
        dirpath = os.path.dirname(foreground)
        temp_mask = os.path.join(dirpath, 'mask_temp.mha')
        temp_foreground = os.path.join(dirpath, 'foreground_temp.mha')
        temp_background = os.path.join(dirpath, 'background_temp.mha')
        
        if len(masks) == 0:
            raise Exception('Need at least one mask file.')
            
        elif len(masks) == 1:
            shutil.copyfile(masks[0], temp_mask)
            
        else:
            PlastimatchAdaptive.get_union(temp_mask, *masks)
        
        PlastimatchAdaptive.run('mask',
                                input = foreground,
                                mask = temp_mask,
                                mask_value = 0,
                                output = temp_foreground)
        
        PlastimatchAdaptive.run('fill',
                                input = background,
                                mask = temp_mask,
                                mask_value = 0,
                                output = temp_background)

        PlastimatchAdaptive.run('add', temp_foreground, temp_background,
                                output = output_file)
        
        os.remove(temp_foreground)
        os.remove(temp_background)
        os.remove(temp_mask)
    
    @staticmethod
    def scale_image_linear(input_file, output_file, *parameters):
        '''
        Applies piece-wise linear function to image.
        
        Args:
            input_file --> path to input file (str)
            output_file --> path to output file (str)
            parameters --> pair-wise values for linear function (int / float)
        '''      
        # Example values used before: "-1024,-1024,261,58" (261 from CBCT, 58 from CT)
        
        transform_str = ''
        
        for val in parameters:
            transform_str = transform_str + '{},'.format(val)
        
        PlastimatchAdaptive.run('adjust',
                                input = input_file,
                                output = output_file,
                                pw_linear = transform_str)

    @staticmethod
    def scale_image_factor(input_file, output_file, factor):
        '''
        Multiplies factor to image voxel by voxel.
        Slow implementation! Use "apply_image_weight" instead.
        
        Args:
            input_file --> path to input file (str)
            output_file --> path to output file (str)
            factor --> multiplication factor (int / float)
        '''

        stats = PlastimatchAdaptive.get_stats(input_file)
        
        min_value = int(stats['MIN'])
        max_value = int(stats['MAX']+1)
        
        values_ini =   [i for i in range(min_value, max_value+1)]
        values_mod =   [i*factor for i in values_ini]
        values_input = [j for k in zip(values_ini, values_mod) for j in k]
        
        PlastimatchAdaptive.scale_image_linear(input_file, output_file, *values_input)
    
    @staticmethod
    def apply_image_weight(input_file, output_file, weight):
        '''
        Applies weight to image.
        Faster implementation than "scale_image_factor".
        
        Args:
            input_file --> path to input file (str)
            output_file --> path to output file (str)
            weight --> multiplication weight (int / float)
        '''

        PlastimatchAdaptive.run('add',
                                input_file,
                                weight = weight,
                                output = output_file)

    @staticmethod
    def sum_images(output_image, *images):
        '''
        Sums input images.
        
        Args:
            output_image --> path to output image (str)
            images --> paths to input images (str)
        '''
        
        if len(images) < 2:
            raise Exception('Need at least two image files.')
        
        images_string = ' '.join(images)
        input_string = 'plastimatch add '+images_string+ ' --output {}'.format(output_image)
        
        sum_images = subprocess.Popen([input_string], shell=True)
        sum_images.wait()

    @staticmethod
    def apply_manual_translation(input_file,
                                 output_file,
                                 output_vf,
                                 x, y, z,
                                 unit='mm',
                                 frame='shift',
                                 discrete_voxels=False):
        '''
        Applies manual translation to input image, dose, structure or vector field.
        
        Args:
            input_file --> path to input file (str)
            output_file --> path to output file (str)
            output_vf --> path to output vector field (str)
            x, y, z --> parameters for 3D shift in mm (int / float)
            unit --> specifies unit for shift distance, "vox" or "mm" (str)
            frame --> specifies how image frame is handled, "shift" or "fix" (str)
            discrete_voxels --> option to apply shift in discrete number of voxels (bool)
        '''
        
        dirpath = os.path.dirname(input_file)
        temp_path = os.path.join(dirpath, 'ext_temp.mha')
        
        arr = RTArray(path = input_file)
        
        if discrete_voxels == True:
            
            if unit == 'vox':
                
                x = round(x)
                y = round(y)
                z = round(z)
                
            elif unit == 'mm':
                
                x = x / arr.spacing_x
                y = y / arr.spacing_y
                z = z / arr.spacing_z
                
                x = round(x)
                y = round(y)
                z = round(z)
                
                x = x * arr.spacing_x
                y = y * arr.spacing_y
                z = z * arr.spacing_z
        
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
        
        PlastimatchAdaptive.extend_or_crop(input_file,
                                           temp_path,
                                           x_lower, x_upper,
                                           y_lower, y_upper,
                                           z_lower, z_upper,
                                           unit)
        if unit == 'vox':
            x = x * arr.spacing_x
            y = y * arr.spacing_y
            z = z * arr.spacing_z
        
        translation_str = '{} {} {}'.format(x, y, z)
        
        PlastimatchAdaptive.run('synth-vf',
                                fixed = temp_path,
                                xf_trans = translation_str,
                                output = output_vf)
        
        if arr.data_type is np.dtype('uint8'):
            PlastimatchAdaptive.warp_mask(input_file,
                                          output_file,
                                          output_vf)
        
        elif arr.data_type is np.dtype('f'):
            PlastimatchAdaptive.warp_image(input_file,
                                           output_file,
                                           output_vf)
        
        os.remove(temp_path)
    
        if frame == 'shift':
            PlastimatchAdaptive.extend_or_crop(output_file,
                                               output_file,
                                               -x_upper, -x_lower,
                                               -y_upper, -y_lower,
                                               -z_upper, -z_lower,
                                               unit)
        
        elif frame == 'fix':
            PlastimatchAdaptive.extend_or_crop(output_file,
                                               output_file,
                                               -x_lower, -x_upper,
                                               -y_lower, -y_upper,
                                               -z_lower, -z_upper,
                                               unit)
        
        print('\nApplied translation:\n'\
              'x = {} mm | = {} voxels\n'\
              'y = {} mm | = {} voxels\n'\
              'z = {} mm | = {} voxels'\
              .format(-x, -x/arr.spacing_x,
                      -y, -y/arr.spacing_y,
                      -z, -z/arr.spacing_z))

    @staticmethod
    def warp_image(input_file, output_file, input_vf, default_value=-1001):
        '''
        Warps image using an input vector field.
        For binary masks use "warp_mask" instead.
        
        Args:
            input_file --> path to input file (str)
            output_file --> path to output file (str)
            input_vf --> path to input vector field (str)
            default_value --> numeric value that is applied to the background (int / float)
        '''
        
        PlastimatchAdaptive.run('convert',
                                input = input_file,
                                output_img = output_file,
                                xf = input_vf,
                                default_value = default_value,
                                # algorithm = 'itk',
                                # output_type = 'float'
                                )
    
    @staticmethod
    def mask_image(input_file, output_file, mask, mask_value=-1001):
        '''
        Masks image using an input mask.
        
        Args:
            input_file --> path to input file (str)
            output_file --> path to output file (str)
            mask --> path to input mask (str)
            mask_value --> numeric value that is applied outside the mask volume (int / float)
        '''
        
        PlastimatchAdaptive.run('mask',
                                input = input_file,
                                output = output_file,
                                mask = mask,
                                mask_value = mask_value)

    @staticmethod
    def fill_image(input_file, output_file, mask, mask_value=-1001):
        '''
        Fills image using an input mask.
        
        Args:
            input_file --> path to input file (str)
            output_file --> path to output file (str)
            mask --> path to input mask (str)
            mask_value --> numeric value that is applied inside the mask volume (int / float)
        '''
        
        PlastimatchAdaptive.run('fill',
                                input = input_file,
                                output = output_file,
                                mask = mask,
                                mask_value = mask_value)
    
    @staticmethod
    def fill_image_threshold(input_file, output_file, threshold, option='above', mask_value=-1001):
        '''
        Fills image voxels above or below a given threshold.
        
        Args:
            input_file --> path to input file (str)
            output_file --> path to output file (str)
            threshold --> threshold value used to separate voxels (int / float)
            option --> defines how voxels are separated: "above" or "below" (str)
            mask_value --> numeric value that is applied inside the mask volume (int / float)
        '''
        dirpath = os.path.dirname(input_file)
        temp = os.path.join(dirpath, 'mask_temp.mha')
        
        if option == 'above':
            PlastimatchAdaptive.run('threshold',
                                    above = threshold,
                                    input = input_file,
                                    output = temp)
        elif option == 'below':
            PlastimatchAdaptive.run('threshold',
                                    below = threshold,
                                    input = input_file,
                                    output = temp)
        PlastimatchAdaptive.run('fill',
                                input = input_file,
                                output = output_file,
                                mask = temp,
                                mask_value = mask_value)
        os.remove(temp)

    @staticmethod
    def DICOM_to_ITK(input_dicom, output_image=None, structures=None, dose_map=None):
        '''
        Converts DICOM to ITK.
        
        Args:
            input_dicom --> path to input dicom directory or file (str)
            output_image --> path to output image (str)
            structures --> path to output directory for structures (str)
            dose_map --> path to output dose image (str)
        '''
        
        PlastimatchAdaptive.run('convert',
                                input = input_dicom,
                                output_type = 'float',
                                output_img = output_image,
                                output_prefix = structures,
                                output_dose_img = dose_map)
    
    @staticmethod
    def ITK_to_DICOM(output_dicom_dir, input_image=None, structures=None, dose_map=None):
        '''
        Converts ITK to DICOM.
        
        Args:
            output_dicom_dir --> path to output dicom directory (str)
            input_image --> path to input image (str)
            structures --> path to input directory for structures (str)
            dose_map --> path to input dose image (str)
        '''
        
        PlastimatchAdaptive.run('convert',
                                input = input_image,
                                input_prefix = structures,
                                input_dose_img = dose_map,
                                output_dicom = output_dicom_dir)

    @staticmethod
    def resample_to_reference(input_file, output_file, reference_image, default_value=-1001):
        '''
        Resamples image based on reference.
        
        Args:
            input_file --> path to input file (str)
            output_file --> path to output file (str)
            reference_image --> path to reference image (str)
            default_value --> numeric value that is applied to the background (int / float)
        '''
        
        PlastimatchAdaptive.run('resample',
                                input = input_file,
                                output = output_file,
                                fixed = reference_image,
                                default_value = default_value)

    #%% Structure methods
    
    @staticmethod
    def expand_mask(input_mask, output_mask, distance=0):
        '''
        Expands mask.
        
        Args:
            input_mask --> path to input mask (str)
            output_mask --> path to output mask (str)
            distance --> expansion in mm (int / float)
        '''
        
        dirpath = os.path.dirname(input_mask)
        temp = os.path.join(dirpath, 'mask_dmap_temp.mha')
        
        PlastimatchAdaptive.run('dmap',
                                input = input_mask,
                                # algorithm = 'maurer',
                                output = temp)
        
        PlastimatchAdaptive.run('threshold',
                                input = temp,
                                output = output_mask,
                                below = distance)
        
        os.remove(temp)

    @staticmethod
    def invert_mask(input_mask, output_mask):
        '''
        Inverts mask.
        
        Args:
            input_mask --> path to input mask (str)
            output_mask --> path to output mask (str)
        '''
        
        PlastimatchAdaptive.run('threshold',
                                input = input_mask,
                                output = output_mask,
                                below = 0.5)

    @staticmethod
    def warp_mask(input_mask, output_mask, input_vf):
        '''
        Warps mask using an input vector field.
        This method does not lose voxels due to resampling.
        
        Args:
            input_mask --> path to input mask (str)
            output_mask --> path to output mask (str)
            input_vf --> path to input vector field (str)
        '''
        
        PlastimatchAdaptive.run('convert',
                                input = input_mask,
                                output_img = output_mask,
                                output_type = 'float')
        
        PlastimatchAdaptive.run('convert',
                                input = output_mask,
                                output_img = output_mask,
                                xf = input_vf)
        
        PlastimatchAdaptive.run('threshold',
                                input = output_mask,
                                output = output_mask,
                                above = 0.5)

    @staticmethod
    def get_union(output_mask, *masks):
        '''
        Generates union of input masks.
        
        Args:
            output_mask --> path to output mask (str)
            masks --> paths to input masks (str)
        '''
        
        if len(masks) < 2:
            raise Exception('Need at least two mask files.')
        
        dirpath = os.path.dirname(masks[0])
        temp = os.path.join(dirpath, 'mask_temp.mha')
        
        shutil.copyfile(masks[0], temp)
        
        for mask in masks[1:]:
            
            PlastimatchAdaptive.run('union',
                                    temp,
                                    mask,
                                    output = temp)
        
        shutil.move(temp, output_mask)

    @staticmethod
    def get_intersection(output_mask, *masks):
        '''
        Generates intersection of input masks.
        
        Args:
            output_mask --> path to output mask (str)
            masks --> paths to input masks (str)
        '''
        
        if len(masks) < 2:
            raise Exception('Need at least two mask files.')
        
        dirpath = os.path.dirname(masks[0])
        temp = os.path.join(dirpath, 'mask_temp.mha')
        
        shutil.copyfile(masks[0], temp)
        
        for mask in masks[1:]:
            
            PlastimatchAdaptive.run('add',
                                    temp,
                                    mask,
                                    output = temp)
    
        PlastimatchAdaptive.run('threshold',
                                input = temp,
                                output = output_mask,
                                above = len(masks))
        
        os.remove(temp)

    @staticmethod
    def exclude_masks(input_mask, output_mask, *excluded_masks):
        '''
        Excludes masks from "input_mask" (logical minus).
        
        Args:
            input_mask --> path to input mask (str)
            output_mask --> path to output mask (str)
            excluded_masks --> paths to masks to exclude from "input_mask" (str)
        '''
        
        if len(excluded_masks) < 1:
            raise Exception('Need at least one mask file to exclude.')
        
        dirpath = os.path.dirname(input_mask)
        temp = os.path.join(dirpath, 'mask_temp.mha')
        
        shutil.copyfile(input_mask, temp)
        
        for mask in excluded_masks:

            PlastimatchAdaptive.run('diff',
                                    temp,
                                    mask,
                                    temp)
        
        PlastimatchAdaptive.run('convert',
                                input = temp,
                                output_type = 'uchar',
                                output_img = output_mask)
        
        os.remove(temp)
    
    @staticmethod
    def get_empty_mask(reference_image, output_mask, value='zeros'):
        '''
        Creates empty mask based on reference image.
        
        Args:
            reference_image --> path to reference image (str)
            output_mask --> path to output mask (str)
            value --> value of output mask, 'zeros' or 'ones' (str)
        '''
    
        stats = PlastimatchAdaptive.get_stats(reference_image)
        
        if value == 'zeros':
            threshold_value = stats['MIN'] - 1
        
        elif value == 'ones':
            threshold_value = stats['MAX'] + 1
        
        PlastimatchAdaptive.run('threshold',
                                input = reference_image,
                                output = output_mask,
                                below = threshold_value)
    
    @staticmethod
    def get_bbox(input_mask, output_mask, margin):
        '''
        Generates a bounding box around input mask.
        
        Args:
            input_mask --> path to input mask (str)
            output_mask --> path to output mask (str)
            margin --> margin that is applied around input mask in mm (int / float)
        '''
        
        PlastimatchAdaptive.run('bbox',
                                input_mask,
                                margin = margin,
                                output = output_mask)
    
    @staticmethod
    def get_shell(input_mask, output_mask, distance):
        '''
        Generates a shell around input mask.
        Convention: (+) for outer shell, (-) for inner shell.
        
        Args:
            input_mask --> path to input mask (str)
            output_mask --> path to output mask (str)
            distance --> expansion in mm (int / float) or range by passing two values (list / tuple)
        '''
        
        dirpath = os.path.dirname(input_mask)
        temp = os.path.join(dirpath, 'mask_dmap_temp.mha')
        
        if type(distance) is list or type(distance) is tuple:
            range_str = '{},{}'.format(distance[0], distance[1])
            
        elif type(distance) is int or type(distance) is float:
            edge = 0.001
            range_str = '{},{}'.format(edge, distance) if distance > 0 else '{},{}'.format(distance, edge)
        
        PlastimatchAdaptive.run('dmap',
                                input = input_mask,
                                output = temp)
        
        PlastimatchAdaptive.run('threshold',
                                input = temp,
                                output = output_mask,
                                range = range_str)
        
        os.remove(temp)
    
    #%% Image registration methods

    @staticmethod
    def mask_to_img(input_mask, output_file, background=0, foreground=1, mode='mask', shell_distance=10):
        '''
        Converts binary mask file to synthetic image file.
        Output file can be used for image registration.
        
        input_mask --> path to input mask (str)
        output_file --> path to output file (str)
        background --> background value assigned to output image (int / float)
        foreground --> foreground value assigned to output image (int / float)
        mode --> 'mask'
             --> 'shell_outer'
             --> 'shell_inner'
             --> 'shell_uniform'
             --> 'shell_nonuniform'
        shell_distance --> distance in mm that is applied if a shell mode is used (int / float)
        '''
        
        dirpath = os.path.dirname(input_mask)
        temp = os.path.join(dirpath, 'temp.mha')
        
        PlastimatchAdaptive.run('synth',
                                background = background,
                                fixed = input_mask,
                                output = temp)
        
        if mode == 'mask':
            PlastimatchAdaptive.run('fill',
                                    input = temp,
                                    mask = input_mask,
                                    mask_value = foreground,
                                    output = output_file)
        elif mode == 'shell_outer':
            temp_shell = os.path.join(dirpath, 'temp_shell.mha')
            PlastimatchAdaptive.get_shell(input_mask, temp_shell, shell_distance)
            
            PlastimatchAdaptive.run('fill',
                                    input = temp,
                                    mask = temp_shell,
                                    mask_value = foreground,
                                    output = output_file)
            os.remove(temp_shell)
            
        elif mode == 'shell_inner':
            temp_shell = os.path.join(dirpath, 'temp_shell.mha')
            PlastimatchAdaptive.get_shell(input_mask, temp_shell, -shell_distance)
            
            PlastimatchAdaptive.run('fill',
                                    input = temp,
                                    mask = temp_shell,
                                    mask_value = foreground,
                                    output = output_file)
            os.remove(temp_shell)
            
        elif mode == 'shell_uniform':
            temp_shell_1 = os.path.join(dirpath, 'temp_shell_1.mha')
            temp_shell_2 = os.path.join(dirpath, 'temp_shell_2.mha')
            temp_shell   = os.path.join(dirpath, 'temp_shell.mha')
            PlastimatchAdaptive.get_shell(input_mask, temp_shell_1, shell_distance/2)
            PlastimatchAdaptive.get_shell(input_mask, temp_shell_2, -shell_distance/2)
            PlastimatchAdaptive.get_union(temp_shell, temp_shell_1, temp_shell_2)
            
            PlastimatchAdaptive.run('fill',
                                    input = temp,
                                    mask = temp_shell,
                                    mask_value = foreground,
                                    output = output_file)
            os.remove(temp_shell_1)
            os.remove(temp_shell_2)
            os.remove(temp_shell)

        elif mode == 'shell_nonuniform':
            temp_shell_1 = os.path.join(dirpath, 'temp_shell_1.mha')
            temp_shell_2 = os.path.join(dirpath, 'temp_shell_2.mha')
            PlastimatchAdaptive.get_shell(input_mask, temp_shell_1, shell_distance/2)
            PlastimatchAdaptive.get_shell(input_mask, temp_shell_2, -shell_distance/2)
            
            PlastimatchAdaptive.run('fill',
                                    input = temp,
                                    mask = temp_shell_2,
                                    mask_value = -foreground,
                                    output = temp)
            
            PlastimatchAdaptive.run('fill',
                                    input = temp,
                                    mask = temp_shell_1,
                                    mask_value = foreground,
                                    output = output_file)
            os.remove(temp_shell_1)
            os.remove(temp_shell_2)

        os.remove(temp)

    @staticmethod
    def register_deformable_bspline(fixed_image,
                                    moving_image,
                                    output_image=None,
                                    output_vf=None,
                                    fixed_mask=None,
                                    moving_mask=None,
                                    metric='mse',
                                    reg_factor=1,
                                    default_value=-1001):
        '''
        Writes Plastimatch command file and runs deformable image registration.
        
        Args:
            fixed_image --> path to input fixed image (str)
            moving_image --> path to input moving image (str)
            output_image --> path to output deformed image (str)
            output_vf --> path to output vector field (str)
            fixed_mask --> path to input fixed mask (str)
            moving_mask --> path to input moving mask (str)
            metric --> cost function metric to optimize (str)
            reg_factor --> regularization multiplier (int / float)
            default_value --> numeric value that is applied to the background (int / float)
        '''

        dirpath = os.path.dirname(moving_image)
        command_file_path = os.path.join(dirpath, 'register_bspline_command_file.txt')
    
        with open(command_file_path, 'w') as f:
            
            f.write(PlastimatchAdaptive.image_registration_global(fixed = fixed_image,
                                                                  moving = moving_image,
                                                                  img_out = output_image,
                                                                  vf_out = output_vf,
                                                                  fixed_mask = fixed_mask,
                                                                  moving_mask = moving_mask,
                                                                  default_value = default_value))
            
            f.write(PlastimatchAdaptive.image_registration_stage(xform = 'bspline',
                                                                 optim = 'lbfgsb',
                                                                 impl = 'plastimatch',
                                                                 threading = 'cuda',
                                                                 max_its = 50,
                                                                 grid_spac = '100 100 100',
                                                                 res = '8 8 4',
                                                                 regularization_lambda = 1*reg_factor,
                                                                 metric = metric))
                                                                  
            f.write(PlastimatchAdaptive.image_registration_stage(xform = 'bspline',
                                                                 optim = 'lbfgsb',
                                                                 impl = 'plastimatch',
                                                                 threading = 'cuda',
                                                                 max_its = 50,
                                                                 grid_spac = '80 80 80',
                                                                 res = '4 4 2',
                                                                 regularization_lambda = 0.1*reg_factor,
                                                                 metric = metric))
                    
            f.write(PlastimatchAdaptive.image_registration_stage(xform = 'bspline',
                                                                 optim = 'lbfgsb',
                                                                 impl = 'plastimatch',
                                                                 threading = 'cuda',
                                                                 max_its = 40,
                                                                 grid_spac = '60 60 60',
                                                                 res = '2 2 1',
                                                                 regularization_lambda = 0.1*reg_factor,
                                                                 metric = metric))
            
            f.write(PlastimatchAdaptive.image_registration_stage(xform = 'bspline',
                                                                 optim = 'lbfgsb',
                                                                 impl = 'plastimatch',
                                                                 threading = 'cuda',
                                                                 max_its = 40,
                                                                 grid_spac = '20 20 20',
                                                                 res = '1 1 1',
                                                                 regularization_lambda = 0.05*reg_factor,
                                                                 metric = metric))
            
            f.write(PlastimatchAdaptive.image_registration_stage(xform = 'bspline',
                                                                 optim = 'lbfgsb',
                                                                 impl = 'plastimatch',
                                                                 threading = 'cuda',
                                                                 max_its = 40,
                                                                 grid_spac = '10 10 10',
                                                                 res = '1 1 1',
                                                                 regularization_lambda = 0.01*reg_factor,
                                                                 metric = metric))
        
        PlastimatchAdaptive.run(command_file_path)
        os.remove(command_file_path)

    @staticmethod
    def register_3_DOF(fixed_image,
                       moving_image,
                       output_image=None,
                       output_vf=None,
                       fixed_mask=None,
                       moving_mask=None,
                       metric='mse',
                       default_value=-1001):
        '''
        Writes Plastimatch command file and runs 3-DOF image registration.
        
        Args:
            fixed_image --> path to input fixed image (str)
            moving_image --> path to input moving image (str)
            output_image --> path to output deformed image (str)
            output_vf --> path to output vector field (str)
            fixed_mask --> path to input fixed mask (str)
            moving_mask --> path to input moving mask (str)
            metric --> cost function metric to optimize (str)
            default_value --> numeric value that is applied to the background (int / float)
        '''

        dirpath = os.path.dirname(moving_image)
        command_file_path = os.path.join(dirpath, 'register_3_DOF_command_file.txt')

        with open(command_file_path, 'w') as f:
            
            f.write(PlastimatchAdaptive.image_registration_global(fixed = fixed_image,
                                                                  moving = moving_image,
                                                                  img_out = output_image,
                                                                  vf_out = output_vf,
                                                                  fixed_mask = fixed_mask,
                                                                  moving_mask = moving_mask,
                                                                  default_value = default_value))
            
            f.write(PlastimatchAdaptive.image_registration_stage(xform = 'align_center'))
            
            f.write(PlastimatchAdaptive.image_registration_stage(xform = 'translation',
                                                                 optim = 'rsg',
                                                                 threading = 'cuda',
                                                                 max_its = 200,
                                                                 res = '8 8 4',
                                                                 metric = metric))
                                                                  
            f.write(PlastimatchAdaptive.image_registration_stage(xform = 'translation',
                                                                 optim = 'rsg',
                                                                 threading = 'cuda',
                                                                 max_its = 200,
                                                                 res = '4 4 2',
                                                                 metric = metric))
                    
            f.write(PlastimatchAdaptive.image_registration_stage(xform = 'translation',
                                                                 optim = 'rsg',
                                                                 threading = 'cuda',
                                                                 max_its = 200,
                                                                 res = '2 2 1',
                                                                 metric = metric))
        
        PlastimatchAdaptive.run(command_file_path)
        os.remove(command_file_path)

    @staticmethod
    def register_6_DOF(fixed_image,
                       moving_image,
                       output_image=None,
                       output_vf=None,
                       fixed_mask=None,
                       moving_mask=None,
                       metric='mse',
                       default_value=-1001):
        '''
        Writes Plastimatch command file and runs 6-DOF image registration.
        
        Args:
            fixed_image --> path to input fixed image (str)
            moving_image --> path to input moving image (str)
            output_image --> path to output deformed image (str)
            output_vf --> path to output vector field (str)
            fixed_mask --> path to input fixed mask (str)
            moving_mask --> path to input moving mask (str)
            metric --> cost function metric to optimize (str)
            default_value --> numeric value that is applied to the background (int / float)
        '''

        dirpath = os.path.dirname(moving_image)
        command_file_path = os.path.join(dirpath, 'register_6_DOF_command_file.txt')

        with open(command_file_path, 'w') as f:
            
            f.write(PlastimatchAdaptive.image_registration_global(fixed = fixed_image,
                                                                  moving = moving_image,
                                                                  img_out = output_image,
                                                                  vf_out = output_vf,
                                                                  fixed_mask = fixed_mask,
                                                                  moving_mask = moving_mask,
                                                                  default_value = default_value))
            
            f.write(PlastimatchAdaptive.image_registration_stage(xform = 'align_center'))
            
            f.write(PlastimatchAdaptive.image_registration_stage(xform = 'rigid',
                                                                 optim = 'versor',
                                                                 threading = 'cuda',
                                                                 max_its = 200,
                                                                 res = '8 8 4',
                                                                 metric = metric))
                                                                  
            f.write(PlastimatchAdaptive.image_registration_stage(xform = 'rigid',
                                                                 optim = 'versor',
                                                                 threading = 'cuda',
                                                                 max_its = 200,
                                                                 res = '4 4 2',
                                                                 metric = metric))
                    
            f.write(PlastimatchAdaptive.image_registration_stage(xform = 'rigid',
                                                                 optim = 'versor',
                                                                 threading = 'cuda',
                                                                 max_its = 200,
                                                                 res = '2 2 1',
                                                                 metric = metric))
        
        PlastimatchAdaptive.run(command_file_path)
        os.remove(command_file_path)
        
    @staticmethod
    def match_position_3_DOF(fixed_image,
                             moving_image,
                             output_image,
                             output_vf,
                             metric='mi'):
        '''
        Matches patient position with a 3-DOF vector field.
        Retains original image dimensions/size/spacing.
        For big offsets, metric='mi' works better. (?)
        
        Args:
            fixed_image --> path to input fixed image (str)
            moving_image --> path to input moving image (str)
            output_image --> path to output image (str)
            output_vf --> path to output vector field (str)
            metric --> cost function metric to optimize (str)
        '''
        
        dirpath = os.path.dirname(output_vf)
        vf_temp_path = os.path.join(dirpath, 'vf_temp.mha')
        
        PlastimatchAdaptive.register_3_DOF(fixed_image,
                                           moving_image,
                                           output_vf = vf_temp_path,
                                           metric = metric)
        
        vf_temp = TranslationVF(vf_temp_path)
        x,y,z = vf_temp.get_shifts()
        os.remove(vf_temp.path)
        
        PlastimatchAdaptive.apply_manual_translation(moving_image,
                                                     output_image,
                                                     output_vf,
                                                     -x, -y, -z,
                                                     unit = 'mm',
                                                     frame = 'shift',
                                                     discrete_voxels = True)
    
    @staticmethod
    def apply_vf_to_contours(input_contours,
                             output_contours,
                             input_vf):
        '''
        Applies input vector field to all contours in directory.
        
        Args:
            input_contours --> path to directory containing input structures (str)
            output_contours --> path to directory to store deformed structures (str)
            input_vf --> path to input vector field (str)
        '''
        
        for filename in os.listdir(input_contours):
            if os.path.isfile(os.path.join(input_contours, filename)):
                PlastimatchAdaptive.warp_mask(os.path.join(input_contours, filename), 
                                              os.path.join(output_contours, filename),
                                              input_vf)
    
    @staticmethod
    def propagate_contours(input_contours,
                           output_contours,
                           fixed_image,
                           moving_image,
                           output_image=None,
                           output_vf=None,
                           fixed_mask=None,
                           moving_mask=None,
                           reg_factor = 1,
                           translate_first=True):
        '''
        Propagates contours from one image to another.
        
        Args:
            input_contours --> path to directory containing input structures (str)
            output_contours --> path to directory to store deformed structures (str)
            fixed_image --> path to input fixed image (str)
            moving_image --> path to input moving image (str)
            output_image --> path to output deformed image (str)
            output_vf --> path to output vector field (str)
            fixed_mask --> path to input fixed mask (str)
            moving_mask --> path to input moving mask (str)
            reg_factor --> regularization multiplier (int / float)
            translate_first --> option to match images in 3D before running DIR (bool)
        '''
        
        dirpath = os.path.dirname(moving_image)
        var = os.path.splitext(moving_image)
        
        deformation_vf = os.path.join(dirpath, 'vf_dir.mha')
        
        if translate_first is True:
            
            translated_img = var[0] + '_translated' + var[1]
            translation_vf = os.path.join(dirpath, 'vf_3_DOF.mha')
            temp_contours = os.path.join(dirpath, 'temp_contours')
            
            PlastimatchAdaptive.match_position_3_DOF(fixed_image,
                                                     moving_image,
                                                     translated_img,
                                                     translation_vf,)
                                                     # metric='mi')
            
            PlastimatchAdaptive.apply_vf_to_contours(input_contours,
                                                     temp_contours,
                                                     translation_vf)
                
            input_contours = temp_contours
            moving_image = translated_img
        
        PlastimatchAdaptive.register_deformable_bspline(fixed_image,
                                                        moving_image,
                                                        output_image,
                                                        deformation_vf,
                                                        fixed_mask,
                                                        moving_mask,
                                                        reg_factor = reg_factor)
        
        PlastimatchAdaptive.apply_vf_to_contours(input_contours,
                                                 output_contours,
                                                 deformation_vf)
        
        if translate_first is True:
            if output_vf is not None:
                PlastimatchAdaptive.run('compose',
                                        translation_vf,
                                        deformation_vf,
                                        output_vf)
            os.remove(translated_img)
            os.remove(translation_vf)
            shutil.rmtree(temp_contours)
        
        if translate_first is False:
            if output_vf is not None:
                shutil.copyfile(deformation_vf, output_vf)
        
        os.remove(deformation_vf)
            
        print('\nContour propagation: DONE.')

    #%% Histogram-based correction: use for CBCT HU correction
    
    @staticmethod
    def values_histogram_matching(input_image,
                                  reference_image,
                                  output_image,
                                  input_image_masks=None,
                                  reference_image_masks=None,
                                  output_image_masks=None,
                                  histogram_threshold_min=-500,
                                  histogram_threshold_max=1500,
                                  correction_type='mean'):
        '''
        Modifies values of input image to match the histogram of reference image.
        
        Args:
            input_image             --> path to input image (str)
            reference_image         --> path to reference image (str)
            output_image            --> path to output image (str)
            input_image_masks       --> paths to masks specifying the ROI for the input image histogram (list / tuple)
            reference_image_masks   --> paths to masks specifying the ROI for the reference image histogram (list / tuple)
            output_image_masks      --> paths to masks specifying the ROI where in the input image value matching is applied (list / tuple)
            histogram_threshold_min --> lower threshold for the values to be considered in the HU correction (int / float)
            histogram_threshold_max --> upper threshold for the values to be considered in the HU correction (int / float)
            correction_type         --> specifies which correction method is used (str)
                                    --> 'full' will apply a piecewise linear correction based on cumulative histograms
                                    --> 'mean' will shift values based on the mean of the histogram
                                    --> 'median' will shift values based on the median of the histogram
                                    --> 'none' will just read and compare the histograms of the two image files without any corrections
        '''
        dirpath = os.path.dirname(output_image)
        histogram_matching_dir = os.path.join(dirpath, 'histogram_matching_{}'.format(correction_type))
        if not os.path.exists(histogram_matching_dir):
            os.makedirs(histogram_matching_dir)
        input_image_temp = os.path.join(dirpath, 'input_temp.mha')
        reference_image_temp = os.path.join(dirpath, 'reference_temp.mha')
        # Mask image files
        if input_image_masks is not None:
            input_mask_temp = os.path.join(dirpath, 'input_mask_temp.mha')
            if len(input_image_masks) > 1:
                PlastimatchAdaptive.get_union(input_mask_temp, *input_image_masks)
            elif len(input_image_masks) == 1:
                shutil.copyfile(input_image_masks[0], input_mask_temp)
            PlastimatchAdaptive.mask_image(input_image,
                                           input_image_temp,
                                           input_mask_temp,
                                           histogram_threshold_min-1)
        if reference_image_masks is not None:
            reference_mask_temp = os.path.join(dirpath, 'reference_mask_temp.mha')
            if len(reference_image_masks) > 1:
                PlastimatchAdaptive.get_union(reference_mask_temp, *reference_image_masks)
            elif len(reference_image_masks) == 1:
                shutil.copyfile(reference_image_masks[0], reference_mask_temp)
            PlastimatchAdaptive.mask_image(reference_image,
                                           reference_image_temp,
                                           reference_mask_temp,
                                           histogram_threshold_min-1)
        # Load arrays
        arr_1 = RTArray(reference_image_temp).array_1D
        arr_2 = RTArray(input_image_temp).array_1D
        arr_1 = np.delete(arr_1, np.where(arr_1 < histogram_threshold_min))
        arr_2 = np.delete(arr_2, np.where(arr_2 < histogram_threshold_min))
        arr_1 = np.delete(arr_1, np.where(arr_1 > histogram_threshold_max))
        arr_2 = np.delete(arr_2, np.where(arr_2 > histogram_threshold_max))
        n_bins = int(histogram_threshold_max - histogram_threshold_min)
        # Generate plot 1:
        reference_name = os.path.splitext(os.path.basename(reference_image))[0]
        input_name = os.path.splitext(os.path.basename(input_image))[0]
        fig, axes = plt.subplots(2)
        counts_arr_1, range_arr_1, bar_container = axes[1].hist(arr_1, bins = n_bins,      cumulative=True, alpha=0.7, label=reference_name)
        counts_arr_2, range_arr_2, bar_container = axes[1].hist(arr_2, bins = range_arr_1, cumulative=True, alpha=0.7, label=input_name)
        axes[0].hist(arr_1, bins = n_bins,      cumulative=False, alpha=0.7, label=reference_name)
        axes[0].hist(arr_2, bins = range_arr_1, cumulative=False, alpha=0.7, label=input_name)
        axes[0].axvline(np.mean(arr_1), ls='--', c='darkblue', alpha=0.8, linewidth=1, label='{} mean'.format(reference_name), zorder=-1)
        axes[0].axvline(np.mean(arr_2), ls='--', c='maroon',   alpha=0.8, linewidth=1, label='{} mean'.format(input_name),     zorder=-1)
        axes[0].legend()
        axes[0].set_xlabel('Image values (a.u.)')
        axes[1].set_xlabel('Image values (a.u.)')
        fig.set_dpi(100)
        plt.tight_layout()
        plt.savefig(os.path.join(histogram_matching_dir, 'histograms_pre_matching.png'))
        plt.show()
        os.remove(input_image_temp)
        os.remove(reference_mask_temp)
        if correction_type == 'none':
            os.remove(reference_image_temp)
            os.remove(input_mask_temp)
            return
        # Method 1: Piecewise linear correction based on cumulative histograms
        elif correction_type == 'full':
            f = interp1d(counts_arr_1, range_arr_1[:-1], fill_value='extrapolate')
            new_image_values = []
            conversion_str = ''
            for arr_1_value in range_arr_1[:-1]:
                iterator = np.where(range_arr_1 == arr_1_value)[0][0]
                arr_2_value = float(f(counts_arr_2[iterator]))
                new_image_values.append(arr_2_value)
                conversion_str = conversion_str + '{},{},'.format(arr_1_value, arr_2_value)
            g = interp1d(range_arr_1[:-1], new_image_values)
            plt.figure(dpi=100)
            plt.plot(range_arr_1[:-1], g(range_arr_1[:-1]))
            plt.plot(range_arr_1[:-1], range_arr_1[:-1], alpha = 0.7, ls='--')
            plt.grid(ls='--')
            plt.xlabel('Values old')
            plt.ylabel('Values new')
            plt.savefig(os.path.join(histogram_matching_dir, 'histogram_matching_curve.png'))
            plt.show()
            PlastimatchAdaptive.run('adjust',
                                    input = input_image,
                                    output = output_image,
                                    pw_linear = conversion_str[:-1])
            with open(os.path.join(histogram_matching_dir, 'histogram_matching_info.txt'), 'w') as f:
                f.write('Applied piece-wise linear correction.\n')
                f.write('Function:\n')
                f.write(conversion_str[:-1])
        # Method 2: Shifting values based on mean or median
        else:
            values_temp = os.path.join(dirpath, 'values.mha')
            if correction_type == 'mean':
                difference = np.mean(arr_1) - np.mean(arr_2)
            if correction_type == 'median':
                difference = np.median(arr_1) - np.median(arr_2)
            PlastimatchAdaptive.run('synth',
                                    fixed = input_image,
                                    output = values_temp,
                                    background = str(difference),
                                    foreground = str(difference))
            PlastimatchAdaptive.sum_images(output_image, input_image, values_temp)
            with open(os.path.join(histogram_matching_dir, 'histogram_matching_info.txt'), 'w') as f:
                f.write('Applied shift based on {}.\n'.format(correction_type))
                f.write('Function:\n')
                f.write('(+) {}'.format(difference))
            os.remove(values_temp)
        # Compare histograms for corrected image --> generate plot 2
        if input_image_masks is not None:
            output_image_temp = os.path.join(dirpath, 'output_image_temp.mha')
            PlastimatchAdaptive.mask_image(output_image,
                                           output_image_temp,
                                           input_mask_temp,
                                           histogram_threshold_min-1)
            arr_2 = RTArray(output_image_temp).array_1D
            os.remove(output_image_temp)
            os.remove(input_mask_temp)
        else:
            arr_2 = RTArray(output_image).array_1D
        arr_1 = RTArray(reference_image_temp).array_1D
        os.remove(reference_image_temp)
        arr_1 = np.delete(arr_1, np.where(arr_1 < histogram_threshold_min))
        arr_2 = np.delete(arr_2, np.where(arr_2 < histogram_threshold_min))
        arr_1 = np.delete(arr_1, np.where(arr_1 > histogram_threshold_max))
        arr_2 = np.delete(arr_2, np.where(arr_2 > histogram_threshold_max))
        n_bins = int(histogram_threshold_max - histogram_threshold_min)
        # Generate plot 1:
        reference_name = os.path.splitext(os.path.basename(reference_image))[0]
        input_name = os.path.splitext(os.path.basename(input_image))[0]
        fig, axes = plt.subplots(2)
        counts_arr_1, range_arr_1, bar_container = axes[1].hist(arr_1, bins = n_bins,      cumulative=True, alpha=0.7, label=reference_name)
        counts_arr_2, range_arr_2, bar_container = axes[1].hist(arr_2, bins = range_arr_1, cumulative=True, alpha=0.7, label=input_name)
        axes[0].hist(arr_1, bins = n_bins,      cumulative=False, alpha=0.7, label=reference_name)
        axes[0].hist(arr_2, bins = range_arr_1, cumulative=False, alpha=0.7, label=input_name)
        axes[0].axvline(np.mean(arr_1), ls='--', c='darkblue', alpha=0.8, linewidth=1, label='{} mean'.format(reference_name), zorder=-1)
        axes[0].axvline(np.mean(arr_2), ls='--', c='maroon',   alpha=0.8, linewidth=1, label='{} mean'.format(input_name),     zorder=-1)
        axes[0].legend()
        axes[0].set_xlabel('Image values (a.u.)')
        axes[1].set_xlabel('Image values (a.u.)')
        fig.set_dpi(100)
        plt.tight_layout()
        plt.savefig(os.path.join(histogram_matching_dir, 'histograms_post_matching.png'))
        plt.show()
        # Getting final image
        if output_image_masks is not None:
            PlastimatchAdaptive.merge_images(input_image,
                                             output_image,
                                             output_image,
                                             *output_image_masks)
            