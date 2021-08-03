# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 10:28:25 2021

@author: MBo
"""

import os
import shutil
import subprocess
# from RTArrays import *
from RTArrays import RTArray, Image, ImageCT, ImageCBCT, ImageMRI, DoseMap, Structure, VectorField, TranslationVF, RigidVF, BSplineVF

'''
Plastimatch extension to support the adaptive proton therapy project.
'''

class PlastimatchAdaptive(object):
    
    def __init__(self):
        pass

    def __input_check(input_file, supported_cls):
        '''
        Method to check input.
        
        Args:
            input_file --> instance of RTArray class
            supported_cls --> list or tuple of supported classes
        '''
        if not any([issubclass(input_file.__class__, i) for i in supported_cls]):
            raise TypeError('Wrong input')
        else:
            return True
        
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
            input_file --> instance of RTArray class
        '''
        supported_cls = (RTArray,)
        PlastimatchAdaptive.__input_check(input_file, supported_cls)
        
        PlastimatchAdaptive.run('stats', input_file.path)
        
    @staticmethod
    def get_stats(input_file):
        '''
        Returns stats of input image, dose, or structure as a dictionary.
        
        Args:
            input_file --> instance of RTArray class (except VectorField)
        '''
        #TODO: Make it work for VectorField

        supported_cls = (Image, DoseMap, Structure)
        PlastimatchAdaptive.__input_check(input_file, supported_cls)
        
        stats  = PlastimatchAdaptive.run('stats', input_file.path)
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
        PlastimatchAdaptive.__input_check(input_file, supported_cls)
        
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
    
        origin_new = '{} {} {}'.format(origin_x_new, origin_y_new, origin_z_new)
        size_new = '{} {} {}'.format(size_x_new, size_y_new, size_z_new)
        spacing_new = '{} {} {}'.format(spacing_x_new, spacing_y_new, spacing_z_new)
    
        PlastimatchAdaptive.run('resample',
                                input = input_file.path,
                                output = output_file_path,
                                origin = origin_new,
                                dim = size_new,
                                spacing = spacing_new,
                                default_value = default_value)
        
        if unit == 'mm':
            print('\nExtend/crop input converted from [mm] to [vox]:'\
                  '\nx_lower={}\nx_upper={}\ny_lower={}\ny_upper={}\nz_lower={}\nz_upper={}\n'\
                  .format(x_lower_vox, x_upper_vox, y_lower_vox, y_upper_vox, z_lower_vox, z_upper_vox))
    
        return input_file.__class__(output_file_path)
    
    @staticmethod
    def merge_images(background, foreground, output_file_path, *masks):
        '''
        Merges two images (foreground on background).
        Masks (Structure class) specify voxels where the foreground is applied.
        Returns RTArray object for the output file (same class as foreground).
        
        Args:
            background --> instance of RTArray class for background image file
            foreground --> instance of RTArray class for foreground image file
            output_file_path --> path to output image file (string)
            masks --> instances of Structure class for masks
        '''
        supported_cls = (RTArray,)
        PlastimatchAdaptive.__input_check(background, supported_cls)
        PlastimatchAdaptive.__input_check(foreground, supported_cls)
        
        dirpath = os.path.dirname(foreground.path)
        temp_mask = os.path.join(dirpath, 'mask_temp.mha')
        temp_foreground = os.path.join(dirpath, 'foreground_temp.mha')
        temp_background = os.path.join(dirpath, 'background_temp.mha')
        
        if len(masks) == 0:
            raise Exception('Need at least one mask file.')
            
        elif len(masks) == 1:
            shutil.copyfile(masks[0].path, temp_mask)
            
        else:
            PlastimatchAdaptive.get_union(temp_mask, *masks)
        
        PlastimatchAdaptive.run('mask',
                                input = foreground.path,
                                mask = temp_mask,
                                mask_value = 0,
                                output = temp_foreground)
        
        PlastimatchAdaptive.run('fill',
                                input = background.path,
                                mask = temp_mask,
                                mask_value = 0,
                                output = temp_background)

        PlastimatchAdaptive.run('add', temp_foreground, temp_background,
                                output = output_file_path)
        
        os.remove(temp_foreground)
        os.remove(temp_background)
        os.remove(temp_mask)
        
        return foreground.__class__(output_file_path)
    
    @staticmethod
    def scale_image_linear(input_file, output_file_path, *parameters):
        '''
        Applies piece-wise linear function to image.
        Returns RTArray object for the output file (same class as input_file).
        
        Args:
            input_file --> instance of Image or DoseMap
            output_file_path --> path to output file (string)
            parameters --> pair-wise values for linear function (int or float)
        '''      
        # Example values used before: "-1024,-1024,261,58" (261 from CBCT, 58 from CT)
        
        supported_cls = (Image, DoseMap)
        PlastimatchAdaptive.__input_check(input_file, supported_cls)
        
        transform_str = ''
        
        for val in parameters:
            transform_str = transform_str + '{},'.format(val)
        
        PlastimatchAdaptive.run('adjust',
                                input = input_file.path,
                                output = output_file_path,
                                pw_linear = transform_str)

        return input_file.__class__(output_file_path)

    @staticmethod
    def scale_image_factor(input_file, output_file_path, factor):
        '''
        Multiplies factor to image voxel by voxel.
        Slow implementation! Use "apply_image_weight" instead.
        Returns RTArray object for the output file (same class as input_file).
        
        Args:
            input_file --> instance of Image or DoseMap
            output_file_path --> path to output file (string)
            factor --> multiplication factor (float or int)
        '''     
        supported_cls = (Image, DoseMap)
        PlastimatchAdaptive.__input_check(input_file, supported_cls)

        stats = PlastimatchAdaptive.get_stats(input_file) 
        
        min_value = int(stats['MIN'])
        max_value = int(stats['MAX']+1)
        
        values_ini =   [i for i in range(min_value, max_value+1)]
        values_mod =   [i*factor for i in values_ini]
        values_input = [j for k in zip(values_ini, values_mod) for j in k]
        
        PlastimatchAdaptive.scale_image_linear(input_file, output_file_path, *values_input)

        return input_file.__class__(output_file_path)
    
    @staticmethod
    def apply_image_weight(input_file, output_file_path, weight):
        '''
        Applies weight to image.
        Faster implementation than "scale_image_factor".
        Returns RTArray object for the output file (same class as input_file).
        
        Args:
            input_file --> instance of Image or DoseMap
            output_file_path --> path to output file (string)
            weight --> multiplication weight (float or int)
        '''     
        supported_cls = (Image, DoseMap)
        PlastimatchAdaptive.__input_check(input_file, supported_cls)

        PlastimatchAdaptive.run('add',
                                input_file.path,
                                weight = weight,
                                output = output_file_path)

        return input_file.__class__(output_file_path)

    @staticmethod
    def apply_manual_translation(input_file,
                                 output_file_path,
                                 output_vf_path,
                                 x, y, z,
                                 unit='mm',
                                 frame='shift',
                                 discrete_voxels=False):
        '''
        Applies manual translation to input image, dose, structure or vector field.
        Returns RTArray object for output file as well as TranslationVF.
        
        Args:
            input_file --> instance of RTArray class
            output_file_path --> path to output file (string)
            output_vf_path --> path to output vector field (string)
            x, y, z --> parameters for 3D shift in mm (int or float)
            unit --> specifies unit for shift distance, "vox" or "mm" (str)
            frame --> specifies how image frame is handled, "shift" or "fix" (str)
            discrete_voxels --> option to apply shift in discrete number of voxels (bool)
        '''
        supported_cls = (RTArray,)
        PlastimatchAdaptive.__input_check(input_file, supported_cls)
        
        dirpath = os.path.dirname(input_file.path)
        temp_path = os.path.join(dirpath, 'ext_temp.mha')
        translation_vf = TranslationVF(output_vf_path)
        
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
        
        PlastimatchAdaptive.extend_or_crop(input_file,
                                           temp_path,
                                           x_lower, x_upper,
                                           y_lower, y_upper,
                                           z_lower, z_upper,
                                           unit)
        if unit == 'vox':
            x = x * input_file.spacing_x
            y = y * input_file.spacing_y
            z = z * input_file.spacing_z
        
        translation_str = '{} {} {}'.format(x, y, z)
        
        PlastimatchAdaptive.run('synth-vf',
                                fixed = temp_path,
                                xf_trans = translation_str,
                                output = translation_vf.path)
        
        if input_file.__class__ is Structure:            
            PlastimatchAdaptive.warp_mask(input_file,
                                          output_file_path,
                                          translation_vf)
        else:
            PlastimatchAdaptive.warp_image(input_file,
                                           output_file_path,
                                           translation_vf)
        
        os.remove(temp_path)
    
        if frame == 'shift':
            PlastimatchAdaptive.extend_or_crop(input_file.__class__(output_file_path),
                                               output_file_path,
                                               -x_upper, -x_lower,
                                               -y_upper, -y_lower,
                                               -z_upper, -z_lower,
                                               unit)
        
        elif frame == 'fix':
            PlastimatchAdaptive.extend_or_crop(input_file.__class__(output_file_path),
                                               output_file_path,
                                               -x_lower, -x_upper,
                                               -y_lower, -y_upper,
                                               -z_lower, -z_upper,
                                               unit)
        
        print('\nApplied translation:\n'\
              'x = {} mm | = {} voxels\n'\
              'y = {} mm | = {} voxels\n'\
              'z = {} mm | = {} voxels'\
              .format(-x, -x/input_file.spacing_x,
                      -y, -y/input_file.spacing_y,
                      -z, -z/input_file.spacing_z))
            
        return input_file.__class__(output_file_path), translation_vf

    @staticmethod
    def warp_image(input_file, output_file_path, vf_file, default_value=None):
        '''
        Warps image using an input vector field.
        For Structure use "warp_mask" instead.
        Returns RTArray object for the output file (same class as input_file).
        
        Args:
            input_file --> instance of RTArray class (except Structure)
            output_file_path --> path to output file (string)
            vf_file --> instance of VectorField class
            default_value --> numeric value that is applied to the background (float)
                          --> will be applied according to class if not specified
        '''
        supported_cls = (Image, DoseMap, VectorField)
        PlastimatchAdaptive.__input_check(input_file, supported_cls)
        PlastimatchAdaptive.__input_check(vf_file, (VectorField,))
        
        default_value = input_file.base_value if default_value == None else default_value
        
        PlastimatchAdaptive.run('convert',
                                input = input_file.path,
                                output_img = output_file_path,
                                xf = vf_file.path,
                                default_value = default_value,
                                # algorithm = 'itk',
                                # output_type = 'float'
                                )
        
        return input_file.__class__(output_file_path)
    
    @staticmethod
    def mask_image(input_file, output_file_path, mask, mask_value=None):
        '''
        Masks image using an input mask.
        Returns RTArray object for the output file (same class as input_file).
        
        Args:
            input_file --> instance of RTArray class (except Structure)
            output_file_path --> path to output file (string)
            mask --> instance of Structure class
            mask_value --> numeric value that is applied outside the mask volume (float)
                       --> will be applied according to class if not specified
        '''
        supported_cls = (Image, DoseMap, VectorField)
        PlastimatchAdaptive.__input_check(input_file, supported_cls)
        PlastimatchAdaptive.__input_check(mask, (Structure,))
        
        mask_value = input_file.base_value if mask_value == None else mask_value
        
        PlastimatchAdaptive.run('mask',
                                input = input_file.path,
                                output = output_file_path,
                                mask = mask.path,
                                mask_value = mask_value)
        
        return input_file.__class__(output_file_path)

    @staticmethod
    def fill_image(input_file, output_file_path, mask, mask_value=None):
        '''
        Fills image using an input mask.
        Returns RTArray object for the output file (same class as input_file).
        
        Args:
            input_file --> instance of RTArray class (except Structure)
            output_file_path --> path to output file (string)
            mask --> instance of Structure class
            mask_value --> numeric value that is applied inside the mask volume (float)
                       --> will be applied according to class if not specified
        '''
        supported_cls = (Image, DoseMap, VectorField)
        PlastimatchAdaptive.__input_check(input_file, supported_cls)
        PlastimatchAdaptive.__input_check(mask, (Structure,))
        
        mask_value = input_file.base_value if mask_value == None else mask_value
        
        PlastimatchAdaptive.run('fill',
                                input = input_file.path,
                                output = output_file_path,
                                mask = mask.path,
                                mask_value = mask_value)
        
        return input_file.__class__(output_file_path)

    @staticmethod
    def DICOM_to_ITK(input_dicom, output_image=None, structures=None, dose_map=None):
        '''
        Converts DICOM to ITK.
        
        Args:
            input_dicom --> path to input dicom folder or file (string)
            output_image --> path to output image file (string)
            structures --> path to folder for structures (string)
            dose_map --> path to output dose image file (string)
        '''
        # TODO: modify method once we have appropriate classes like e.g. Patient, StructureSet, DicomFolder etc.
        
        PlastimatchAdaptive.run('convert',
                                input = input_dicom,
                                output_type = 'float',
                                output_img = output_image,
                                output_prefix = structures,
                                output_dose_img = dose_map)
    
    def ITK_to_DICOM(output_dicom_folder, input_image=None, structures=None, dose_map=None):
        '''
        Converts ITK to DICOM.
        
        Args:
            output_dicom_folder --> path to output dicom folder (string)
            input_image --> path to input image file (string)
            structures --> path to folder for structures (string)
            dose_map --> path to input dose image file (string)
        '''
        # TODO: modify method once we have appropriate classes like e.g. Patient, StructureSet, DicomFolder etc.
        
        PlastimatchAdaptive.run('convert',
                                input = input_image,
                                input_prefix = structures,
                                input_dose_img = dose_map,
                                output_dicom = output_dicom_folder)

    @staticmethod
    def resample_to_reference(input_file, output_file_path, reference_image):
        '''
        Resamples RTArray based on reference image.
        Returns RTArray object for the output file (same class as input_file).
        
        Args:
            input_file --> instance of RTArray class
            output_file_path --> path to output file (string)
            reference_image --> instance of RTArray class
        '''
        supported_cls = (RTArray,)
        PlastimatchAdaptive.__input_check(input_file, supported_cls)
        PlastimatchAdaptive.__input_check(reference_image, supported_cls)
        
        PlastimatchAdaptive.run('resample',
                                input = input_file.path,
                                output = output_file_path,
                                fixed = reference_image.path)
        
        return input_file.__class__(output_file_path)

    #%% Structure methods
    
    @staticmethod
    def expand_mask(input_mask, output_mask_path, distance=0):
        '''
        Expands mask.
        Returns Structure object for output mask file.
        
        Args:
            input_mask --> instance of Structure class
            output_mask_path --> path to output mask file (string)
            distance --> expansion in mm (int or float)
        '''
        supported_cls = (Structure,)
        PlastimatchAdaptive.__input_check(input_mask, supported_cls)
        
        dirpath = os.path.dirname(input_mask.path)
        temp = os.path.join(dirpath, 'mask_dmap_temp.mha')
        
        PlastimatchAdaptive.run('dmap',
                                input = input_mask.path,
                                # algorithm = 'maurer',
                                output = temp)
        
        PlastimatchAdaptive.run('threshold',
                                input = temp,
                                output = output_mask_path,
                                below = distance)
        
        os.remove(temp)
        
        return Structure(output_mask_path)

    @staticmethod
    def invert_mask(input_mask, output_mask_path):
        '''
        Inverts mask.
        Returns Structure object for output mask file.
        
        Args:
            input_mask --> instance of Structure class
            output_mask_path --> path to output mask file (string)
        '''
        supported_cls = (Structure,)
        PlastimatchAdaptive.__input_check(input_mask, supported_cls)
        
        PlastimatchAdaptive.run('threshold',
                                input = input_mask.path,
                                output = output_mask_path,
                                below = 0.5)
        
        return Structure(output_mask_path)

    @staticmethod
    def warp_mask(input_mask, output_mask_path, vf_file):
        '''
        Warps mask using an input vector field.
        This method does not lose voxels due to resampling.
        Returns Structure object for output mask file.
        
        Args:
            input_mask --> instance of Structure class
            output_mask_path --> path to output mask file (string)
            vf_file --> instance of VectorField class
        '''
        supported_cls = (Structure,)
        PlastimatchAdaptive.__input_check(input_mask, supported_cls)
        PlastimatchAdaptive.__input_check(vf_file, (VectorField,))
        
        PlastimatchAdaptive.run('convert',
                                input = input_mask.path,
                                output_img = output_mask_path,
                                output_type = 'float')
        
        PlastimatchAdaptive.run('convert',
                                input = output_mask_path,
                                output_img = output_mask_path,
                                xf = vf_file.path)
        
        PlastimatchAdaptive.run('threshold',
                                input = output_mask_path,
                                output = output_mask_path,
                                above = 0.5)
        
        return Structure(output_mask_path)

    @staticmethod
    def get_union(output_mask_path, *masks):
        '''
        Generates union of provided masks.
        Returns Structure object for output mask file.
        
        Args:
            output_mask_path --> path to output mask file (string)
            masks --> Strucutre objects of input masks
        '''
        supported_cls = (Structure,)
        
        if len(masks) < 2:
            raise Exception('Need at least two mask files.')
            
        all(PlastimatchAdaptive.__input_check(mask, supported_cls) for mask in masks)
        
        if os.path.exists(output_mask_path):
            os.remove(output_mask_path)
        
        shutil.copyfile(masks[0].path, output_mask_path)
        
        for mask in masks[1:]:
            
            PlastimatchAdaptive.run('union',
                                    output_mask_path,
                                    mask.path,
                                    output = output_mask_path)
            
        return Structure(output_mask_path)

    @staticmethod
    def get_intersection(output_mask_path, *masks):
        '''
        Generates intersection of provided masks.
        Returns Structure object for output mask file.
        
        Args:
            output_mask_path --> path to output mask file (string)
            masks --> Strucutre objects of input masks
        '''
        supported_cls = (Structure,)
        
        if len(masks) < 2:
            raise Exception('Need at least two mask files.')
            
        all(PlastimatchAdaptive.__input_check(mask, supported_cls) for mask in masks)
        
        shutil.copyfile(masks[0].path, output_mask_path)
        
        for mask in masks[1:]:
            
            PlastimatchAdaptive.run('add',
                                    output_mask_path,
                                    mask.path,
                                    output = output_mask_path)
    
        PlastimatchAdaptive.run('threshold',
                                input = output_mask_path,
                                output = output_mask_path,
                                above = len(masks))
    
        return Structure(output_mask_path)

    @staticmethod
    def exclude_masks(input_mask, output_mask_path, *excluded_masks):
        '''
        Excludes masks from "input_mask" (logical minus).
        Returns Structure object for output mask file.
        
        Args:
            input_mask --> instance of Structure class
            output_mask_path --> path to output mask file (string)
            excluded_masks --> Strucutre objects of masks to exclude from input_mask
        '''
        # TODO: Check if function works well.
        supported_cls = (Structure,)
        PlastimatchAdaptive.__input_check(input_mask, supported_cls)
        
        if len(excluded_masks) < 1:
            raise Exception('Need at least one mask file to exclude.')
            
        all(PlastimatchAdaptive.__input_check(mask, supported_cls) for mask in excluded_masks)
        
        shutil.copyfile(input_mask.path, output_mask_path)
        
        for mask in excluded_masks:

            PlastimatchAdaptive.run('diff',
                                    output_mask_path,
                                    mask.path,
                                    output_mask_path)
        
        PlastimatchAdaptive.run('convert',
                                input = output_mask_path,
                                output_type = 'uchar',
                                output_img = output_mask_path)
        
        return Structure(output_mask_path)
    
    @staticmethod
    def get_empty_mask(reference_image, output_mask_path, values='zeros'):
        '''
        Creates empty mask based on reference image.
        Returns Structure object for output mask file.
        
        Args:
            reference_image --> instance of RTArray class
            output_mask_path --> path to output mask file (string)
            values --> values of output mask, 'zeros' or 'ones' (string)
        '''
        supported_cls = (RTArray,)
        PlastimatchAdaptive.__input_check(reference_image, supported_cls)
    
        stats = PlastimatchAdaptive.get_stats(reference_image)
        
        if values == 'zeros':
            threshold_value = stats['MIN'] - 1
        
        elif values == 'ones':
            threshold_value = stats['MAX'] + 1
        
        PlastimatchAdaptive.run('threshold',
                                input = reference_image.path,
                                output = output_mask_path,
                                below = threshold_value)
    
        return Structure(output_mask_path)
    
    def get_bbox(input_mask, output_mask_path, margin):
        '''
        Generates a bounding box around input mask.
        Returns Structure object for output mask file.
        
        Args:
            input_mask --> instance of Structure class
            output_mask_path --> path to output mask file (string)
            margin --> margin that is applied around the input mask (mm)
        '''
        supported_cls = (Structure,)
        PlastimatchAdaptive.__input_check(input_mask, supported_cls)
        
        PlastimatchAdaptive.run('bbox',
                                input_mask.path,
                                margin = margin,
                                output = output_mask_path)
        
        return Structure(output_mask_path)
    
    def get_shell(input_mask, output_mask_path, distance):
        '''
        Generates a shell around input mask.
        Convention: (+) for outer shell, (-) for inner shell.
        Returns Structure object for output mask file.
        
        Args:
            input_mask --> instance of Structure class
            output_mask_path --> path to output mask file (string)
            distance --> expansion in mm (int or float) or range if passing two values (list or tuple)
        '''
        supported_cls = (Structure,)
        PlastimatchAdaptive.__input_check(input_mask, supported_cls)
        
        dirpath = os.path.dirname(input_mask.path)
        temp = os.path.join(dirpath, 'mask_dmap_temp.mha')
        
        if type(distance) is list or type(distance) is tuple:
            range_str = '{},{}'.format(distance[0], distance[1])
            
        elif type(distance) is int or type(distance) is float:
            edge = 0.001
            range_str = '{},{}'.format(edge, distance) if distance > 0 else '{},{}'.format(distance, edge)
        
        PlastimatchAdaptive.run('dmap',
                                input = input_mask.path,
                                output = temp)
        
        PlastimatchAdaptive.run('threshold',
                                input = temp,
                                output = output_mask_path,
                                range = range_str)
        
        os.remove(temp)
        
        return Structure(output_mask_path)
    
    #%% Image registration methods

    @staticmethod
    def mask_to_img(input_mask, output_file_path, background=0, foreground=1, mode='mask'):
        '''
        Converts binary mask file to synthetic image file.
        Output file can be used for image registration.
        Returns ImageCT object for output file.
        
        input_mask --> instance of Structure class
        output_file_path --> path to output image file (string)
        background --> background value assigned to output image (int or float)
        foreground --> foreground value assigned to output image (int or float)
        mode --> 'mask'
             --> 'shell_outer'
             --> 'shell_inner'
             --> 'shell_uniform'
             --> 'shell_nonuniform'
        '''
        supported_cls = (Structure,)
        PlastimatchAdaptive.__input_check(input_mask, supported_cls)
        
        dirpath = os.path.dirname(input_mask.path)
        temp = os.path.join(dirpath, 'temp.mha')
        
        PlastimatchAdaptive.run('synth',
                                background = background,
                                fixed = input_mask.path,
                                output = temp)
        
        if mode == 'mask':
            PlastimatchAdaptive.run('fill',
                                    input = temp,
                                    mask = input_mask.path,
                                    mask_value = foreground,
                                    output = output_file_path)
        elif mode == 'shell_outer':
            temp_shell = os.path.join(dirpath, 'temp_shell.mha')
            shell = PlastimatchAdaptive.get_shell(input_mask, temp_shell, 10)
            
            PlastimatchAdaptive.run('fill',
                                    input = temp,
                                    mask = shell.path,
                                    mask_value = foreground,
                                    output = output_file_path)
            os.remove(temp_shell)
            
        elif mode == 'shell_inner':
            temp_shell = os.path.join(dirpath, 'temp_shell.mha')
            shell = PlastimatchAdaptive.get_shell(input_mask, temp_shell, -10)
            
            PlastimatchAdaptive.run('fill',
                                    input = temp,
                                    mask = shell.path,
                                    mask_value = foreground,
                                    output = output_file_path)
            os.remove(temp_shell)
            
        elif mode == 'shell_uniform':
            temp_shell_1 = os.path.join(dirpath, 'temp_shell_1.mha')
            temp_shell_2 = os.path.join(dirpath, 'temp_shell_2.mha')
            temp_shell   = os.path.join(dirpath, 'temp_shell.mha')
            shell_1 = PlastimatchAdaptive.get_shell(input_mask, temp_shell_1, 5)
            shell_2 = PlastimatchAdaptive.get_shell(input_mask, temp_shell_2, -5)
            shell   = PlastimatchAdaptive.get_union(temp_shell, shell_1, shell_2)
            
            PlastimatchAdaptive.run('fill',
                                    input = temp,
                                    mask = shell.path,
                                    mask_value = foreground,
                                    output = output_file_path)
            os.remove(temp_shell_1)
            os.remove(temp_shell_2)
            os.remove(temp_shell)

        elif mode == 'shell_nonuniform':
            temp_shell_1 = os.path.join(dirpath, 'temp_shell_1.mha')
            temp_shell_2 = os.path.join(dirpath, 'temp_shell_2.mha')
            temp_shell   = os.path.join(dirpath, 'temp_shell.mha')
            shell_1 = PlastimatchAdaptive.get_shell(input_mask, temp_shell_1, 5)
            shell_2 = PlastimatchAdaptive.get_shell(input_mask, temp_shell_2, -5)
            
            PlastimatchAdaptive.run('fill',
                                    input = temp,
                                    mask = shell_2.path,
                                    mask_value = -foreground,
                                    output = temp)
            
            PlastimatchAdaptive.run('fill',
                                    input = temp,
                                    mask = shell_1.path,
                                    mask_value = foreground,
                                    output = output_file_path)
            os.remove(temp_shell_1)
            os.remove(temp_shell_2)

        os.remove(temp)
        
        return ImageCT(output_file_path)

    @staticmethod
    def register_deformable_bspline(fixed_image,
                                    moving_image,
                                    output_image_path=None,
                                    output_vf_path=None,
                                    fixed_mask=None,
                                    moving_mask=None,
                                    metric='mse',
                                    reg_factor=1):
        '''
        Writes Plastimatch command file and runs deformable image registration.
        Returns appropriate objects for output image and/or output vector field.
        
        Args:
            fixed_image --> instance of Image class
            moving_image --> instance of Image class
            output_image_path --> path to output image file (string)
            output_vf_path --> path to output vector field file (string)
            fixed_mask --> instance of Structure class
            moving_mask --> instance of Structure class
            metric --> cost function metric to optimize (string)
            reg_factor --> regularization multiplier (float or int)
        '''
        supported_cls = (Image,)
        PlastimatchAdaptive.__input_check(fixed_image, supported_cls)
        PlastimatchAdaptive.__input_check(moving_image, supported_cls)
        
        if fixed_mask != None:
            PlastimatchAdaptive.__input_check(fixed_mask, (Structure,))
        if moving_mask != None:
            PlastimatchAdaptive.__input_check(moving_mask, (Structure,))
            
        fixed_mask_path  = fixed_mask.path  if fixed_mask  != None else None
        moving_mask_path = moving_mask.path if moving_mask != None else None

        dirpath = os.path.dirname(moving_image.path)
        command_file_path = os.path.join(dirpath, 'register_bspline_command_file.txt')
    
        with open(command_file_path, 'w') as f:
            
            f.write(PlastimatchAdaptive.image_registration_global(fixed = fixed_image.path,
                                                                  moving = moving_image.path,
                                                                  img_out = output_image_path,
                                                                  vf_out = output_vf_path,
                                                                  fixed_mask = fixed_mask_path,
                                                                  moving_mask = moving_mask_path,
                                                                  default_value = moving_image.base_value))
            
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
        
        if output_image_path != None and output_vf_path != None:
            return moving_image.__class__(output_image_path), BSplineVF(output_vf_path)
            
        elif output_image_path != None and output_vf_path == None:
            return moving_image.__class__(output_image_path)
        
        elif output_image_path == None and output_vf_path != None:
            return BSplineVF(output_vf_path)

    @staticmethod
    def register_3_DOF(fixed_image,
                       moving_image,
                       output_image_path=None,
                       output_vf_path=None,
                       fixed_mask=None,
                       moving_mask=None,
                       metric='mse'):
        '''
        Writes Plastimatch command file and runs 3-DOF image registration.
        Returns appropriate objects for output image and/or output vector field.
        
        Args:
            fixed_image --> instance of Image class
            moving_image --> instance of Image class
            output_image_path --> path to output image file (string)
            output_vf_path --> path to output vector field file (string)
            fixed_mask --> instance of Structure class
            moving_mask --> instance of Structure class
            metric --> cost function metric to optimize (string)
        '''
        supported_cls = (Image,)
        PlastimatchAdaptive.__input_check(fixed_image, supported_cls)
        PlastimatchAdaptive.__input_check(moving_image, supported_cls)
        
        if fixed_mask != None:
            PlastimatchAdaptive.__input_check(fixed_mask, (Structure,))
        if moving_mask != None:
            PlastimatchAdaptive.__input_check(moving_mask, (Structure,))
            
        fixed_mask_path  = fixed_mask.path  if fixed_mask  != None else None
        moving_mask_path = moving_mask.path if moving_mask != None else None

        dirpath = os.path.dirname(moving_image.path)
        command_file_path = os.path.join(dirpath, 'register_3_DOF_command_file.txt')

        with open(command_file_path, 'w') as f:
            
            f.write(PlastimatchAdaptive.image_registration_global(fixed = fixed_image.path,
                                                                  moving = moving_image.path,
                                                                  img_out = output_image_path,
                                                                  vf_out = output_vf_path,
                                                                  fixed_mask = fixed_mask_path,
                                                                  moving_mask = moving_mask_path,
                                                                  default_value = moving_image.base_value))
            
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
        
        if output_image_path != None and output_vf_path != None:
            return moving_image.__class__(output_image_path), TranslationVF(output_vf_path)
            
        elif output_image_path != None and output_vf_path == None:
            return moving_image.__class__(output_image_path)
        
        elif output_image_path == None and output_vf_path != None:
            return TranslationVF(output_vf_path)

    @staticmethod
    def register_6_DOF(fixed_image,
                       moving_image,
                       output_image_path=None,
                       output_vf_path=None,
                       fixed_mask=None,
                       moving_mask=None,
                       metric='mse'):
        '''
        Writes Plastimatch command file and runs 6-DOF image registration.
        Returns appropriate objects for output image and/or output vector field.
        
        Args:
            fixed_image --> instance of Image class
            moving_image --> instance of Image class
            output_image_path --> path to output image file (string)
            output_vf_path --> path to output vector field file (string)
            fixed_mask --> instance of Structure class
            moving_mask --> instance of Structure class
            metric --> cost function metric to optimize (string)
        '''
        supported_cls = (Image,)
        PlastimatchAdaptive.__input_check(fixed_image, supported_cls)
        PlastimatchAdaptive.__input_check(moving_image, supported_cls)
        
        if fixed_mask != None:
            PlastimatchAdaptive.__input_check(fixed_mask, (Structure,))
        if moving_mask != None:
            PlastimatchAdaptive.__input_check(moving_mask, (Structure,))
            
        fixed_mask_path  = fixed_mask.path  if fixed_mask  != None else None
        moving_mask_path = moving_mask.path if moving_mask != None else None

        dirpath = os.path.dirname(moving_image.path)
        command_file_path = os.path.join(dirpath, 'register_6_DOF_command_file.txt')

        with open(command_file_path, 'w') as f:
            
            f.write(PlastimatchAdaptive.image_registration_global(fixed = fixed_image.path,
                                                                  moving = moving_image.path,
                                                                  img_out = output_image_path,
                                                                  vf_out = output_vf_path,
                                                                  fixed_mask = fixed_mask_path,
                                                                  moving_mask = moving_mask_path,
                                                                  default_value = moving_image.base_value))
            
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
        
        if output_image_path != None and output_vf_path != None:
            return moving_image.__class__(output_image_path), RigidVF(output_vf_path)
            
        elif output_image_path != None and output_vf_path == None:
            return moving_image.__class__(output_image_path)
        
        elif output_image_path == None and output_vf_path != None:
            return RigidVF(output_vf_path)
        
    @staticmethod
    def match_position_3_DOF(fixed_image,
                             moving_image,
                             output_image_path,
                             output_vf_path,
                             metric='mse'):
        '''
        Matches patient position with a 3-DOF vector field.
        Retains original image dimensions/size/spacing.
        Returns appropriate objects for output image and vector field.
        
        Args:
            fixed_image --> instance of Image class
            moving_image --> instance of Image class
            output_image_path --> path to output image file (string)
            output_vf_path --> path to output vector field file (string)
            metric --> cost function metric to optimize (string)
        '''
        supported_cls = (Image, Structure)
        PlastimatchAdaptive.__input_check(fixed_image, supported_cls)
        PlastimatchAdaptive.__input_check(moving_image, supported_cls)
        
        dirpath = os.path.dirname(output_vf_path)
        vf_temp_path = os.path.join(dirpath, 'vf_temp.mha')
        
        vf_temp = PlastimatchAdaptive.register_3_DOF(fixed_image,
                                                     moving_image,
                                                     output_vf_path = vf_temp_path,
                                                     metric = metric)
        
        x,y,z = vf_temp.get_shifts()
        os.remove(vf_temp.path)
        
        PlastimatchAdaptive.apply_manual_translation(moving_image,
                                                     output_image_path,
                                                     output_vf_path,
                                                     -x, -y, -z,
                                                     unit = 'mm',
                                                     frame = 'shift',
                                                     discrete_voxels = True)
        
        return moving_image.__class__(output_image_path), TranslationVF(output_vf_path)
    
    @staticmethod
    def apply_vf_to_contours(input_contours,
                             output_contours,
                             vf_file):
        '''
        Applies input vector field to all contours in folder.
        
        Args:
            input_contours --> path to folder containing input structure files (string)
            output_contours --> path to folder to store deformed structures (string)
            vf_file --> instance of VectorField class
        '''
        PlastimatchAdaptive.__input_check(vf_file, (VectorField,))
        
        for filename in os.listdir(input_contours):
            if os.path.isfile(os.path.join(input_contours, filename)):
                PlastimatchAdaptive.warp_mask(Structure(os.path.join(input_contours, filename)), 
                                              os.path.join(output_contours, filename),
                                              vf_file)
    
    @staticmethod
    def propagate_contours(input_contours,
                           output_contours,
                           fixed_image,
                           moving_image,
                           fixed_mask=None,
                           moving_mask=None,
                           reg_factor = 1,
                           translate_first=True):
        '''
        Propagates contours from one image to another.
        
        Args:
            input_contours --> path to folder containing input structure files (string)
            output_contours --> path to folder to store deformed structures (string)
            fixed_image --> instance of Image class
            moving_image --> instance of Image class
            fixed_mask --> instance of Structure class
            moving_mask --> instance of Structure class
            translate_first --> option to match images in 3D before running DIR (bool)
        '''
        supported_cls = (Image, Structure)
        PlastimatchAdaptive.__input_check(fixed_image, supported_cls)
        PlastimatchAdaptive.__input_check(moving_image, supported_cls)
        
        dirpath = os.path.dirname(moving_image.path)
        var = os.path.splitext(moving_image.path)
        
        deformed_img_path = var[0] + '_deformed' + var[1]
        deformation_vf_path = os.path.join(dirpath, 'vf_dir.mha')
        
        if translate_first is True:
            
            translated_img_path = var[0] + '_translated' + var[1]
            translation_vf_path = os.path.join(dirpath, 'vf_3_DOF.mha')
            temp_contours_path = os.path.join(dirpath, 'temp_contours')
            
            img, vf = PlastimatchAdaptive.match_position_3_DOF(fixed_image,
                                                               moving_image,
                                                               translated_img_path,
                                                               translation_vf_path,)
                                                               # metric='mi')
            
            PlastimatchAdaptive.apply_vf_to_contours(input_contours, temp_contours_path, vf)
                
            input_contours = temp_contours_path
            moving_image = img
        
        _, vf = PlastimatchAdaptive.register_deformable_bspline(fixed_image,
                                                                moving_image,
                                                                deformed_img_path,
                                                                deformation_vf_path,
                                                                fixed_mask,
                                                                moving_mask,
                                                                reg_factor = reg_factor)
        
        PlastimatchAdaptive.apply_vf_to_contours(input_contours, output_contours, vf)
            
        if translate_first is True:
            shutil.rmtree(temp_contours_path)
            
        print('\nContour propagation: DONE.')

