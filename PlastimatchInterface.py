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
        
        PlastimatchInterface.run('stats', input_file.path)
        
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
        
        stats  = PlastimatchInterface.run('stats', input_file.path)
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
    
        origin_new = '{} {} {}'.format(origin_x_new, origin_y_new, origin_z_new)
        size_new = '{} {} {}'.format(size_x_new, size_y_new, size_z_new)
        spacing_new = '{} {} {}'.format(spacing_x_new, spacing_y_new, spacing_z_new)
    
        PlastimatchInterface.run('resample',
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
        PlastimatchInterface.__input_check(background, supported_cls)
        PlastimatchInterface.__input_check(foreground, supported_cls)
        
        dirpath = os.path.dirname(foreground)
        temp_mask = os.path.join(dirpath, 'mask_temp.mha')
        temp_foreground = os.path.join(dirpath, 'foreground_temp.mha')
        temp_background = os.path.join(dirpath, 'background_temp.mha')
        
        if len(masks) == 0:
            raise Exception('Need at least one mask file.')
            
        elif len(masks) == 1:
            shutil.copyfile(masks[0].path, temp_mask)
            
        else:
            PlastimatchInterface.get_union(temp_mask, *masks)
        
        PlastimatchInterface.run('mask',
                                 input = foreground.path,
                                 mask = temp_mask,
                                 mask_value = 0,
                                 output = temp_foreground)
        
        PlastimatchInterface.run('fill',
                                 input = background.path,
                                 mask = temp_mask,
                                 mask_value = 0,
                                 output = temp_background)

        PlastimatchInterface.run('add', temp_foreground, temp_background,
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
        
        PlastimatchInterface.run('adjust',
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
            weight --> multiplication weight (float or int)
        '''     
        supported_cls = (PatientImage, DoseMap)
        PlastimatchInterface.__input_check(input_file, supported_cls)

        PlastimatchInterface.run('add',
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
        PlastimatchInterface.__input_check(input_file, supported_cls)
        
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
        
        PlastimatchInterface.extend_or_crop(input_file,
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
        
        PlastimatchInterface.run('synth-vf',
                                 fixed = temp_path,
                                 xf_trans = translation_str,
                                 output = translation_vf.path)
        
        if input_file.__class__ is Structure:            
            PlastimatchInterface.warp_mask(input_file,
                                           output_file_path,
                                           translation_vf)
        else:
            PlastimatchInterface.warp_image(input_file,
                                            output_file_path,
                                            translation_vf)
        
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
              .format(-x, -x/input_file.spacing_x,
                      -y, -y/input_file.spacing_y,
                      -z, -z/input_file.spacing_z))
            
        return input_file.__class__(output_file_path), translation_vf

    @staticmethod
    def get_translation_vf_shifts(vf_file):
        '''
        Returns shifts of a 3-DOF vector field in mm.
        
        Args:
            vf_file --> instance of TranslationVF class
        '''
        supported_cls = (TranslationVF,)
        PlastimatchInterface.__input_check(vf_file, supported_cls)
        
        vf_file.load_file()
        
        # Assuming each voxel receives the same shift:
        voxel = vf_file.ndarray[0][0][0]
        
        shift_x = voxel[0]
        shift_y = voxel[1]
        shift_z = voxel[2]
                
        return shift_x, shift_y, shift_z

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
        supported_cls = (PatientImage, DoseMap, VectorField)
        PlastimatchInterface.__input_check(input_file, supported_cls)
        PlastimatchInterface.__input_check(vf_file, (VectorField,))
        
        default_value = input_file.base_value if default_value == None else default_value
        
        PlastimatchInterface.run('convert',
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
            mask_value --> numeric value that is applied to the background (float)
                       --> will be applied according to class if not specified
        '''
        supported_cls = (PatientImage, DoseMap, VectorField)
        PlastimatchInterface.__input_check(input_file, supported_cls)
        PlastimatchInterface.__input_check(mask, (Structure,))
        
        mask_value = input_file.base_value if mask_value == None else mask_value
        
        PlastimatchInterface.run('mask',
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
        
        PlastimatchInterface.run('convert',
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
        
        PlastimatchInterface.run('convert',
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
        PlastimatchInterface.__input_check(input_file, supported_cls)
        PlastimatchInterface.__input_check(reference_image, supported_cls)
        
        PlastimatchInterface.run('resample',
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
        PlastimatchInterface.__input_check(input_mask, supported_cls)
        
        dirpath = os.path.dirname(input_mask)
        temp = os.path.join(dirpath, 'mask_dmap_temp.mha')
        
        PlastimatchInterface.run('dmap',
                                 input = input_mask.path,
                                 # algorithm = 'maurer',
                                 output = temp)
        
        PlastimatchInterface.run('threshold',
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
        PlastimatchInterface.__input_check(input_mask, supported_cls)
        
        PlastimatchInterface.run('threshold',
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
        PlastimatchInterface.__input_check(input_mask, supported_cls)
        PlastimatchInterface.__input_check(vf_file, (VectorField,))
        
        PlastimatchInterface.run('convert',
                                 input = input_mask.path,
                                 output_img = output_mask_path,
                                 output_type = 'float')
        
        PlastimatchInterface.run('convert',
                                 input = output_mask_path,
                                 output_img = output_mask_path,
                                 xf = vf_file.path)
        
        PlastimatchInterface.run('threshold',
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
            
        all(PlastimatchInterface.__input_check(mask, supported_cls) for mask in masks)
        
        shutil.copyfile(masks[0].path, output_mask_path)
        
        for mask in masks[1:]:
            
            PlastimatchInterface.run('union',
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
            
        all(PlastimatchInterface.__input_check(mask, supported_cls) for mask in masks)
        
        shutil.copyfile(masks[0].path, output_mask_path)
        
        for mask in masks[1:]:
            
            PlastimatchInterface.run('add',
                                     output_mask_path,
                                     mask.path,
                                     output = output_mask_path)
    
        PlastimatchInterface.run('threshold',
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
        PlastimatchInterface.__input_check(input_mask, supported_cls)
        
        if len(excluded_masks) < 1:
            raise Exception('Need at least one mask file to exclude.')
            
        all(PlastimatchInterface.__input_check(mask, supported_cls) for mask in excluded_masks)
        
        shutil.copyfile(input_mask.path, output_mask_path)
        
        for mask in excluded_masks:

            PlastimatchInterface.run('diff',
                                     output_mask_path,
                                     mask.path,
                                     output_mask_path)
        
        PlastimatchInterface.run('convert',
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
        PlastimatchInterface.__input_check(reference_image, supported_cls)
    
        stats = PlastimatchInterface.get_stats(reference_image)
        
        if values == 'zeros':
            threshold_value = stats['MIN'] - 1
        
        elif values == 'ones':
            threshold_value = stats['MAX'] + 1
        
        PlastimatchInterface.run('threshold',
                                 input = reference_image.path,
                                 output = output_mask_path,
                                 below = threshold_value)
    
        return Structure(output_mask_path)
    
    #%% Image registration methods

    @staticmethod
    def register_deformable_bspline(fixed_image,
                                    moving_image,
                                    output_image_path=None,
                                    output_vf_path=None,
                                    fixed_mask=None,
                                    moving_mask=None,
                                    metric='mse'):
        '''
        Writes Plastimatch command file and runs deformable image registration.
        Returns appropriate objects for output image and/or output vector field.
        
        Args:
            fixed_image --> instance of PatientImage class
            moving_image --> instance of PatientImage class
            output_image_path --> path to output image file (string)
            output_vf_path --> path to output vector field file (string)
            fixed_mask --> instance of Structure class
            moving_mask --> instance of Structure class
            metric --> cost function metric to optimize (string)
        '''
        PlastimatchInterface.__input_check(fixed_image, (PatientImage,))
        PlastimatchInterface.__input_check(moving_image, (PatientImage,))
        
        if fixed_mask != None:
            PlastimatchInterface.__input_check(fixed_mask, (Structure,))
        if moving_mask != None:
            PlastimatchInterface.__input_check(moving_mask, (Structure,))
            
        fixed_mask_path  = fixed_mask.path  if fixed_mask  != None else None
        moving_mask_path = moving_mask.path if moving_mask != None else None

        dirpath = os.path.dirname(moving_image.path)
        command_file_path = os.path.join(dirpath, 'register_bspline_command_file.txt')
    
        with open(command_file_path, 'w') as f:
            
            f.write(PlastimatchInterface.image_registration_global(fixed = fixed_image.path,
                                                                   moving = moving_image.path,
                                                                   img_out = output_image_path,
                                                                   vf_out = output_vf_path,
                                                                   fixed_mask = fixed_mask_path,
                                                                   moving_mask = moving_mask_path,
                                                                   default_value = moving_image.base_value))
            
            f.write(PlastimatchInterface.image_registration_stage(xform = 'bspline',
                                                                  optim = 'lbfgsb',
                                                                  impl = 'plastimatch',
                                                                  threading = 'cuda',
                                                                  max_its = 50,
                                                                  grid_spac = '100 100 100',
                                                                  res = '8 8 4',
                                                                  regularization_lambda = 1,
                                                                  metric = metric))
                                                                  
            f.write(PlastimatchInterface.image_registration_stage(xform = 'bspline',
                                                                  optim = 'lbfgsb',
                                                                  impl = 'plastimatch',
                                                                  threading = 'cuda',
                                                                  max_its = 50,
                                                                  grid_spac = '80 80 80',
                                                                  res = '4 4 2',
                                                                  regularization_lambda = 0.1,
                                                                  metric = metric))
                    
            f.write(PlastimatchInterface.image_registration_stage(xform = 'bspline',
                                                                  optim = 'lbfgsb',
                                                                  impl = 'plastimatch',
                                                                  threading = 'cuda',
                                                                  max_its = 40,
                                                                  grid_spac = '60 60 60',
                                                                  res = '2 2 1',
                                                                  regularization_lambda = 0.1,
                                                                  metric = metric))
            
            f.write(PlastimatchInterface.image_registration_stage(xform = 'bspline',
                                                                  optim = 'lbfgsb',
                                                                  impl = 'plastimatch',
                                                                  threading = 'cuda',
                                                                  max_its = 40,
                                                                  grid_spac = '20 20 20',
                                                                  res = '1 1 1',
                                                                  regularization_lambda = 0.01,
                                                                  metric = metric))
            
            f.write(PlastimatchInterface.image_registration_stage(xform = 'bspline',
                                                                  optim = 'lbfgsb',
                                                                  impl = 'plastimatch',
                                                                  threading = 'cuda',
                                                                  max_its = 40,
                                                                  grid_spac = '10 10 10',
                                                                  res = '1 1 1',
                                                                  regularization_lambda = 0.01,
                                                                  metric = metric))
        
        PlastimatchInterface.run(command_file_path)
        
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
            fixed_image --> instance of PatientImage class
            moving_image --> instance of PatientImage class
            output_image_path --> path to output image file (string)
            output_vf_path --> path to output vector field file (string)
            fixed_mask --> instance of Structure class
            moving_mask --> instance of Structure class
            metric --> cost function metric to optimize (string)
        '''
        PlastimatchInterface.__input_check(fixed_image, (PatientImage,))
        PlastimatchInterface.__input_check(moving_image, (PatientImage,))
        
        if fixed_mask != None:
            PlastimatchInterface.__input_check(fixed_mask, (Structure,))
        if moving_mask != None:
            PlastimatchInterface.__input_check(moving_mask, (Structure,))
            
        fixed_mask_path  = fixed_mask.path  if fixed_mask  != None else None
        moving_mask_path = moving_mask.path if moving_mask != None else None

        dirpath = os.path.dirname(moving_image.path)
        command_file_path = os.path.join(dirpath, 'register_3_DOF_command_file.txt')
    
        with open(command_file_path, 'w') as f:
            
            f.write(PlastimatchInterface.image_registration_global(fixed = fixed_image.path,
                                                                   moving = moving_image.path,
                                                                   img_out = output_image_path,
                                                                   vf_out = output_vf_path,
                                                                   fixed_mask = fixed_mask_path,
                                                                   moving_mask = moving_mask_path,
                                                                   default_value = moving_image.base_value))
            
            f.write(PlastimatchInterface.image_registration_stage(xform = 'translation',
                                                                  optim = 'rsg',
                                                                  threading = 'cuda',
                                                                  max_its = 200,
                                                                  res = '8 8 4',
                                                                  metric = metric))
                                                                  
            f.write(PlastimatchInterface.image_registration_stage(xform = 'translation',
                                                                  optim = 'rsg',
                                                                  threading = 'cuda',
                                                                  max_its = 200,
                                                                  res = '4 4 2',
                                                                  metric = metric))
                    
            f.write(PlastimatchInterface.image_registration_stage(xform = 'translation',
                                                                  optim = 'rsg',
                                                                  threading = 'cuda',
                                                                  max_its = 200,
                                                                  res = '2 2 1',
                                                                  metric = metric))
        
        PlastimatchInterface.run(command_file_path)
        
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
            fixed_image --> instance of PatientImage class
            moving_image --> instance of PatientImage class
            output_image_path --> path to output image file (string)
            output_vf_path --> path to output vector field file (string)
            fixed_mask --> instance of Structure class
            moving_mask --> instance of Structure class
            metric --> cost function metric to optimize (string)
        '''
        PlastimatchInterface.__input_check(fixed_image, (PatientImage,))
        PlastimatchInterface.__input_check(moving_image, (PatientImage,))
        
        if fixed_mask != None:
            PlastimatchInterface.__input_check(fixed_mask, (Structure,))
        if moving_mask != None:
            PlastimatchInterface.__input_check(moving_mask, (Structure,))
            
        fixed_mask_path  = fixed_mask.path  if fixed_mask  != None else None
        moving_mask_path = moving_mask.path if moving_mask != None else None

        dirpath = os.path.dirname(moving_image.path)
        command_file_path = os.path.join(dirpath, 'register_6_DOF_command_file.txt')
    
        with open(command_file_path, 'w') as f:
            
            f.write(PlastimatchInterface.image_registration_global(fixed = fixed_image.path,
                                                                   moving = moving_image.path,
                                                                   img_out = output_image_path,
                                                                   vf_out = output_vf_path,
                                                                   fixed_mask = fixed_mask_path,
                                                                   moving_mask = moving_mask_path,
                                                                   default_value = moving_image.base_value))
            
            f.write(PlastimatchInterface.image_registration_stage(xform = 'rigid',
                                                                  optim = 'versor',
                                                                  threading = 'cuda',
                                                                  max_its = 200,
                                                                  res = '8 8 4',
                                                                  metric = metric))
                                                                  
            f.write(PlastimatchInterface.image_registration_stage(xform = 'rigid',
                                                                  optim = 'versor',
                                                                  threading = 'cuda',
                                                                  max_its = 200,
                                                                  res = '4 4 2',
                                                                  metric = metric))
                    
            f.write(PlastimatchInterface.image_registration_stage(xform = 'rigid',
                                                                  optim = 'versor',
                                                                  threading = 'cuda',
                                                                  max_its = 200,
                                                                  res = '2 2 1',
                                                                  metric = metric))
        
        PlastimatchInterface.run(command_file_path)
        
        if output_image_path != None and output_vf_path != None:
            return moving_image.__class__(output_image_path), RigidVF(output_vf_path)
            
        elif output_image_path != None and output_vf_path == None:
            return moving_image.__class__(output_image_path)
        
        elif output_image_path == None and output_vf_path != None:
            return RigidVF(output_vf_path)
        
    @staticmethod
    def match_position_3_DOF(fixed_image, moving_image, output_image_path, output_vf_path, metric='mse'):
        '''
        Matches patient position with a 3-DOF vector field.
        Retains original image dimensions/size/spacing.
        Returns appropriate objects for output image and vector field.
        
        Args:
            fixed_image --> instance of PatientImage class
            moving_image --> instance of PatientImage class
            output_image_path --> path to output image file (string)
            output_vf_path --> path to output vector field file (string)
            metric --> cost function metric to optimize (string)
        '''
        PlastimatchInterface.__input_check(fixed_image, (PatientImage,))
        PlastimatchInterface.__input_check(moving_image, (PatientImage,))
        
        dirpath = os.path.dirname(output_vf_path)
        vf_temp_path = os.path.join(dirpath, 'vt_temp.mha')
        
        vf_temp = PlastimatchInterface.register_3_DOF(fixed_image,
                                                      moving_image,
                                                      output_vf_path = vf_temp_path,
                                                      metric = metric)
        
        x,y,z = PlastimatchInterface.get_translation_vf_shifts(vf_temp)
        os.remove(vf_temp.path)
        
        PlastimatchInterface.apply_manual_translation(moving_image,
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
        PlastimatchInterface.__input_check(vf_file, (VectorField,))
        
        for filename in os.listdir(input_contours):
            if os.path.isfile(os.path.join(input_contours, filename)):
                PlastimatchInterface.warp_mask(Structure(os.path.join(input_contours, filename)), 
                                               os.path.join(output_contours, filename),
                                               vf_file)
    
    @staticmethod
    def propagate_contours(input_contours,
                           output_contours,
                           fixed_image,
                           moving_image,
                           moving_mask=None,
                           apply_fixed_box=False,
                           match_images=True):
        '''
        Propagates contours from one image to another.
        
        Args:
            input_contours --> path to folder containing input structure files (string)
            output_contours --> path to folder to store deformed structures (string)
            fixed_image --> instance of PatientImage class
            moving_image --> instance of PatientImage class
            moving_mask --> instance of Structure class for moving mask
            apply_fixed_box --> option to use box as fixed mask (bool)
            match_images --> option to match images in 3D before running DIR (bool)
        '''
        PlastimatchInterface.__input_check(fixed_image, (PatientImage,))
        PlastimatchInterface.__input_check(moving_image, (PatientImage,))
        
        dirpath = os.path.dirname(moving_image.path)
        var = os.path.splitext(moving_image.path)
        
        deformed_img_path = var[0] + '_deformed' + var[1]
        deformation_vf_path = os.path.join(dirpath, 'vf_dir.mha')
        
        if match_images is True:
            
            translated_img_path = var[0] + '_translated' + var[1]
            translation_vf_path = os.path.join(dirpath, 'vf_3_DOF.mha')
            temp_contours_path = os.path.join(dirpath, 'temp_contours')
            
            img, vf = PlastimatchInterface.match_position_3_DOF(fixed_image,
                                                                moving_image,
                                                                translated_img_path,
                                                                translation_vf_path)
            
            PlastimatchInterface.apply_vf_to_contours(input_contours, temp_contours_path, vf)
                
            input_contours = temp_contours_path
            moving_image = img
        
        if apply_fixed_box is True:
            box_temp_path = os.path.join(output_contours, 'box_temp.mha')
            box_temp = PlastimatchInterface.get_empty_mask(moving_image, box_temp_path, values='ones')
            fixed_mask = PlastimatchInterface.resample_to_reference(box_temp, box_temp.path, fixed_image)
        else:
            fixed_mask = None
        
        _, vf = PlastimatchInterface.register_deformable_bspline(fixed_image,
                                                                 moving_image,
                                                                 deformed_img_path,
                                                                 deformation_vf_path,
                                                                 fixed_mask,
                                                                 moving_mask)
        
        PlastimatchInterface.apply_vf_to_contours(input_contours, output_contours, vf)
        
        if apply_fixed_box is True:
            os.remove(fixed_mask.path)
            
        if match_images is True:
            shutil.rmtree(temp_contours_path)
            
        print('\nContour propagation: DONE.')
        
