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
            weight --> multiplication weight (float or int)
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
    def get_translation_vf_shifts(vf_file):
        '''
        Returns shifts of a 3-DOF vector field as a dictionary.
        
        Args:
            vf_file --> instance of TranslationVF class
        '''
        supported_cls = (TranslationVF,)
        PlastimatchInterface.__input_check(vf_file, supported_cls)
        
        vf_file.load_file()
        
        # Assuming each voxel receives the same shift:
        voxel = vf_file.ndarray[0][0][0]
        
        shifts = {'x_shift' : voxel[0],
                  'y_shift' : voxel[1],
                  'z_shift' : voxel[2]}
                
        return shifts

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
        
        mask = subprocess.Popen(['plastimatch','mask',
                                 '--input', input_file.path,
                                 '--output', output_file_path,
                                 '--mask', mask.path,
                                 '--mask-value', str(mask_value)
                                 ])
        mask.wait()
        
        return input_file.__class__(output_file_path)

    @staticmethod
    def convert_dicom_folder(input_dicom_folder, output_image, structures=None, dose_map=None):
        '''
        Function extracts images from dicom folder.
        
        Args:
            input_dicom_folder --> path to input dicom folder (string)
            output_image --> path to output image file (string)
            structures --> path to folder for structures (string)
            dose_map --> path to output dose image file (string)
        '''
        # TODO: modify method once we have appropriate classes like e.g. Patient, StructureSet, DicomFolder etc.
        get_image = subprocess.Popen(['plastimatch','convert',
                                      '--input', input_dicom_folder,
                                      '--output-type', 'float',
                                      '--output-img', output_image,
                                      ])
        get_image.wait()
            
        if structures != None:
            get_structures = subprocess.Popen(['plastimatch','convert',
                                               '--input', input_dicom_folder,
                                               '--fixed', output_image,
                                               '--output-type', 'float',
                                               '--output-prefix', structures,
                                               ])
            get_structures.wait()
            
        if dose_map != None:
            get_dose_map = subprocess.Popen(['plastimatch','convert',
                                             '--input', input_dicom_folder,
                                             '--fixed', output_image,
                                             '--output-type', 'float',
                                             '--output-dose-img', dose_map,
                                             ])
            get_dose_map.wait()

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
        
        get_dmap = subprocess.Popen(['plastimatch','dmap',
                                     '--input', input_mask.path,
                                     # '--algorithm', 'maurer',
                                     '--output', 'mask_dmap_temp.mha',
                                     ], cwd = dirpath)
        get_dmap.wait()
                                 
        threshold_mask = subprocess.Popen(['plastimatch','threshold',
                                           '--input', 'mask_dmap_temp.mha',
                                           '--output', output_mask_path,
                                           '--below', '{}'.format(distance)
                                           ], cwd = dirpath)  
        threshold_mask.wait()
        
        os.remove(os.path.join(dirpath, 'mask_dmap_temp.mha'))
        
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
        
        threshold = subprocess.Popen(['plastimatch','threshold',
                                      '--input', input_mask.path,
                                      '--output', output_mask_path,
                                      '--below', '0.5'
                                      ])
        threshold.wait()
        
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
    
        convert_mask = subprocess.Popen(['plastimatch','convert',
                                         '--input', input_mask.path,
                                         '--output-img', output_mask_path,
                                         '--output-type', 'float'
                                         ])
        convert_mask.wait()
        
        warp_mask = subprocess.Popen(['plastimatch','convert',
                                      '--input', output_mask_path,
                                      '--output-img', output_mask_path,
                                      '--xf', vf_file.path
                                      ])
        warp_mask.wait()
        
        threshold = subprocess.Popen(['plastimatch','threshold',
                                      '--input', output_mask_path,
                                      '--output', output_mask_path,
                                      '--above', '0.5'
                                      ])
        threshold.wait()
        
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
        
            union = subprocess.Popen(['plastimatch', 'union',
                                      output_mask_path,
                                      mask.path,
                                      '--output', output_mask_path,
                                      ])
            union.wait()
            
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
        
            add_masks = subprocess.Popen(['plastimatch', 'add',
                                          output_mask_path,
                                          mask.path,
                                          '--output', output_mask_path,
                                          ])
            add_masks.wait()
    
        threshold = subprocess.Popen(['plastimatch','threshold',
                                      '--input', output_mask_path,
                                      '--output', output_mask_path,
                                      '--above', str(len(masks))
                                      ])
        threshold.wait()
        
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
        
            subtract = subprocess.Popen(['plastimatch','diff',
                                         '{}'.format(output_mask_path),
                                         '{}'.format(mask.path),
                                         output_mask_path
                                         ])  
            subtract.wait()
        
        convert = subprocess.Popen(['plastimatch','convert',
                                    '--input', output_mask_path,
                                    '--output-type', 'uchar',
                                    '--output-img', output_mask_path,
                                    ])
        convert.wait()
        
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
    
        threshold = subprocess.Popen(['plastimatch','threshold',
                                      '--input', reference_image.path,
                                      '--output', output_mask_path,
                                      '--below', str(threshold_value),
                                      ])
        threshold.wait()
    
        return Structure(output_mask_path)
    
        #%% Image registration methods
        
        



