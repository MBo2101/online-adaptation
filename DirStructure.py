# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 10:28:25 2021

@author: MBo
"""

import os
import shutil
import pandas as pd
from time import time
from DataPlotter import DVHPlotter
from DicomPlan import DicomPlan
from EvaluationManager import EvaluationManager
from MoquiManager import MoquiManager
from PlanAdaptation import PlanAdaptation
from PlastimatchAdaptive import PlastimatchAdaptive

class DirStructure(object):
    
    def __init__(self, patient_dir):
        self.patient_dir = patient_dir
        self.fractions_parent_dir = self.patient_dir
        self.plans_dir = os.path.join(self.patient_dir, 'plans')
        self.set_internal_names()
    
    def get_fraction_dirs(self, fraction_name):
        '''
        Sets up paths.
        '''
        fraction_dir   = os.path.join(self.fractions_parent_dir, fraction_name)
        image_file     = os.path.join(fraction_dir, fraction_name+'.mha')
        contours_dir   = os.path.join(fraction_dir, 'contours')
        transforms_dir = os.path.join(fraction_dir, 'transforms')
        dose_maps_dir  = os.path.join(fraction_dir, 'dose_maps')
        dvhs_dir       = os.path.join(fraction_dir, 'DVH')
        dct = dict(zip(['fraction_name', 'fraction_dir', 'image_file', 'contours_dir', 'transforms_dir', 'dose_maps_dir', 'dvhs_dir'],
                       [ fraction_name,   fraction_dir,   image_file,   contours_dir,   transforms_dir,   dose_maps_dir,   dvhs_dir]))
        return dct

    def set_internal_names(self):
        self.__cbcts_info_filename  = 'cbcts_info.csv'
        self.__goal_dose_filename   = 'goal_dose.mha'
        self.__target_filename      = 'target.mha'
        self.__dij_mask_filename    = 'dij_mask.mha'
        self.__shell_filename       = 'conformality_shell.mha'
        self.__FOV_full_filename    = 'FOV_full.mha'
        self.__FOV_small_filename   = 'FOV_small.mha'
        self.__vf_3_DOF_filename    = 'vf_1_cbct_to_{}_3_DOF_cbct_grid.mha'
        self.__vf_6_DOF_filename    = 'vf_2_cbct_to_{}_6_DOF_cbct_grid.mha'
        self.__vf_DIR_filename      = 'vf_3_{}_to_cbct_DIR.mha'
        self.__ct_deformed_filename = 'ct_deformed.mha'
        
    def import_cbcts(self, input_cbcts_dir, selected_fractions=None):
        cbcts_list = os.listdir(input_cbcts_dir)
        cbcts_list.sort()
        fraction_indices = [i for i in range(1, len(cbcts_list)+1)]
        cbct_names = ['cbct_' + str(i) for i in fraction_indices]
        fraction_dirs       = [self.get_fraction_dirs(name)['fraction_dir']   for name in cbct_names]
        cbct_image_files    = [self.get_fraction_dirs(name)['image_file']     for name in cbct_names]
        cbct_contour_dirs   = [self.get_fraction_dirs(name)['contours_dir']   for name in cbct_names]
        cbct_transform_dirs = [self.get_fraction_dirs(name)['transforms_dir'] for name in cbct_names]
        cbct_dose_map_dirs  = [self.get_fraction_dirs(name)['dose_maps_dir']  for name in cbct_names]
        pd_data = list(zip(fraction_indices,
                           cbct_names,
                           cbcts_list,
                           fraction_dirs,
                           cbct_image_files,
                           cbct_contour_dirs,
                           cbct_transform_dirs,
                           cbct_dose_map_dirs))
        pd_names = ['fraction_indices',
                    'names',
                    'names_original',
                    'fraction_dirs',
                    'image_files',
                    'contour_dirs',
                    'transform_dirs',
                    'dose_map_dirs']
        self.cbcts = pd.DataFrame(pd_data, columns=pd_names)
        if selected_fractions is not None:
            self.cbcts = self.cbcts.drop([i-1 for i in fraction_indices if i not in selected_fractions])
        if not os.path.exists(self.fractions_parent_dir) : os.makedirs(self.fractions_parent_dir)
        self.cbcts.to_csv(os.path.join(self.fractions_parent_dir, self.__cbcts_info_filename))
        # Copy image files accordingly
        for cbct_src in cbcts_list:
            index = cbcts_list.index(cbct_src)
            if selected_fractions is None or index+1 in selected_fractions:
                file_src = os.path.join(input_cbcts_dir, cbct_src)
                name = cbct_names[index]
                dir_out = fraction_dirs[index]
                file_out = cbct_image_files[index]
                contours_dir = cbct_contour_dirs[index]
                transforms_dir = cbct_transform_dirs[index]
                dose_maps_dir = cbct_dose_map_dirs[index]
                for i in [dir_out, contours_dir, transforms_dir, dose_maps_dir]:
                    if not os.path.exists(i) : os.makedirs(i)
                shutil.copyfile(file_src, file_out)
                print('Copied fraction image: {}'.format(name))
    
    def import_plan(self, input_dicom_dir, plan_name=None, ct_name='ct'):
        if plan_name is None : plan_name = os.path.basename(input_dicom_dir.rstrip('/'))
        self.set_reference_ct(ct_name)
        plan_dose = os.path.join(self.ct_dose_maps_dir, ct_name+'_'+plan_name+'_TPS.mha')
        plan_dir = os.path.join(self.plans_dir, plan_name)
        if os.path.exists(plan_dir) : shutil.rmtree(plan_dir)
        shutil.copytree(input_dicom_dir, plan_dir)
        PlastimatchAdaptive.DICOM_to_ITK(input_dicom_dir,
                                         output_image = self.ct_image_file,
                                         structures = self.ct_contours_dir,
                                         dose_map = plan_dose)
        # By default sets the nominal plan dose as the goal dose
        self.set_goal_dose(plan_dose)

    def set_reference_ct(self, ct_name):
        self.ct_name = ct_name
        dct = self.get_fraction_dirs(ct_name)
        self.ct_dir            = dct['fraction_dir']
        self.ct_image_file     = dct['image_file']
        self.ct_contours_dir   = dct['contours_dir']
        self.ct_transforms_dir = dct['transforms_dir']
        self.ct_dose_maps_dir  = dct['dose_maps_dir']
        for i in [self.ct_dir,
                  self.ct_contours_dir,
                  self.ct_transforms_dir,
                  self.ct_dose_maps_dir]:
            if not os.path.exists(i) : os.makedirs(i)

    def set_goal_dose(self, dose_map_file):
        self.goal_dose = os.path.join(self.ct_dose_maps_dir, self.__goal_dose_filename)
        PlastimatchAdaptive.resample_to_reference(dose_map_file,
                                                  self.goal_dose,
                                                  self.ct_image_file,
                                                  default_value = 0)

    def set_structure_names(self, targets, oars, external, artifacts):
        dij_masks = targets + oars
        if None in dij_masks : dij_masks.remove(None)
        self.targets = targets
        self.oars = oars
        self.external = external
        self.artifacts = artifacts
        self.dij_masks = dij_masks
    
    def get_structure_files(self, fraction_name):
        dct = self.get_fraction_dirs(fraction_name)
        contours_dir = dct['contours_dir']
        lst = [os.path.splitext(i)[0] for i in os.listdir(contours_dir) if '.mha' in i]
        return dict(zip(lst, [os.path.join(contours_dir, i+'.mha') for i in lst]))

    def generate_masks(self, fraction_name, shell_distance=10):
        '''
        Generates helper mask files.
        '''
        dct = self.get_fraction_dirs(fraction_name)
        contours_dir = dct['contours_dir']
        output_target = os.path.join(contours_dir, self.__target_filename)
        output_dij    = os.path.join(contours_dir, self.__dij_mask_filename)
        output_shell  = os.path.join(contours_dir, self.__shell_filename)
        output_temp   = os.path.join(contours_dir, 'temp.mha')
        target_masks = [os.path.join(contours_dir, name+'.mha') for name in self.targets]
        dij_masks    = [os.path.join(contours_dir, name+'.mha') for name in self.dij_masks]
        # 1: target
        if len(target_masks) > 1:
            PlastimatchAdaptive.get_union(output_target, *target_masks)
        elif len(target_masks) == 1:
            shutil.copyfile(target_masks[0], output_target)
        # 2: dij_mask
        if len(dij_masks) > 1:
            PlastimatchAdaptive.get_union(output_temp, *dij_masks)
        elif len(dij_masks) == 1:
            shutil.copyfile(dij_masks[0], output_temp)
        if shell_distance > 0:
            PlastimatchAdaptive.expand_mask(output_temp, output_temp, shell_distance)
        if self.external is not None:
            external_mask = os.path.join(contours_dir, self.external+'.mha')
            PlastimatchAdaptive.get_intersection(output_dij, external_mask, output_temp)
        elif self.external is None:
            shutil.copyfile(output_temp, output_dij)
        os.remove(output_temp)
        # 3: conformality_shell
        output_shell = os.path.join(contours_dir, self.__shell_filename)
        PlastimatchAdaptive.exclude_masks(output_dij, output_shell, *target_masks)

    def modify_image_HUs(self, fraction_name):
        '''
        Modifies HUs in image file based on the external and artifacts mask.
        '''
        dct = self.get_fraction_dirs(fraction_name)
        image_file = dct['image_file']
        contours_dir = dct['contours_dir']
        # Raise values below -1000 HUs to -1000 (Moqui considers -1000 as vacuum)
        PlastimatchAdaptive.fill_image_threshold(image_file,
                                                 image_file,
                                                 threshold = -1000,
                                                 option = 'below',
                                                 mask_value= -1000)
        if self.external is not None:
            external_mask = os.path.join(contours_dir, self.external+'.mha')
            PlastimatchAdaptive.mask_image(image_file, image_file, external_mask, mask_value=-1001)
        if self.artifacts is not None:
            artifacts_mask = os.path.join(contours_dir, self.artifacts+'.mha')
            PlastimatchAdaptive.fill_image(image_file, image_file, artifacts_mask, mask_value=26)

    def match_cbcts_isocenter(self, reference_ct_name='ct'):
        '''
        Matches CBCTs to reference CT images.
        Generates CBCT FOV mask to be used for DIR.
        CBCTs and masks are resampled to match the CT grid.
        '''
        self.set_reference_ct(reference_ct_name)
        for fraction_name in self.cbcts.names:
            dct = self.get_fraction_dirs(fraction_name)
            fraction_dir = dct['fraction_dir']
            cbct_image_file = dct['image_file']
            cbct_contours_dir = dct['contours_dir']
            cbct_transforms_dir = dct['transforms_dir']
            FOV_full  = os.path.join(cbct_contours_dir, self.__FOV_full_filename)
            FOV_small = os.path.join(cbct_contours_dir, self.__FOV_small_filename)
            vf_3_DOF = os.path.join(cbct_transforms_dir, self.__vf_3_DOF_filename.format(self.ct_name))
            vf_6_DOF = os.path.join(cbct_transforms_dir, self.__vf_6_DOF_filename.format(self.ct_name))
            # Generate CBCT's FOV mask
            PlastimatchAdaptive.run('synth',
                                    fixed = cbct_image_file,
                                    pattern = 'cylinder',
                                    radius = '130',
                                    background = '0',
                                    foreground = '1',
                                    output = FOV_full,
                                    output_type = 'uchar')
            PlastimatchAdaptive.run('synth',
                                    fixed = cbct_image_file,
                                    pattern = 'cylinder',
                                    radius = '120',
                                    background = '0',
                                    foreground = '1',
                                    output = FOV_small,
                                    output_type = 'uchar')
            # 3-DOF matching; retains original image dimensions/size/spacing
            PlastimatchAdaptive.match_position_3_DOF(fixed_image = self.ct_image_file,
                                                     moving_image = cbct_image_file,
                                                     output_image = cbct_image_file,
                                                     output_vf = vf_3_DOF)
            # 6-DOF matching; need to resample VF to CBCT after registration
            vf_rigid_ct_grid = os.path.join(fraction_dir, 'vf_temp.mha')
            target_mask_file = os.path.join(self.ct_contours_dir, self.__target_filename)
            registration_mask = os.path.join(cbct_contours_dir, 'mask_temp.mha')
            PlastimatchAdaptive.get_bbox(target_mask_file, registration_mask, 20)
            PlastimatchAdaptive.register_6_DOF(fixed_image = self.ct_image_file,
                                               moving_image = cbct_image_file,
                                               output_image = None,
                                               output_vf = vf_rigid_ct_grid,
                                               fixed_mask = registration_mask,
                                               moving_mask = None)
            PlastimatchAdaptive.resample_to_reference(vf_rigid_ct_grid, vf_6_DOF, cbct_image_file)
            PlastimatchAdaptive.warp_image(cbct_image_file,
                                           cbct_image_file,
                                           vf_6_DOF)
            os.remove(registration_mask)
            os.remove(vf_rigid_ct_grid)
            # Match CBCT's FOV mask
            PlastimatchAdaptive.warp_mask(FOV_full, FOV_full, vf_3_DOF)
            PlastimatchAdaptive.warp_mask(FOV_full, FOV_full, vf_6_DOF)
            PlastimatchAdaptive.warp_mask(FOV_small, FOV_small, vf_3_DOF)
            PlastimatchAdaptive.warp_mask(FOV_small, FOV_small, vf_6_DOF)
            # Resample CBCT and FOV mask to CT dimensions
            PlastimatchAdaptive.resample_to_reference(cbct_image_file, cbct_image_file, self.ct_image_file, -1000)
            PlastimatchAdaptive.resample_to_reference(FOV_full, FOV_full, self.ct_image_file, 0)
            PlastimatchAdaptive.resample_to_reference(FOV_small, FOV_small, self.ct_image_file, 0)

    def run_cbcts_DIR(self, reference_ct_name='ct'):
        '''
        Runs DIR between CT and CBCT.
        Output VF is used for contour propagation and dose deformation.
        '''
        self.set_reference_ct(reference_ct_name)
        time_list = []
        for fraction_name in self.cbcts.names:
            dct = self.get_fraction_dirs(fraction_name)
            fraction_dir = dct['fraction_dir']
            cbct_image_file = dct['image_file']
            cbct_contours_dir = dct['contours_dir']
            cbct_transforms_dir = dct['transforms_dir']
            cbct_dose_maps_dir = dct['dose_maps_dir']
            ct_external_mask = os.path.join(self.ct_contours_dir, self.external+'.mha')
            cbct_FOV_mask = os.path.join(cbct_contours_dir, self.__FOV_full_filename)
            ct_deformed = os.path.join(fraction_dir, self.__ct_deformed_filename)
            vf_dir = os.path.join(cbct_transforms_dir, self.__vf_DIR_filename.format(self.ct_name))
            t1 = time()
            PlastimatchAdaptive.propagate_contours(input_contours = self.ct_contours_dir,
                                                   output_contours = cbct_contours_dir,
                                                   fixed_image = cbct_image_file,
                                                   moving_image = self.ct_image_file,
                                                   output_image = ct_deformed,
                                                   output_vf = vf_dir,
                                                   fixed_mask = cbct_FOV_mask,
                                                   moving_mask = ct_external_mask,
                                                   reg_factor = 50,
                                                   translate_first = False)
            t2 = time()
            time_list.append(t2-t1)
        self.cbcts['contour_propagation_time'] = time_list
        self.cbcts.to_csv(os.path.join(self.fractions_parent_dir, self.__cbcts_info_filename))
        cbct_goal_dose = os.path.join(cbct_dose_maps_dir, self.__goal_dose_filename)
        PlastimatchAdaptive.warp_image(self.goal_dose, cbct_goal_dose, vf_dir, default_value = 0)
    
    def modify_cbct_beyond_FOV(self, fraction_name):
        # Using deformed CT as the background image
        dct = self.get_fraction_dirs(fraction_name)
        fraction_dir = dct['fraction_dir']
        image_file = dct['image_file']
        contours_dir = dct['contours_dir']
        FOV_full = os.path.join(contours_dir, self.__FOV_full_filename)
        # background_image = self.ct_image_file
        background_image = os.path.join(fraction_dir, self.__ct_deformed_filename)
        PlastimatchAdaptive.merge_images(background_image, image_file, image_file, FOV_full)
    
    def apply_cbct_HU_histogram_correction(self, fraction_name):
        # Using deformed CT as the reference image
        dct = self.get_fraction_dirs(fraction_name)
        fraction_dir = dct['fraction_dir']
        image_file = dct['image_file']
        contours_dir = dct['contours_dir']
        ct_deformed = os.path.join(fraction_dir, self.__ct_deformed_filename)
        # Get histogram mask --> intersect external with small FOV, exclude artifacts
        histogram_mask = os.path.join(fraction_dir, 'histogram_mask_temp.mha')
        PlastimatchAdaptive.get_intersection(histogram_mask,
                                             os.path.join(contours_dir, self.external+'.mha'),
                                             os.path.join(contours_dir, self.__FOV_small_filename))
        PlastimatchAdaptive.exclude_masks(histogram_mask,
                                          histogram_mask,
                                          os.path.join(contours_dir, self.artifacts+'.mha'))
        for method in ['mean', 'median', 'full']:
            output_file = os.path.join(fraction_dir, 'corrected_{}.mha'.format(method))
            PlastimatchAdaptive.values_histogram_matching(image_file,
                                                          ct_deformed,
                                                          output_file,
                                                          [histogram_mask],
                                                          [histogram_mask],
                                                          [histogram_mask],
                                                          -500, 1500, method)
        os.remove(histogram_mask)
        
    def run_dose_calculation(self, fraction_name, plan_name):
        dct = self.get_fraction_dirs(fraction_name)
        fraction_dir = dct['fraction_dir']
        image_file = dct['image_file']
        output_dir = os.path.join(fraction_dir, plan_name)
        plan_dir = os.path.join(self.plans_dir, plan_name)
        plan = DicomPlan(dicom_dir = plan_dir, machine_name = 'r330_01r')
        moqui = MoquiManager(plan.machine_name)
        moqui.run_simulation(output_dir = output_dir,
                             rtplan_file = plan.rtplan_file,
                             mode = 'dose',
                             image_file = image_file)
        log_path = os.path.join(fraction_dir, plan_name, 'log_dose.txt')
        with open(log_path, 'w') as f:
            f.write(moqui.log)
        self.convert_dose(fraction_name, plan_name)
    
    def convert_dose(self, fraction_name, plan_name):
        dct = self.get_fraction_dirs(fraction_name)
        fraction_dir = dct['fraction_dir']
        image_file = dct['image_file']
        dose_maps_dir = dct['dose_maps_dir']
        output_dir = os.path.join(fraction_dir, plan_name)
        dose_names = [i for i in os.listdir(output_dir) if '_dose.mha' in i]
        dose_files = [os.path.join(output_dir, i) for i in dose_names]
        output_dose_file = os.path.join(dose_maps_dir, fraction_name+'_'+plan_name+'.mha')
        if len(dose_files) > 1:
            PlastimatchAdaptive.sum_images(output_dose_file, *dose_files)
        elif len(dose_files) == 1:
            shutil.copyfile(dose_files[0], output_dose_file)
        PlastimatchAdaptive.run('convert',
                                input = output_dose_file,
                                output_img = output_dose_file,
                                fixed = image_file,
                                output_type = 'float')

    def run_plan_adaptation(self, fraction_name, plan_name):
        dct = self.get_fraction_dirs(fraction_name)
        fraction_dir = dct['fraction_dir']
        image_file = dct['image_file']
        contours_dir = dct['contours_dir']
        dose_maps_dir = os.path.join(fraction_dir, 'dose_maps')
        output_dir = os.path.join(fraction_dir, plan_name+'_adapted')
        plan_dir = os.path.join(self.plans_dir, plan_name)
        plan = DicomPlan(dicom_dir = plan_dir, machine_name = 'r330_01r')
        plan_adapter = PlanAdaptation(plan)
        goal_dose = os.path.join(dose_maps_dir, self.__goal_dose_filename)
        dij_mask = os.path.join(contours_dir, 'dij_mask.mha')
        structures = self.get_structure_files(fraction_name)
        plan_adapter.run(image_file = image_file,
                         output_dir = output_dir,
                         mask_file = dij_mask,
                         dose_file = goal_dose,
                         structures = structures)
        self.convert_dose(fraction_name, plan_name+'_adapted')

    def get_DVHs(self, fraction_name, structure_names=['all']):
        dct = self.get_fraction_dirs(fraction_name)
        image_file = dct['image_file']
        contours_dir = dct['contours_dir']
        dose_maps_dir = dct['dose_maps_dir']
        dvhs_dir = dct['dvhs_dir']
        evaluation = EvaluationManager(structures_dir = contours_dir)
        for dose_name in os.listdir(dose_maps_dir):
            dose_file = os.path.join(dose_maps_dir, dose_name)
            # Check if dose grid matches image grid (= contours grid)
            if PlastimatchAdaptive.run('header', dose_file) == PlastimatchAdaptive.run('header', image_file):
                output_dvh = os.path.join(dvhs_dir, 'data', os.path.splitext(dose_name)[0]+'.dvh')
                evaluation.set_dose(dose_file)
                evaluation.calculate_DVH(save_path = output_dvh)
        plots = DVHPlotter()
        plots.get_all_patient_plots(dvhs_dir, structure_names)
        
    def get_plastimatch_DVHs(self, fraction_name, structure_names=['all']):
        dct = self.get_fraction_dirs(fraction_name)
        image_file = dct['image_file']
        contours_dir = dct['contours_dir']
        dose_maps_dir = dct['dose_maps_dir']
        dvhs_dir = dct['dvhs_dir']+'_plastimatch'
        output_ss_img = os.path.join(dvhs_dir, 'structs_ss_{}.mha'.format(fraction_name))
        output_ss_list = os.path.join(dvhs_dir, 'structs_ss_{}.txt'.format(fraction_name))
        PlastimatchAdaptive.run('convert',
                                input_prefix = contours_dir,
                                output_type = 'float',
                                output_ss_img = output_ss_img,
                                output_ss_list = output_ss_list,
                                fixed = image_file)
        for dose_name in os.listdir(dose_maps_dir):
            dose_file = os.path.join(dose_maps_dir, dose_name)
            output_dvh = os.path.join(dvhs_dir, 'data', os.path.splitext(dose_name)[0]+'.dvh')
            PlastimatchAdaptive.run('dvh',
                                    'cumulative',
                                    input_dose = dose_file,
                                    input_ss_img = output_ss_img,
                                    input_ss_list = output_ss_list,
                                    output_csv = output_dvh,
                                    dose_units = 'gy',
                                    num_bins = '901',
                                    bin_width = '0.1')
        plots = DVHPlotter()
        plots.get_all_patient_plots(dvhs_dir, structure_names)
    
    def purge(self):
        if os.path.exists(self.patient_dir):
            shutil.rmtree(self.patient_dir)
            