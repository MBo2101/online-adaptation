# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 10:30:55 2021

@author: MBo
"""

import sys
from DirStructure import DirStructure

# Water phantom
# plan_name = 'refcube'
# input_dicom_dir = '/shared/build/moqui/example/refcube/DICOM/'
# input_cbcts_dir = '/home/mislav/Desktop/adaptive_test/cbct_fake/'
# output_patient_dir = '/home/mislav/Desktop/adaptive_test/Patient_01/'
# specific_fractions = [1]
# target_names = ['CTV-Lunder_Medium', 'InnerMedium']
# oar_names = [None]
# external_mask_name = None
# artifacts_mask_name = None

# Head and neck
plan_name = 'OA_SIB_CTV'
input_dicom_dir = '/home/mislav/Desktop/adaptive_test/Dicom_plans/patient_27'
input_cbcts_dir = '/adaptive/XVI_CBCT_data/scatter_corrected/rescan_0518/patient_27'
output_patient_dir = '/home/mislav/Desktop/adaptive_test/Patient_27'
specific_fractions = [1]
target_names = ['HR_CTV', 'LR_CTV_uniform', 'CTV_gradient']
oar_names = ['Larynx', 'Parotid_L', 'Parotid_R', 'Constrictors', 'SpinalCord', 'Brainstem']
external_mask_name = 'skin'
artifacts_mask_name = 'Artifact'

# Prostate
# plan_name = 'plan'
# input_dicom_dir = '/home/mislav/Desktop/adaptive_test/Dicom_plans/Prostate'
# output_patient_dir = '/home/mislav/Desktop/adaptive_test/Prostate_patient'
# target_names = ['PTV70robust']
# oar_names = ['ConformalityShell']
# external_mask_name = 'External'
# artifacts_mask_name = None

# Liver
# plan_name = 'plan'
# input_dicom_dir = '/home/mislav/Desktop/adaptive_test/Dicom_plans/Liver'
# output_patient_dir = '/home/mislav/Desktop/adaptive_test/Liver_patient'
# target_names = ['PTV_2400']
# oar_names = ['ConformalityShell']
# external_mask_name = 'External'
# artifacts_mask_name = None

# Set up directories and import CT data
patient_dir = DirStructure(output_patient_dir)
patient_dir.purge()
patient_dir.import_plan(input_dicom_dir, plan_name)

# Set structure names
patient_dir.set_structure_names(targets = target_names,
                                oars = oar_names,
                                external = external_mask_name,
                                artifacts = artifacts_mask_name)

# CT
patient_dir.generate_masks('ct') # Get additional masks: "target", "dij_mask", "conformality_shell"
patient_dir.modify_image_HUs('ct') # Modify image file: external and artifact masking
patient_dir.run_dose_calculation('ct', plan_name) # Run dose calculation for the nominal plan
patient_dir.run_plan_adaptation('ct', plan_name) # Run plan adaptation
patient_dir.get_plastimatch_DVHs('ct', target_names+oar_names) # Generate DVHs using Plastimatch

sys.exit()

# Import CBCTs & run image registration: loops all fractions
patient_dir.import_cbcts(input_cbcts_dir, specific_fractions)
patient_dir.match_cbcts_isocenter()
patient_dir.run_cbcts_DIR()

# sys.exit()

# CBCTs
for cbct_name in patient_dir.cbcts.names:
    patient_dir.generate_masks(cbct_name) # Get additional masks: "target", "dij_mask", "conformality_shell"
    patient_dir.modify_cbct_beyond_FOV(cbct_name) # Modify image file: filling CBCT outside the FOV
    patient_dir.modify_image_HUs(cbct_name) # Modify image file: external and artifact masking
    patient_dir.apply_cbct_HU_histogram_correction(cbct_name) # Modify image file: HU histogram-based correction
    patient_dir.run_dose_calculation(cbct_name, plan_name) # Run dose calculation for the nominal plan
    patient_dir.run_plan_adaptation(cbct_name, plan_name) # Run plan adaptation
    patient_dir.get_plastimatch_DVHs(cbct_name) # Generate DVHs using Plastimatch



