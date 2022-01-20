# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 10:28:25 2021

@author: MBo
"""

import os
import numpy as np
import matplotlib.pyplot as plt
    
##############################################################################

def single_DVH_plot(file_path, structures=["target","oars"], savepath = None):
    
    with open(file_path, 'r') as f:
        file_lines_list = f.readlines()
    
    structures_list = file_lines_list[0].split(',')
    file_name = file_path.split('/')[-1]
    
    global data
    data = np.genfromtxt(file_path, delimiter = ',', skip_header = 1, names = structures_list)

    data_fields_list = list(data.dtype.names)
    data_fields_list.remove("Dose_Gy")
    
    plt.figure(figsize=(14, 8), dpi=100)
    #plt.subplot(1,1,1)
    
    if structures == ["all"]:
        for structure in data_fields_list:
            plt.plot(data["Dose_Gy"],data[structure])
            plt.legend(data_fields_list)
    else:
        for structure in structures:
            plt.plot(data["Dose_Gy"],data[structure])
            plt.legend(structures)
        
    plt.xlabel("Dose [Gy]")
    plt.ylabel("Volume [%]")    
    plt.grid()
    plt.title(file_name)
    
    if type(savepath) == str:
        plt.savefig(os.path.join(savepath, file_name+'.png'))
        
    plt.show()

##############################################################################
    
def compare_DVH_plot(file_1_path, file_2_path, structures=["target","oars"], savepath = None, image_title = None, label_1 = None, label_2 =None):

    with open(file_1_path, 'r') as f:
        file_lines_list = f.readlines()
    structures_list_1 = file_lines_list[0].split(',')
    #file_name_1 = file_1_path.split('/')[-1]
    data_1 = np.genfromtxt(file_1_path, delimiter = ',', skip_header = 1, names = structures_list_1)
    
    with open(file_2_path, 'r') as f:
        file_lines_list = f.readlines()
    structures_list_2 = file_lines_list[0].split(',')
    #file_name_2 = file_2_path.split('/')[-1]
    data_2 = np.genfromtxt(file_2_path, delimiter = ',', skip_header = 1, names = structures_list_2)

    data_fields_list_1 = list(data_1.dtype.names)
    data_fields_list_1.remove("Dose_Gy")
    data_fields_list_2 = list(data_2.dtype.names)
    data_fields_list_2.remove("Dose_Gy")

    my_colors = ['tab:red','tab:blue','tab:olive','tab:green','tab:purple',
                 'tab:orange','tab:pink','tab:cyan','tab:brown','tab:gray']

    plt.figure(figsize=(14, 8), dpi=100)
    #plt.subplot(1,1,1)
    
    if structures == ["all"]:
        for structure in data_fields_list_1:
            plt.plot(data_1["Dose_Gy"],data_1[structure],"r-")
        for structure in data_fields_list_2:
            plt.plot(data_2["Dose_Gy"],data_2[structure],"b:")
            
    else:
        for structure in structures:
            current_color = my_colors[structures.index(structure)]
            
            plt.plot(data_1["Dose_Gy"],data_1[structure], c=current_color, ls='--', linewidth=1.5, alpha=0.7)
            plt.plot(data_2["Dose_Gy"],data_2[structure], c=current_color, ls='-',  linewidth=1.5, alpha=1, label=structure)
    
    first_legend = plt.legend((label_1, label_2), loc = 'right', prop = {'size':12})
    for legobj in first_legend.legendHandles:
        legobj.set_linewidth(1.5)
        legobj.set_color('k')
    plt.gca().add_artist(first_legend)
    
    second_legend = plt.legend()
    for legobj in second_legend.legendHandles:
        legobj.set_linewidth(2.0)
        
    #plt.legend(('base_plan', 'adapted'), loc = 'lower right', color = 'k')
    
    plt.xlabel("Dose [Gy]")
    plt.ylabel("Volume [%]")        
    plt.grid()
    plt.title(image_title)
    
    if type(savepath) == str:
        plt.savefig(os.path.join(savepath, image_title+'.png'))
        
    plt.show()

##############################################################################
    
def get_all_patient_plots(dvh_folder, structures_list):
    
    data_folder  = os.path.join(dvh_folder, 'data')
    plots_folder = os.path.join(dvh_folder, 'plots')
    if not os.path.exists(plots_folder):
        os.makedirs(plots_folder)
    dvh_files = os.listdir(data_folder)
    for filename in dvh_files:
        file_path = os.path.join(data_folder, filename)
        single_DVH_plot(file_path, structures_list, plots_folder)

##############################################################################
    
def DVH_any_plot(structures, savepath, image_title, reference_ct_dvh_path, *args):
    '''
    Function assumes lists as arguments specifying each DVH file.
    [DVH_file_path, label_string]'''
    
    #my_colors = ['tab:red','tab:blue','limegreen','mediumpurple','tab:orange',
    #             'deepskyblue','tab:pink','tab:olive','tab:brown','tab:gray']
    
    my_colors = ['tab:red','tab:blue','limegreen','mediumpurple','tab:orange',
                 'deepskyblue','tab:pink','tab:olive','tab:brown','tab:gray',
                 'deepskyblue','tab:pink','tab:olive','tab:brown','tab:gray']
    
    ls_densely_dotted = (0, (1, 1))
    
    #my_linestyles = [ls_densely_dotted,'--', '-']
    my_linestyles = [ls_densely_dotted,'--','-']
    my_alphas     = [1, 1, 1]
    
    # my_linestyles = [':',':',':',':','-']
    # my_alphas = [ 1, 1, 1, 1, 1]
    
    if len(args) == 1:
        my_linestyles = ['-']
    
    labels_list = []
    
    structures_dictionary = {
            'target': 'CTV',
            'Parotid_L': 'L. Parotid',
            'Parotid_R': 'R. Parotid',
            'Larynx': 'Larynx',
            'SpinalCord': 'Spinal Cord',
            'Brainstem': 'Brainstem',
            'Cavity_Oral' : 'Oral Cavity',
            'Constrictors': 'Constrictors',
            'Glnd_Submand_R': 'R. Submand. Gl.',
            'Glnd_Submand_L': 'L. Submand. Gl.',
            'Mandible': 'Mandible',
            'Esophagus': 'Esophagus',
            'HR_CTV': 'HR CTV',
            'LR_CTV': 'LR CTV',
            'LR_CTV_uniform': 'LR CTV'
            }
    
    fig = plt.figure(figsize=(8, 5), dpi=200)
    ax = fig.add_subplot(1, 1, 1)
    
    
    for DVH_file_list in args:
        
        file_path = DVH_file_list[0]
        labels_list.append(DVH_file_list[1])
    
        with open(file_path, 'r') as f:
            file_lines_list = f.readlines()
        all_structures_list = file_lines_list[0].split(',')
        data = np.genfromtxt(file_path, delimiter = ',', skip_header = 1, names = all_structures_list)
    
        data_fields_list = list(data.dtype.names)
        data_fields_list.remove("Dose_Gy")
        
        if structures == ["all"]:
    
            for structure in data_fields_list:
                ax.plot(data["Dose_Gy"], data[structure], c=my_colors[args.index(DVH_file_list)])
                
        else:
            for structure in structures:
                
                ax.plot(data["Dose_Gy"], data[structure],
                        color     = my_colors[structures.index(structure)],
                        linestyle = my_linestyles[args.index(DVH_file_list)],
                        alpha = my_alphas[args.index(DVH_file_list)], 
                        #linewidth = 1.5, 
                        linewidth = 1, 
                        label = structure)
                
        if args.index(DVH_file_list) == 0:
            structures_legend = ax.legend()
            
    if reference_ct_dvh_path != None:
        
        with open(reference_ct_dvh_path, 'r') as f:
            file_lines_list = f.readlines()
        all_structures_list = file_lines_list[0].split(',')
        data = np.genfromtxt(reference_ct_dvh_path, delimiter = ',', skip_header = 1, names = all_structures_list)
        
        for structure in structures:
            ax.plot(data["Dose_Gy"], data[structure], 'k-', linewidth=0.3)
        
    case_legend = plt.legend(labels_list, loc = 'lower right', prop = {'size':12})
    for legobj in case_legend.legendHandles:
        legobj.set_linewidth(1.5)
        legobj.set_color('k')
        legobj.set_linestyle(my_linestyles[(case_legend.legendHandles).index(legobj)])
    plt.gca().add_artist(case_legend)
        
    for legobj in structures_legend.legendHandles:
        legobj.set_linewidth(2.0)
        legobj.set_linestyle('-')
            
    for structure_label in (structures_legend.get_texts()):
        structure_label.set_text(structures_dictionary[structure_label.get_text()])
        
    plt.gca().add_artist(structures_legend)

    #plt.xlim(0, 100)

    plt.xlabel("Dose [Gy]")
    plt.ylabel("Volume [%]")        
    plt.grid(alpha=0.7)
    plt.title(image_title)
    
    if type(savepath) == str:
        plt.savefig(os.path.join(savepath, image_title+'.png'))
        
    plt.show()
    
##############################################################################
    
def DVH_bands_plot(structures, savepath, image_title, DVH_filepaths, nominal_DVH_index):
    '''
    Function assumes list containing filepaths to each DVH file.
    First filepath is the nominal DVH (solid line).
    '''
    
    #my_colors = ['tab:red','tab:blue','limegreen','mediumpurple','tab:orange',
    #             'deepskyblue','tab:pink','tab:olive','tab:brown','tab:gray']
    
    my_colors = ['tab:red','tab:blue','limegreen','mediumpurple','tab:orange',
                 'deepskyblue','tab:pink','tab:olive','tab:brown','tab:gray',
                 'deepskyblue','tab:pink','tab:olive','tab:brown','tab:gray']
    
    structures_dictionary = {
            'target': 'CTV',
            'Parotid_L': 'L. Parotid',
            'Parotid_R': 'R. Parotid',
            'Larynx': 'Larynx',
            'SpinalCord': 'Spinal Cord',
            'Brainstem': 'Brainstem',
            'Cavity_Oral' : 'Oral Cavity',
            'Constrictors': 'Constrictors',
            'Glnd_Submand_R': 'R. Submand. Gl.',
            'Glnd_Submand_L': 'L. Submand. Gl.',
            'Mandible': 'Mandible',
            'Esophagus': 'Esophagus',
            'HR_CTV': 'HR CTV',
            'LR_CTV': 'LR CTV',
            'LR_CTV_uniform': 'LR CTV'
            }
    
    fig = plt.figure(figsize=(8, 5), dpi=200)
    ax = fig.add_subplot(1, 1, 1)
    
    nominal_DVH_filepath = DVH_filepaths[nominal_DVH_index]
    
    num_bins = 901 # Should be the same for all DVHs
    dtype_list = [(i, '<f8') for i in structures]
    
    # Create empty array for the data
    data = np.ndarray([num_bins, len(DVH_filepaths)], dtype_list)
    
    ############## Get data from all DVHs
    
    for file_path in DVH_filepaths:
    
        with open(file_path, 'r') as f:
            file_lines_list = f.readlines()
        all_structures_list = file_lines_list[0].split(',')
        data_current_DVH = np.genfromtxt(file_path, delimiter = ',', skip_header = 1, names = all_structures_list)
    
        data_fields_list = list(data_current_DVH.dtype.names)
        data_fields_list.remove("Dose_Gy")
        
        dvh_index = DVH_filepaths.index(file_path)
        
        if structures == ["all"]:
    
            for structure in data_fields_list:

                values = np.reshape(data_current_DVH[structure], (num_bins,1))
                np.put_along_axis(data[structure], np.array([[dvh_index]]), values, axis=1)
                
        else:
            for structure in structures:
                
                values = np.reshape(data_current_DVH[structure], (num_bins,1))
                np.put_along_axis(data[structure], np.array([[dvh_index]]), values, axis=1)
                
    dose_array = data_current_DVH['Dose_Gy']
                
    ##############
    
    data_stats = np.ndarray([num_bins, 3], dtype_list)
                
    for structure in structures:
                
        values_min  = np.reshape(np.amin(data[structure], axis=1), (num_bins,1))
        values_max  = np.reshape(np.amax(data[structure], axis=1), (num_bins,1))        
        values_mean = np.reshape(np.mean(data[structure], axis=1), (num_bins,1))
        
        np.put_along_axis(data_stats[structure], np.array([[0]]), values_min, axis = 1)
        np.put_along_axis(data_stats[structure], np.array([[1]]), values_max, axis = 1)
        np.put_along_axis(data_stats[structure], np.array([[2]]), values_mean, axis = 1)
        
    for structure in structures:

        ax.plot(dose_array, data_stats[structure][:,0], c = my_colors[structures.index(structure)], ls = ':', linewidth=0.7)
        ax.plot(dose_array, data_stats[structure][:,1], c = my_colors[structures.index(structure)], ls = ':', linewidth=0.7)
        ax.fill_between(dose_array,
                        data_stats[structure][:,0], 
                        data_stats[structure][:,1],
                        facecolor = my_colors[structures.index(structure)],
                        alpha=0.3)
        
    with open(nominal_DVH_filepath, 'r') as f:
        file_lines_list = f.readlines()
    all_structures_list = file_lines_list[0].split(',')
    nominal_data = np.genfromtxt(nominal_DVH_filepath, delimiter = ',', skip_header = 1, names = all_structures_list)
    
    for structure in structures:
        ax.plot(nominal_data["Dose_Gy"], 
                nominal_data[structure], 
                c = my_colors[structures.index(structure)], 
                linewidth=1,
                label = structure)

    structures_legend = ax.legend()
        
    for legobj in structures_legend.legendHandles:
        legobj.set_linewidth(2.0)
        legobj.set_linestyle('-')
            
    for structure_label in (structures_legend.get_texts()):
        structure_label.set_text(structures_dictionary[structure_label.get_text()])
        
    plt.gca().add_artist(structures_legend)

    # plt.xlim(0, 100)

    plt.xlabel("Dose [Gy]")
    plt.ylabel("Volume [%]")        
    plt.grid(alpha=0.7)
    plt.title(image_title)
    
    if type(savepath) == str:
        plt.savefig(os.path.join(savepath, image_title+'.png'))
        
    plt.show()
    
    