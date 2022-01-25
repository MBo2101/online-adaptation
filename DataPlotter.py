# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 10:28:25 2021

@author: MBo
"""

'''
Copied from old code. Should rewrite at some point (make object-oriented, etc.)
'''

import os
import numpy as np
import matplotlib.pyplot as plt

class DVHPlotter(object):
    
    def __init__(self, **kwargs):
        self.figsize = kwargs.get('figsize')
        self.dpi = kwargs.get('dpi')
        # Add other variables: colors, linestyles, structures_dictionary, etc.

    def DVH_plot(self, file_path, structures=['all'], savepath = None):
        with open(file_path, 'r') as f:
            file_lines_list = f.readlines()
        structures_list = file_lines_list[0].split(',')
        filename = file_path.split('/')[-1]
        data = np.genfromtxt(file_path, delimiter = ',', skip_header = 1, names = structures_list)
        data_fields_list = list(data.dtype.names)
        data_fields_list.remove('Dose_Gy')
        if 'f0' in data_fields_list : data_fields_list.remove('f0')
        plt.figure(figsize=(14, 8), dpi=100)
        if structures == ['all']:
            for structure in data_fields_list:
                plt.plot(data['Dose_Gy'],data[structure])
                plt.legend(data_fields_list)
        else:
            for structure in structures:
                plt.plot(data['Dose_Gy'],data[structure])
                plt.legend(structures)
        plt.xlabel('Dose [Gy]')
        plt.ylabel('Volume [%]')
        plt.xlim([-5,95])
        plt.grid()
        plt.title(filename)
        if type(savepath) == str:
            plt.savefig(os.path.join(savepath, filename+'.png'))
        plt.show()
    
    def get_all_patient_plots(self, dvh_folder, structures_list):
        data_folder  = os.path.join(dvh_folder, 'data')
        plots_folder = os.path.join(dvh_folder, 'plots')
        if not os.path.exists(plots_folder):
            os.makedirs(plots_folder)
        dvh_files = os.listdir(data_folder)
        for filename in dvh_files:
            file_path = os.path.join(data_folder, filename)
            self.DVH_plot(file_path, structures_list, plots_folder)
    
    def DVH_any_plot(self, structures, savepath, image_title, reference_dvh_path, *args):
        '''
        Function assumes lists as arguments specifying each DVH file.
        [DVH_file_path, label_string]
        '''
        # plot_colors = ['tab:red','tab:blue','limegreen','mediumpurple','tab:orange',
        #                'deepskyblue','tab:pink','tab:olive','tab:brown','tab:gray']
        plot_colors = ['tab:red','tab:blue','limegreen','mediumpurple','tab:orange',
                       'deepskyblue','tab:pink','tab:olive','tab:brown','tab:gray',
                       'deepskyblue','tab:pink','tab:olive','tab:brown','tab:gray']
        ls_densely_dotted = (0, (1, 1))
        plot_linestyles = [ls_densely_dotted,'--', '-']
        plot_linestyles = [ls_densely_dotted,'--','-']
        plot_alphas     = [1, 1, 1]
        # plot_linestyles = [':',':',':',':','-']
        # plot_alphas = [ 1, 1, 1, 1, 1]
        if len(args) == 1:
            plot_linestyles = ['-']
        labels_list = []
        structures_dictionary = {'target': 'CTV',
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
            data_fields_list.remove('Dose_Gy')
            if 'f0' in data_fields_list : data_fields_list.remove('f0')
            if structures == ['all']:
                for structure in data_fields_list:
                    ax.plot(data['Dose_Gy'], data[structure], c=plot_colors[args.index(DVH_file_list)])
            else:
                for structure in structures:
                    ax.plot(data['Dose_Gy'], data[structure],
                            color     = plot_colors[structures.index(structure)],
                            linestyle = plot_linestyles[args.index(DVH_file_list)],
                            alpha = plot_alphas[args.index(DVH_file_list)], 
                            #linewidth = 1.5, 
                            linewidth = 1, 
                            label = structure)
            if args.index(DVH_file_list) == 0:
                structures_legend = ax.legend()
        if reference_dvh_path != None:
            with open(reference_dvh_path, 'r') as f:
                file_lines_list = f.readlines()
            all_structures_list = file_lines_list[0].split(',')
            data = np.genfromtxt(reference_dvh_path, delimiter = ',', skip_header = 1, names = all_structures_list)
            for structure in structures:
                ax.plot(data['Dose_Gy'], data[structure], 'k-', linewidth=0.3)
        case_legend = plt.legend(labels_list, loc = 'lower right', prop = {'size':12})
        for legobj in case_legend.legendHandles:
            legobj.set_linewidth(1.5)
            legobj.set_color('k')
            legobj.set_linestyle(plot_linestyles[(case_legend.legendHandles).index(legobj)])
        plt.gca().add_artist(case_legend)
        for legobj in structures_legend.legendHandles:
            legobj.set_linewidth(2.0)
            legobj.set_linestyle('-')
        for structure_label in (structures_legend.get_texts()):
            structure_label.set_text(structures_dictionary[structure_label.get_text()])
        plt.gca().add_artist(structures_legend)
        #plt.xlim(0, 100)
        plt.xlabel('Dose [Gy]')
        plt.ylabel('Volume [%]')        
        plt.grid(alpha=0.7)
        plt.title(image_title)
        if type(savepath) == str:
            plt.savefig(os.path.join(savepath, image_title+'.png'))
        plt.show()
        
    def DVH_bands_plot(self, structures, savepath, image_title, *args):
        '''
        DVH file paths are passed as *args.
        First file path is the nominal DVH (solid line).
        '''
        # plot_colors = ['tab:red','tab:blue','limegreen','mediumpurple','tab:orange',
        #                'deepskyblue','tab:pink','tab:olive','tab:brown','tab:gray']
        plot_colors = ['tab:red','tab:blue','limegreen','mediumpurple','tab:orange',
                     'deepskyblue','tab:pink','tab:olive','tab:brown','tab:gray',
                     'deepskyblue','tab:pink','tab:olive','tab:brown','tab:gray']
        structures_dictionary = {'target': 'CTV',
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
        nominal_DVH_file_path = args[0]
        num_bins = 901 # Should be the same for all DVHs
        dtype_list = [(i, '<f8') for i in structures]
        data = np.ndarray([num_bins, len(args)], dtype_list) 
        for file_path in args:
            with open(file_path, 'r') as f:
                file_lines_list = f.readlines()
            all_structures_list = file_lines_list[0].split(',')
            data_current_DVH = np.genfromtxt(file_path, delimiter = ',', skip_header = 1, names = all_structures_list)
            data_fields_list = list(data_current_DVH.dtype.names)
            data_fields_list.remove('Dose_Gy')
            if 'f0' in data_fields_list : data_fields_list.remove('f0')
            dvh_index = args.index(file_path)
            if structures == ['all']:
                for structure in data_fields_list:
                    values = np.reshape(data_current_DVH[structure], (num_bins,1))
                    np.put_along_axis(data[structure], np.array([[dvh_index]]), values, axis=1)
            else:
                for structure in structures:
                    values = np.reshape(data_current_DVH[structure], (num_bins,1))
                    np.put_along_axis(data[structure], np.array([[dvh_index]]), values, axis=1)
        dose_array = data_current_DVH['Dose_Gy']
        data_stats = np.ndarray([num_bins, 3], dtype_list)
        for structure in structures:
            values_min  = np.reshape(np.amin(data[structure], axis=1), (num_bins,1))
            values_max  = np.reshape(np.amax(data[structure], axis=1), (num_bins,1))        
            values_mean = np.reshape(np.mean(data[structure], axis=1), (num_bins,1))
            np.put_along_axis(data_stats[structure], np.array([[0]]), values_min, axis = 1)
            np.put_along_axis(data_stats[structure], np.array([[1]]), values_max, axis = 1)
            np.put_along_axis(data_stats[structure], np.array([[2]]), values_mean, axis = 1)
        for structure in structures:
            ax.plot(dose_array, data_stats[structure][:,0], c = plot_colors[structures.index(structure)], ls = ':', linewidth=0.7)
            ax.plot(dose_array, data_stats[structure][:,1], c = plot_colors[structures.index(structure)], ls = ':', linewidth=0.7)
            ax.fill_between(dose_array,
                            data_stats[structure][:,0], 
                            data_stats[structure][:,1],
                            facecolor = plot_colors[structures.index(structure)],
                            alpha=0.3)
        with open(nominal_DVH_file_path, 'r') as f:
            file_lines_list = f.readlines()
        all_structures_list = file_lines_list[0].split(',')
        nominal_data = np.genfromtxt(nominal_DVH_file_path, delimiter = ',', skip_header = 1, names = all_structures_list)
        for structure in structures:
            ax.plot(nominal_data['Dose_Gy'], 
                    nominal_data[structure], 
                    c = plot_colors[structures.index(structure)], 
                    linewidth=1,
                    label = structure)
        structures_legend = ax.legend()
        for legobj in structures_legend.legendHandles:
            legobj.set_linewidth(2.0)
            legobj.set_linestyle('-')
        for structure_label in (structures_legend.get_texts()):
            structure_label.set_text(structures_dictionary[structure_label.get_text()])
        plt.gca().add_artist(structures_legend)
        plt.xlabel('Dose [Gy]')
        plt.ylabel('Volume [%]')        
        plt.grid(alpha=0.7)
        plt.title(image_title)
        if type(savepath) == str:
            plt.savefig(os.path.join(savepath, image_title+'.png'))
        plt.show()
        
    