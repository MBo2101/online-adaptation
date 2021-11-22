# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 10:28:25 2021

@author: MBo
"""

import subprocess
import os

class MoquiManager(object):

    def __init__(self, machine_name):
        self.__exe_path = '/shared/build/moqui/moqui'
        self.__machine_name = machine_name
        self.__stdout_str = ''
    
    # Properties
    
    @property
    def exe_path(self):
        return self.__exe_path
    @property
    def machine_name(self):
        return self.__machine_name
    @property
    def stdout_str(self):
        return self.__stdout_str
    
    # Methods
    
    def print_properties(self):
        c = self.__class__
        props = [p for p in dir(c) if isinstance(getattr(c,p), property) and hasattr(self,p)]
        for p in props:
            print(p + ' = ' +str(getattr(self, p)))
    
    def run_simulation(self,
                       output_dir,
                       rtplan_file,
                       mode,
                       image_file=None,
                       tramps_dir=None,
                       masks=None):
        Scorer = mode
        if masks is not None:
            ScoringMask = 'true'
            Mask = ', '.join([i for i in masks])
        Machine = self.__machine_name
        OutputDir = output_dir
        if mode == 'dose':
            OutputFormat = 'mha'
            launcher_file = os.path.join(output_dir, 'moqui_dose.in')
            log_file = os.path.join(output_dir, 'moqui_dose.log')
        elif mode == 'dij':
            OutputFormat = 'npz'
            launcher_file = os.path.join(output_dir, 'moqui_dij.in')
            log_file = os.path.join(output_dir, 'moqui_dij.log')
        else:
            raise TypeError('\nMode not recognized. Use "dose" or "dij" mode.')
        DicomDir = os.path.dirname(rtplan_file)
        
        file_text = '## Global parameters for simulation'\
                    '\n'\
                    '\nGPUID 0'\
                    '\nRandomSeeed 1000 # (integer, use negative value if use current time)'\
                    '\nUseAbsolutePath true'\
                    '\nTotalThreads -1 # (integer, use negative value for using optimized number of threads)'\
                    '\nMaxHistoriesPerBatch 0'\
                    '\nVerbosity 0'\
                    '\nApertureVolume VOLUME'\
                    '\n'\
                    '\n## Quantity to score'\
                    '\n'\
                    '\nScorer {}'\
                    '\nSupressStd true'\
                    '\nScoringMask {}'\
                    '\nMask {}'\
                    '\nReadStructures false'\
                    '\nROIName External'\
                    '\n'\
                    '\n## Source parameters'\
                    '\n'\
                    '\nSourceType FluenceMap'\
                    '\nSimulationType perBeam'\
                    '\nBeamNumbers 0'\
                    '\nSourceExtension tramp'\
                    '\nParticleHistories 1000000'\
                    '\nCTClipping true'\
                    '\nMachine {}'\
                    '\nScoreToCTGrid'\
                    '\n'\
                    '\n## Output path'\
                    '\n'\
                    '\nOutputDir {}'\
                    '\nOutputFormat {}'\
                    '\nOverwriteResults true'\
                    '\n'\
                    '\n## Data directories'\
                    '\n'\
                    '\nDicomDir {}'\
                    .format(Scorer,
                            ScoringMask,
                            Mask,
                            Machine,
                            OutputDir,
                            OutputFormat,
                            DicomDir)
                    
        if image_file is not None:
            file_text += '\nCTVolumeName {}'.format(image_file)
        if tramps_dir is not None:
            file_text += '\nTrampDir {}'.format(tramps_dir)

        with open(launcher_file, 'w') as f:
            f.write(file_text)
            
        run = subprocess.Popen([self.__exe_path, launcher_file], stdout=subprocess.PIPE)
        while True:
            output_line = run.stdout.readline().decode('utf-8')
            if run.poll() is not None:
                break
            if output_line:
                self.__stdout_str += output_line
                print(output_line.strip())
        if run.poll() == 0:
            return self.__stdout_str
        else:
            raise Exception('Moqui error')
        
        with open(log_file) as f:
            f.write(self.__stdout_str)
