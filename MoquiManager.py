# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 10:28:25 2021

@author: MBo
"""

import subprocess
import os
from time import time

class MoquiManager(object):

    def __init__(self, machine_name):
        self.exe_path = '/shared/build/moqui/moqui_adaptive_v5'
        self.machine_name = machine_name
        self.log = ''
        self.sim_time = 0
    
    def run_simulation(self,
                       output_dir,
                       rtplan_file,
                       mode,
                       image_file=None,
                       tramps_dir=None,
                       masks=None,
                       random_seed=1000):
        '''
        Runs Moqui simulation.
        '''
        RandomSeed = str(random_seed)
        Scorer = mode
        ScoringMask = 'false'
        Mask = None
        if masks is not None and masks != [None]:
            ScoringMask = 'true'
            Mask = ', '.join([i for i in masks])
        Machine = self.machine_name
        OutputDir = output_dir
        if mode == 'dose': OutputFormat = 'mha'
        elif mode == 'dij': OutputFormat = 'npz'
        else: raise TypeError('\nMode not recognized. Use "dose" or "dij" mode.')
        DicomDir = os.path.dirname(rtplan_file)
        
        file_text = '## Global parameters for simulation'\
                    '\n'\
                    '\nGPUID 0'\
                    '\nRandomSeed {} # (integer, use negative value if use current time)'\
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
                    '\nParticlesPerHistory 20000'\
                    '\nCTClipping true'\
                    '\nMachine MGH:{}'\
                    '\nScoreToCTGrid true'\
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
                    .format(RandomSeed,
                            Scorer,
                            ScoringMask,
                            Mask,
                            Machine,
                            OutputDir,
                            OutputFormat,
                            DicomDir)
                    
        if image_file is not None: file_text += '\nCTVolumeName {}'.format(image_file)
        if tramps_dir is not None: file_text += '\nTrampDir {}'.format(tramps_dir)
        if not os.path.exists(output_dir): os.makedirs(output_dir)
        launcher_file = os.path.join(output_dir, 'moqui_{}.in'.format(mode))
        with open(launcher_file, 'w') as f:
            f.write(file_text)
        
        start_sim = time()
        run = subprocess.Popen([self.exe_path, launcher_file], stdout=subprocess.PIPE)
        while True:
            output_line = run.stdout.readline().decode('utf-8')
            if run.poll() is not None: break
            if output_line:
                self.log += output_line
                print(output_line.strip())
        end_sim = time()
        self.sim_time += end_sim - start_sim
        if run.poll() == 0:
            return True
        else:
            raise Exception('Moqui error')
