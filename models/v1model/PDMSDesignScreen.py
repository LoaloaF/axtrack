import pickle
import os
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from scipy.stats import linregress

import config

import matplotlib.pyplot as plt
from plotting import (
    plot_axon_IDs_lifetime,
    plot_target_distance_over_time,
)

class PDMSDesignScreen(object):
    def __init__(self, structure_screens, directory):
        self.structure_screens = structure_screens
        
        self.dir = directory
        os.makedirs(self.dir, exist_ok=True)

        self.str_names = [ss.name for ss in structure_screens]
        self.names = [(n[:4],n[5:8],n[9:13]) for n in self.str_names]
        idx = pd.MultiIndex.from_tuples(self.names, names=['timelapse','design','CLSM_area'])
        self.get_index = pd.Series(range(len(self)), index=idx)
        
        print(f'\n{config.SPACER}\nScreening object initialized with the '
              f'following samples:\n{self.get_index}')
        
    def __len__(self):
        return len(self.structure_screens)

    def __getitem__(self, idx):
        return self.structure_screens[idx]
    
    # def get(self, )


    # single screens
    def sss_ID_lifetime(self, symlink_results=False, plot_kwargs={}):
        print('Plotting ID lifetimes.')
        dest_dir = f'{self.dir}/ID_lifetimes/'
        os.makedirs(dest_dir, exist_ok=True)

        for i in range(len(self)):
            fname = plot_axon_IDs_lifetime(self[i], **plot_kwargs)
            dest_ln_fname = dest_dir+os.path.basename(fname)
            if symlink_results and not os.path.exists(dest_ln_fname):
                os.symlink(fname, dest_ln_fname)
            
    # single screens
    def sss_target_distance_over_time(self, symlink_results=False, plot_kwargs={}):
        print('Plotting target distance over time.')
        dest_dir = f'{self.dir}/target_distance_over_time/'
        os.makedirs(dest_dir, exist_ok=True)

        for i in range(len(self)):
            fname = plot_target_distance_over_time(self[i], **plot_kwargs)
            dest_ln_fname = dest_dir+os.path.basename(fname)
            if symlink_results and not os.path.exists(dest_ln_fname):
                os.symlink(fname, dest_ln_fname)
            
        
    
