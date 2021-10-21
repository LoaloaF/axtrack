import pickle
import os
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from scipy.stats import linregress

import config

from utils import turn_tex

import matplotlib.pyplot as plt
from plotting import (
    plot_axon_IDs_lifetime,
    plot_target_distance_over_time,
    plot_n_axons,
    plot_axon_growthspeed,
    plot_axon_destinations,
    plot_growth_directions,
)

class PDMSDesignScreen(object):
    def __init__(self, structure_screens, directory):
        self.structure_screens = structure_screens
        
        self.dir = directory
        os.makedirs(self.dir, exist_ok=True)

        self.str_names = [ss.name for ss in structure_screens]
        self.identifiers = [ss.identifier for ss in structure_screens]
        # self.metrics = pd.concat([ss.metric for ss in structure_screens], axis=1)
        # self.metrics.columns.set_names(['timelapse','design','CLSM_area'], inplace=True)

        self.index = pd.MultiIndex.from_tuples(self.identifiers, names=['timelapse','design','CLSM_area'])
        self.get_index = pd.Series(range(len(self)), index=self.index)
        self.get_index = self.get_index.to_frame()
        self.get_index['notes'] = [ss.notes for ss in self]
        print(f'\n{config.SPACER}\nScreening object initialized with the '
              f'following samples:\n{self.get_index}\n{config.SPACER}\n')
    
    def __len__(self):
        return len(self.structure_screens)

    def __getitem__(self, idx):
        return self.structure_screens[idx]

    def __iter__(self):
        for ss in self.structure_screens:
            yield ss

    # single screens
    def sss_ID_lifetime(self, symlink_results=False, plot_kwargs={}):
        print('Plotting ID lifetimes.')
        dest_dir = f'{self.dir}/ID_lifetimes/'
        os.makedirs(dest_dir, exist_ok=True)

        turn_tex('on')
        for i in range(len(self)):
            fname = plot_axon_IDs_lifetime(self[i], **plot_kwargs)
            dest_ln_fname = dest_dir+os.path.basename(fname)
            if symlink_results and not os.path.exists(dest_ln_fname):
                os.symlink(fname, dest_ln_fname)
        turn_tex('off')
            
    # single screens
    def sss_target_distance_over_time(self, symlink_results=False, plot_kwargs={}):
        print('Plotting target distance over time.')
        turn_tex('on')
        
        dest_dir = f'{self.dir}/target_distance_over_time/'
        os.makedirs(dest_dir, exist_ok=True)

        for i in range(len(self)):
            fname = plot_target_distance_over_time(self[i], **plot_kwargs)
            dest_ln_fname = dest_dir+os.path.basename(fname)
            if symlink_results and not os.path.exists(dest_ln_fname):
                os.symlink(fname, dest_ln_fname)

        turn_tex('off')
            
        
                # all_sizet_days = [self[i].sizet*self[i].dt /(60*24) for i in range(len(self))]
    
    # compare structures 
    def cs_naxons(self, DIV_range=None, plot_kwargs={}):
        print('Plotting number of axons identified in each structure.')
        turn_tex('on')
        
        n_axons = [ss.get_distances(DIV_range=DIV_range).shape[0] for ss in self]
        n_axons = pd.Series(n_axons, self.index)
        print(n_axons.to_string())

        fname = plot_n_axons(n_axons, self.dir, DIV_range, **plot_kwargs)
        turn_tex('off')
    
    # def show_masks(self, ):
    #     for ss in self:
    #         plt.subplots(figsize=(16,9))
    #         new_mask = ss.goal_mask.copy()
    #         new_mask[ss.start_mask.astype(bool)] = 2
    #         plt.imshow(new_mask)
    #         plt.title(ss.name)
    #         plt.savefig(f'{self.dir}/masks/{ss.name}.png')
    #         # plt.show()
    
    # compare structures 
    def cs_axon_growthspeed(self, DIV_range=None, percentile=None, absolute=True, plot_kwargs={}):
        # print('Plotting number of axons identified in each structure.')
        turn_tex('on')
        
        all_growth_speeds = []
        for ss in self:
            dists = ss.get_distances(DIV_range=DIV_range)
            growth_speed = ss.get_growth_speed(dists, average=True, absolute=absolute).reset_index(drop=True)
            if percentile:
                n = int(growth_speed.shape[0]*percentile)
                growth_speed = growth_speed.sort_values(ascending=False).iloc[:n]
            all_growth_speeds.append(growth_speed.rename(ss.identifier))
        all_growth_speeds = pd.concat(all_growth_speeds, axis=1).T
        all_growth_speeds.index.rename(['timelapse','design','CLSM_area'], inplace=True)

        fname = plot_axon_growthspeed(all_growth_speeds, self.dir, DIV_range, **plot_kwargs)

        turn_tex('off')
    
    # compare structures 
    def cs_axon_growth_direction(self, DIV_range=None, percentile=None, absolute=True, plot_kwargs={}):
        # print('Plotting number of axons identified in each structure.')
        turn_tex('on')
        
        all_growth_direcs = []
        for ss in self:
            dists = ss.get_distances(DIV_range=DIV_range)
            growth_dir = ss.growth_direction_start2end(dists)
            all_growth_direcs.append(growth_dir.rename(ss.identifier))
        all_growth_direcs = pd.concat(all_growth_direcs, axis=1).T
        all_growth_direcs.index.rename(['timelapse','design','CLSM_area'], inplace=True)

        fname = plot_growth_directions(all_growth_direcs, self.dir, **plot_kwargs)

        turn_tex('off')
    # compare structures 
    
    def cs_axon_destinations(self, DIV_range=None, plot_kwargs={}):
        # print('Plotting number of axons identified in each structure.')
        turn_tex('on')
        
        destinations = []
        for ss in self:

            dists = ss.get_distances(DIV_range=DIV_range)
            reached_target_axons, crossgrown_axons = ss.detect_axons_final_location(dists, draw=False)
            destin = pd.Series((len(reached_target_axons), len(crossgrown_axons)), 
                                index=['target','cross_growth'], name=ss.identifier)
            destinations.append(destin)     
        destinations = pd.concat(destinations, axis=1).T
        destinations.index.rename(['timelapse','design','CLSM_area'], inplace=True)
        
        plot_axon_destinations(destinations, self.dir, **plot_kwargs)
        turn_tex('off')