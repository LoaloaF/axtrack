import pickle
import os
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from scipy.stats import linregress

import config

from utils import turn_tex

import matplotlib.pyplot as plt
import plotting
from plotting import (
    plot_axon_IDs_lifetime,
    plot_target_distance_over_time,
    plot_n_axons,
    plot_axon_growthspeed,
    plot_axon_destinations,
    # plot_growth_directions,
    # plot_counted_growth_directions,
    plot_target_distance_over_time_allscreens,
)

class PDMSDesignScreen(object):
    def __init__(self, structure_screens, directory, tex=True):
        self.structure_screens = structure_screens
        
        self.dir = directory
        os.makedirs(self.dir, exist_ok=True)

        self.str_names = [ss.name for ss in structure_screens]
        print(self.str_names)
        self.identifiers = [ss.identifier for ss in structure_screens]
        # self.metrics = pd.concat([ss.metric for ss in structure_screens], axis=1)
        # self.metrics.columns.set_names(['timelapse','design','CLSM_area'], inplace=True)

        self.index = pd.MultiIndex.from_tuples(self.identifiers, names=['timelapse','design','CLSM_area'])
        self.get_index = pd.Series(range(len(self)), index=self.index)
        self.get_index = self.get_index.to_frame()
        self.get_index['notes'] = [ss.notes for ss in self]
        print(f'\n{config.SPACER}\nScreening object initialized with the '
              f'following samples:\n{self.get_index.to_string()}\n{config.SPACER}\n')

        if tex:
            turn_tex('on')

        
    
    def __len__(self):
        return len(self.structure_screens)

    def __getitem__(self, idx):
        return self.structure_screens[idx]

    def __iter__(self):
        for ss in self.structure_screens:
            yield ss

                
    # ==================  VALIDATION  ========================
    # single screens
    def sss_validate_masks(self, symlink_results=False, plot_kwargs={}):
        print('Validating masks of structure screens...', end='', flush=True)
        
        if symlink_results:
            dest_dir = f'{self.dir}/validate_masks/'
            os.makedirs(dest_dir, exist_ok=True)
        
        for ss in self:
            print(f'{ss.name}', end='...', flush=True)
            fname = plotting.draw_mask(ss, **plot_kwargs)
            if symlink_results:
                dest_ln_fname = dest_dir+os.path.basename(fname)
                if not os.path.exists(dest_ln_fname):
                    os.symlink(fname, dest_ln_fname)
        print('Done.\n')
    
    # single screens
    def sss_validate_IDlifetime(self, symlink_results=False, plot_kwargs={}):
        print('Validating ID lifetimes...', flush=True, end='')

        if symlink_results:
            dest_dir = f'{self.dir}/ID_lifetimes/'
            os.makedirs(dest_dir, exist_ok=True)

        for ss in self:
            print(f'{ss.name}', end='...', flush=True)
            fname = plotting.plot_axon_IDs_lifetime(ss, **plot_kwargs)
            if symlink_results:
                dest_ln_fname = dest_dir+os.path.basename(fname)
                if not os.path.exists(dest_ln_fname):
                    os.symlink(fname, dest_ln_fname)
        print('Done.\n')
            
    # single screens
    def sss_target_distance_timeline(self, symlink_results=False, plot_kwargs={}):
        print('Plotting target distance over time...', flush=True, end='')

        if symlink_results:
            dest_dir = f'{self.dir}/target_distance_timeline/'
            print(dest_dir)
            os.makedirs(dest_dir, exist_ok=True)

        for ss in self:
            print(f'{ss.name}', end='...', flush=True)
            fname = plotting.plot_target_distance_over_time(ss, **plot_kwargs)
            if symlink_results:
                dest_ln_fname = dest_dir+os.path.basename(fname)
                if not os.path.exists(dest_ln_fname):
                    os.symlink(fname, dest_ln_fname)
        print('Done.\n')

    
    

    # ======== CHECK GENERAL DEVELOPMENT OF SPEED AND DISTANCE TO TARGET ======
    # compare designs
    def cs_target_distance_timeline(self, speed=False, plot_kwargs={}):
        print('Plotting target distance over time for designs...', flush=True, end='')
        all_delta_dists = []
        for ss in self:
            dists = ss.get_distances()
            if speed:
                delta_dists = ss.get_growth_speed(dists, average=False)
            else:
                delta_dists = ss.subtract_initial_dist(dists)

            names = ['timelapse','design','CLSM_area','axon_id']
            delta_dists.index = pd.MultiIndex.from_product([*[[iden] for iden in ss.identifier], 
                                                            delta_dists.index], names=names)
            all_delta_dists.append(delta_dists)
        all_delta_dists = pd.concat(all_delta_dists, axis=0)
        plot_target_distance_over_time_allscreens(all_delta_dists, self.dir, speed=speed, **plot_kwargs)
        print('Done.')
            
    # compare structures 
    def cs_naxons(self, DIV_range=None, plot_kwargs={}):
        print('Plotting number of axons identified in each structure...', end='')
        
        n_axons = [ss.get_distances(DIV_range=DIV_range).shape[0] for ss in self]
        n_axons = pd.Series(n_axons, self.index)
        print(n_axons.to_string())
        plot_n_axons(n_axons, self.dir, DIV_range, **plot_kwargs)

    # compare structures 
    def cs_axon_growthspeed(self, DIV_range=None, plot_kwargs={}):
        # print('Plotting number of axons identified in each structure.')

        if DIV_range is not None:
            if 'fname_postfix' not in plot_kwargs:
                plot_kwargs['fname_postfix'] = ''
            plot_kwargs['fname_postfix'] += f'_DIV{DIV_range}'

        
        all_growth_speeds = []
        for ss in self:
            dists = ss.get_distances(DIV_range=DIV_range)
            growth_speed = ss.get_growth_speed(dists, average=True, absolute=True).reset_index(drop=True)
            identifier = ss.identifier
            identifier = tuple(list(identifier) + ss.design_features)

            all_growth_speeds.append(growth_speed.rename(identifier))
        all_growth_speeds = pd.concat(all_growth_speeds, axis=1).T
        
        index_names = ['timelapse','design','CLSM_area']
        index_names.extend(config.DESIGN_FEATURE_NAMES)
        all_growth_speeds.index.rename(index_names, inplace=True)

        fname = plotting.plot_axon_distribution(all_growth_speeds, self.dir, **plot_kwargs)

    















    # ===================== BEST DIRECTIONALITY STRCUTRES ======================
    # compare structures 
    def cs_axon_growth_direction(self, DIV_range=None, counted=False, plot_kwargs={}):
        all_growth_direcs = []
        for ss in self:
            dists = ss.get_distances(DIV_range=DIV_range)
            growth_dir = ss.growth_direction_start2end(dists)
            
            identifier = ss.identifier
            identifier = tuple(list(identifier) + ss.design_features)
            all_growth_direcs.append(growth_dir.rename(identifier))

        all_growth_direcs = pd.concat(all_growth_direcs, axis=1).T
        index_names = ['timelapse','design','CLSM_area']
        index_names.extend(config.DESIGN_FEATURE_NAMES)
        all_growth_direcs.index.rename(index_names, inplace=True)

        if not counted:
            fname = plot_growth_directions(all_growth_direcs, self.dir, **plot_kwargs)
        else:
            plot_counted_growth_directions(all_growth_direcs, self.dir, **plot_kwargs)

    # compare structures 
    
    def cs_axon_destinations(self, DIV_range=None, save_single_plots=False, 
                             neg_comp_metric='directions', relative_naxons=False,
                             norm_to_sum=True,
                             plot_kwargs={}):
        print('\nPlotting axon destinations...', end='')
        destinations = []
        for ss in self:
            print(ss.name, end='...', flush=True)
            dists, DIV_mask = ss.get_distances(DIV_range=DIV_range, return_DIV_mask=True)
            
            kwargs = {'dists':dists, 'DIV_mask':DIV_mask, 'draw':save_single_plots}
            reached_target_axons, crossgrown_axons, stagnating_axons = ss.get_special_axons(kwargs)
            directions = ss.get_growth_directions({'dists':dists})

            if neg_comp_metric == 'cross_grown':
                metrics = np.array([len(reached_target_axons),
                                    len(crossgrown_axons),

                ], dtype=float)
            elif neg_comp_metric == 'directions':
                metrics = np.array([(directions<-config.MIN_GROWTH_DELTA).sum(),
                                    (directions>config.MIN_GROWTH_DELTA).sum(),

                ], dtype=float)
            # elif neg_comp_metric == 'wrong_dir_200':
            #     metrics = np.array([len(reached_target_axons),
            #                         (directions<-200).sum(),
            #     ], dtype=float)
            
            if norm_to_sum:
                metrics = np.append(metrics, (metrics[0]+1)/(metrics[1]+1) )
            else:
                if metrics.sum() < 3:
                    metrics = np.array([np.nan, np.nan])
                metrics = np.append(metrics, metrics[0]/(metrics[0]+(metrics[1])))
                metrics[2] = metrics[2]*10
            
            destin = pd.Series(metrics, index=['pos_metric','neg_metric', 'metric'], 
                               name=ss.identifier)

            add_design_features = True
            
            if add_design_features:
                identifier = ss.identifier
                identifier = tuple(list(identifier) + ss.design_features)
                destinations.append(destin.rename(identifier))
            else:
                destinations.append(destin)     


        destinations = pd.concat(destinations, axis=1).T
        index_names = ['timelapse','design','CLSM_area']
        if add_design_features:
            index_names.extend(config.DESIGN_FEATURE_NAMES)

        destinations.index.rename(index_names, inplace=True)
        destinations.to_csv(f'{self.dir}/results.csv')
        plot_axon_destinations(destinations, self.dir, neg_comp_metric, **plot_kwargs)