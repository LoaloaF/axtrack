import pickle
import os
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from scipy.stats import linregress

import config

from scipy.stats import mannwhitneyu
from scipy.stats import kruskal
from scipy.stats import f_oneway
from scipy.stats import normaltest
from scipy.stats import ttest_ind

from utils import turn_tex

import matplotlib.pyplot as plt
import screen_plotting
# from plotting import (
#     plot_axon_IDs_lifetime,
#     plot_target_distance_over_time,
#     # plot_n_axons,
#     plot_axon_destinations,
#     # plot_growth_directions,
#     # plot_counted_growth_directions,
#     plot_target_distance_over_time_allscreens,
# )

class PDMSDesignScreen(object):
    def __init__(self, structure_screens, directory):
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
            fname = screen_plotting.draw_mask(ss, **plot_kwargs)
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
            fname = screen_plotting.plot_axon_IDs_lifetime(ss, **plot_kwargs)
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
            fname = screen_plotting.plot_target_distance_over_time(ss, **plot_kwargs)
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
    def cs_axon_growth_counted(self, which_metric, DIV_range=None, plot_kwargs={}):
        all_metrics = []
        columns = [which_metric]
        for ss in self:
            dists, DIV_mask = ss.get_distances(DIV_range=DIV_range, add_side_dim=True, 
                                               return_DIV_mask=True)
            
            # n axons metric
            if which_metric == 'n_axons':
                metric_left = dists.index.get_level_values('side').values.sum()
                metric_right = dists.shape[0] - metric_left
                
            if which_metric == 'stagnating':
                left_dists = dists.loc(0)[(slice(None),0)].droplevel('side')
                right_dists = dists.loc(0)[(slice(None),1)].droplevel('side')
                metric_left = ss.detect_axons_final_location(dists=left_dists, DIV_mask=DIV_mask)
                metric_right = ss.detect_axons_final_location(dists=right_dists, DIV_mask=DIV_mask)
                
                metric_left = len(metric_left[2]),
                metric_right = len(metric_right[2]),
            
            
            elif which_metric == 'growth_direction_counted':
                directions = ss.get_growth_directions({'dists':dists})
                n_left, n_right = np.unique(directions.index.get_level_values('side'), return_counts=True)[1]
                neg_deltas = directions<-config.MIN_GROWTH_DELTA
                pos_deltas = directions>config.MIN_GROWTH_DELTA
                
                metric_left = neg_deltas.loc[(slice(None),0)].sum(), pos_deltas.loc[(slice(None),0)].sum()
                metric_right = neg_deltas.loc[(slice(None),1)].sum(), pos_deltas.loc[(slice(None),1)].sum()
                metric_left = np.array(metric_left) /n_left
                metric_right = np.array(metric_right) /n_right
                columns = ['n_right_dir', 'n_wrong_dir']
            
            elif which_metric == 'destinations':
                left_dists = dists.loc(0)[(slice(None),0)].droplevel('side')
                right_dists = dists.loc(0)[(slice(None),1)].droplevel('side')
                n_left, n_right = left_dists.shape[0], right_dists.shape[0]
                metric_left = ss.detect_axons_final_location(dists=left_dists, DIV_mask=DIV_mask)
                metric_right = ss.detect_axons_final_location(dists=right_dists, DIV_mask=DIV_mask)
                
                metric_left = len(metric_left[0]), len(metric_left[1])
                metric_left = metric_left if metric_left[0] or metric_left[1] else (np.nan, np.nan)

                metric_right = len(metric_right[0]), len(metric_right[1])
                metric_right = metric_right if metric_right[0] or metric_right[1] else (np.nan, np.nan)
                columns = ['target', 'neighbor']
            
            idx = [(list(ss.identifier) + [s] + ss.design_features) for s in ('left', 'right')]
            idx = pd.MultiIndex.from_tuples(idx, names=['timelapse','design','CLSM_area', 'side'] 
                                                        + config.DESIGN_FEATURE_NAMES)
            metric = pd.DataFrame([metric_left, metric_right], index=idx,
                                  columns=columns)
            all_metrics.append(metric)
        all_metrics = pd.concat(all_metrics)
        
        testing_data = screen_plotting.plot_axon_distribution(all_metrics, self.dir, 
                                       which_metric=which_metric, **plot_kwargs)
        name=f'{which_metric}_{plot_kwargs.get("split_by")}{plot_kwargs.get("fname_postfix")}'
        self.test(testing_data, name=name,
                  dest_dir=self.dir)

    # compare structures v
    def cs_axon_growth(self, DIV_range=None, which_metric='speed', plot_kwargs={}):
        print(f'\n\n\n\n\nPlotting: {plot_kwargs}...')

        all_metrics = []
        for ss in self:
            dists = ss.get_distances(DIV_range=DIV_range)
            if which_metric == 'speed':
                metric = ss.get_growth_speed(dists, average=True, absolute=True).reset_index(drop=True)
            elif which_metric == 'growth_direction':
                metric = ss.growth_direction_start2end(dists, drop_axon_names=True)
            elif which_metric == 'growth_direction_capped':
                metric = ss.growth_direction_start2end(dists, drop_axon_names=True, 
                                                       drop_small_deltas=True)
            
            identifier = tuple(list(ss.identifier) + ss.design_features)
            all_metrics.append(metric.rename(identifier))
        all_metrics = pd.concat(all_metrics, axis=1).T
        
        index_names = ['timelapse','design','CLSM_area']
        index_names.extend(config.DESIGN_FEATURE_NAMES)
        all_metrics.index.rename(index_names, inplace=True)

        testing_data = screen_plotting.plot_axon_distribution(all_metrics, self.dir, 
                                        which_metric=which_metric, **plot_kwargs)
        
        self.test(testing_data, name=f'{which_metric}_{plot_kwargs.get("split_by")}',
                    dest_dir=self.dir)

    def test(self, testing_data, name, dest_dir, parametric=False):
        if not isinstance(testing_data, tuple):
            testing_data = testing_data,

        output = ''
        for testing_dat in testing_data:
            if not parametric:
                values = [vals for vals in testing_dat.values() if len(np.unique(vals))>1] 
                t_stat, p = kruskal(*values)
                which_test = 'Kruskal Wallis'
            else:
                t_stat, p = f_oneway(*testing_dat.values())
                which_test = 'ANOVA'
                var_homog = sorted([np.var(gv) for gv in testing_dat.values()])
                var_homog = max(var_homog)/min(var_homog)
                normal = [normaltest(gv).pvalue if len(gv) > 7 else 0 for gv in testing_dat.values()]
                normal = [p>.05 if p != 0 else 'too few samples' for p in normal]
                
            sig = '***' if p <.05 else 'N.S.'
            lbl = (f'{name}\n{config.SPACER}\nlevels:{testing_dat.keys()}:\n {which_test} '
                f't={t_stat:.3f}, p={p:.5f}\n --> {sig}\n\n')
            if parametric:
                lbl += f'Within group normal distr: {normal}\n'
                lbl += f'Variance homogeneity (max(group_var)/min(group_var)): {var_homog}\n'
            
            lbl = self.compute_singles(testing_dat.values(), testing_dat.keys(), 
                                    parametric, lbl)
            print('\n\n')
            print(lbl)
            with open(f'{dest_dir}/{name}_statistics.txt', "w") as text_file:
                text_file.write(lbl)
            
    def compute_singles(self, group_values, group_levels, parametric, lbl):
        mwu_pvalues = []
        for x, x_name in zip(group_values, group_levels):
            mwu_pvals = []
            for y, y_name in zip(group_values, group_levels):
                if len(np.unique(y))>1:
                    p = mannwhitneyu(x, y).pvalue if not parametric else ttest_ind(x, y).pvalue
                else:
                    p = 1
                mwu_pvals.append(p)
                lbl += f'{x_name}, {y_name}: p: {p:.5f}\t'

            mwu_pvalues.append(mwu_pvals)
            lbl += '\n'
        mwu_pvalues = np.array(mwu_pvalues)
        return lbl









    # # ===================== BEST DIRECTIONALITY STRCUTRES ======================
    # # compare structures 
    # def cs_axon_growth_direction(self, DIV_range=None, counted=False, plot_kwargs={}):
    #     all_growth_direcs = []
    #     for ss in self:
    #         dists = ss.get_distances(DIV_range=DIV_range)
    #         growth_dir = ss.growth_direction_start2end(dists)
            
    #         identifier = ss.identifier
    #         identifier = tuple(list(identifier) + ss.design_features)
    #         all_growth_direcs.append(growth_dir.rename(identifier))

    #     all_growth_direcs = pd.concat(all_growth_direcs, axis=1).T
    #     index_names = ['timelapse','design','CLSM_area']
    #     index_names.extend(config.DESIGN_FEATURE_NAMES)
    #     all_growth_direcs.index.rename(index_names, inplace=True)

    #     print(all_growth_direcs)
    #     exit()

    #     if not counted:
    #         fname = plot_growth_directions(all_growth_direcs, self.dir, **plot_kwargs)
    #     else:
    #         plot_counted_growth_directions(all_growth_direcs, self.dir, **plot_kwargs)

    # compare structures 
    
    # def cs_axon_destinations(self, DIV_range=None, save_single_plots=False, 
    #                          neg_comp_metric='directions', relative_naxons=False,
    #                          norm_to_sum=True,
    #                          plot_kwargs={}):
    #     print('\nPlotting axon destinations...', end='')
    #     destinations = []
    #     for ss in self:
    #         print(ss.name, end='...', flush=True)
    #         dists, DIV_mask = ss.get_distances(DIV_range=DIV_range, return_DIV_mask=True)
            
    #         kwargs = {'dists':dists, 'DIV_mask':DIV_mask, 'draw':save_single_plots}
    #         reached_target_axons, crossgrown_axons, stagnating_axons = ss.get_special_axons(kwargs)
    #         directions = ss.get_growth_directions({'dists':dists})

    #         if neg_comp_metric == 'cross_grown':
    #             metrics = np.array([len(reached_target_axons),
    #                                 len(crossgrown_axons),

    #             ], dtype=float)
    #         elif neg_comp_metric == 'directions':
    #             metrics = np.array([(directions<-config.MIN_GROWTH_DELTA).sum(),
    #                                 (directions>config.MIN_GROWTH_DELTA).sum(),

    #             ], dtype=float)
    #         # elif neg_comp_metric == 'wrong_dir_200':
    #         #     metrics = np.array([len(reached_target_axons),
    #         #                         (directions<-200).sum(),
    #         #     ], dtype=float)
            
    #         if norm_to_sum:
    #             metrics = np.append(metrics, (metrics[0]+1)/(metrics[1]+1) )
    #         else:
    #             if metrics.sum() < 3:
    #                 metrics = np.array([np.nan, np.nan])
    #             metrics = np.append(metrics, metrics[0]/(metrics[0]+(metrics[1])))
    #             metrics[2] = metrics[2]*10
            
    #         destin = pd.Series(metrics, index=['pos_metric','neg_metric', 'metric'], 
    #                            name=ss.identifier)

    #         add_design_features = True
            
    #         if add_design_features:
    #             identifier = ss.identifier
    #             identifier = tuple(list(identifier) + ss.design_features)
    #             destinations.append(destin.rename(identifier))
    #         else:
    #             destinations.append(destin)     


    #     destinations = pd.concat(destinations, axis=1).T
    #     index_names = ['timelapse','design','CLSM_area']
    #     if add_design_features:
    #         index_names.extend(config.DESIGN_FEATURE_NAMES)

    #     destinations.index.rename(index_names, inplace=True)
    #     destinations.to_csv(f'{self.dir}/results.csv')
    #     plot_axon_destinations(destinations, self.dir, neg_comp_metric, **plot_kwargs)