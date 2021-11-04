import pickle
import os

import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from scipy.stats import linregress

from config import OUTPUT_DIR
import config

import matplotlib.pyplot as plt

class StructureScreen(object):
    def __init__(self, timelapse, directory, axon_detections, 
                 cache_target_distances='from'):
        self.name = timelapse.name
        
        self.identifier = (self.name[:4],int(self.name[6:8]),self.name[9:13])
        self.mask = timelapse.mask
        self.dt = timelapse.dt

        self.notes = timelapse.notes
        self.sizet = timelapse.sizet
        self.pixelsize = timelapse.pixelsize
        self.goal_mask = timelapse.goal_mask
        self.start_mask = timelapse.start_mask
        self.incubation_time = timelapse.incubation_time
        self.structure_outputchannel_coo = timelapse.structure_outputchannel_coo
        
        self.mask_size = self._compute_mask_size()
        self.design_features = config.DESIGN_FEATURES[self.identifier[1]]
        
        self.dir = directory
        os.makedirs(self.dir, exist_ok=True)
        
        # when comparing with ground truth
        # self._set_hacky_gt_properties(timelapse)

        axon_detections.compute_target_distances(cache=cache_target_distances)
        self.distances = axon_detections.to_target_dists
        self.all_dets = axon_detections.get_all_IDassiged_dets()
        self.goalmask2target_maxdist = self._compute_goalmask2target_maxdist()
        
        # set to all IDs in the beginning
        self.set_axon_subset(self.distances.index)
        
        # hyperparamters for metrics
        self.trans_det_distance = 4
        self.trans_det_prominence = 200
        self.trans_det_threshold = 4
        self.trans_det_width = (1,24)
        self.stagnating_axon_std_thr = 44

        self.crossgrowth_speed_thr = 250
        self.last_growth_trend_n = 10

    def __len__(self):
        return self.sizet

    def set_axon_subset(self, which_axons):
        self.axon_subset = which_axons

    def get_id_lifetime(self):
        return self.all_dets.loc[self.axon_subset, (slice(None), 'conf')].fillna(False).astype(bool).values
    
    def get_distances(self, px2um=True, timepoints2minutes=True, DIV_range=None,
                      add_incubation_time=True, interpolate_missing_det=True,
                      timeopints2DIV=True):
        dists = self.distances.loc[self.axon_subset]
        if interpolate_missing_det:
            dists = dists.interpolate(axis=1, limit_area='inside')
        if px2um:
            dists *= self.pixelsize
        if timepoints2minutes:
            dists.columns = dists.columns *self.dt
        if add_incubation_time:
            dists.columns = dists.columns + self.incubation_time
            if DIV_range:
                tps = dists.columns.values
                in_range = (tps > DIV_range[0]*24*60) & (tps < DIV_range[1]*24*60)
                dists = dists.T[in_range].T.dropna(how='all')
            if timeopints2DIV:
                dists.columns = dists.columns /(60*24)
        return dists
    
    def _compute_goalmask2target_maxdist(self):
        yx = np.stack(np.where(self.goal_mask)).T
        # euclidean distance from every point in the mask to the target coo
        # the one furthest away is the distance to the target where an axon made it
        return np.max(np.sqrt((yx-self.structure_outputchannel_coo)**2))
    
    def _compute_mask_size(self):
        return self.mask[0].getnnz() * self.pixelsize
    
    # for lazyness - cache screen again to not need this thing below
    def hacky_adhoc_param_setter(self):
        # self.pixelsize = 0.61
        # self.dt = 31
        # self.identifier = (self.name[:4],int(self.name[6:8]),self.name[9:13])
        # self.mask_size = 234000
        # self.metric = pd.Series([self.n_axons()], index=['n_axons'], name=self.identifier)
        # self.incubation_time = 52*60
        # self.crossgrowth_speed_thr = 250
        # self.last_growth_trend_n = 10

        from glob import glob
        goal_mask_file = glob(f'{config.RAW_INFERENCE_DATA_DIR}*{self.name}*goal.npy')
        self.goal_mask = np.load(goal_mask_file[0])
        self.goal_mask = np.pad(self.goal_mask, 50, mode='constant')
        
        start_mask_file = glob(f'{config.RAW_INFERENCE_DATA_DIR}*{self.name}*start.npy')
        self.start_mask = np.load(start_mask_file[0])
        self.start_mask = np.pad(self.start_mask, 50, mode='constant')

        self.design_features = config.DESIGN_FEATURES[self.identifier[1]]



        pass

    def compute_metrics(self, draw=False):

        # get the change in distance to outputchannel between each t, t+1
        pred_gr_directions = self.growth_direction(self.distances)
        pred_axon_gr_directions = self.growth_direction_start2end(self.distances)

        pred_gr_directions *= self.pixelsize
        pred_axon_gr_directions *= self.pixelsize
        
        if draw:
            plt.figure()
            plt.hist(pred_gr_directions, alpha=.5, bins=80, density=True, range=(-400,400), label='pred')
            plt.legend()
            
            plt.figure()
            plt.hist(pred_axon_gr_directions, alpha=.5, bins=80, range=(-2000,2000), density=True, label='pred')
            plt.legend()
            plt.show()
        stagnating_axons, good_trans_axons, bad_trans_axons = self.detect_transitions(self.distances, draw=draw)
        reached_target_axons, crossgrown_axons = self.detect_axons_final_location(self.distances, self.all_dets, draw=draw)

        metrics = pred_gr_directions, pred_axon_gr_directions, stagnating_axons, good_trans_axons, bad_trans_axons, reached_target_axons, crossgrown_axons
        return metrics
        


    def detect_axons_final_location(self, dists, crossgrowth_params=None, draw=True):
        if draw:
            # one panel shows 100 axons
            panels_needed = dists.shape[0]//100 +1
            panels = [plt.subplots(10,10, figsize=(18,12), 
                                  sharex=True, sharey=True) for _ in range(panels_needed)]
            [fig.subplots_adjust(wspace=0, hspace=0, top=.96, left=.06, bottom=.06, right=.99) 
             for fig, _ in panels]
            panels[0][0].suptitle(self.name)
            mask =  self.start_mask
            # mask[self.goal_mask] = 2
            mask_fig, mask_ax = plt.subplots(figsize=(mask.shape[1]/350, mask.shape[0]/350))

        axons_reached_target, axons_crossgrown = [], []
        for i in range(dists.shape[0]):
            axon_det = self.all_dets.iloc[i].dropna()
            x_coos = axon_det.loc[slice(None),['anchor_x']].astype(int).values
            y_coos = axon_det.loc[slice(None),['anchor_y']].astype(int).values

            reached_goal = self.goal_mask[y_coos, x_coos]
            if reached_goal.any():
                axons_reached_target.append(dists.index[i])

            if crossgrowth_params is not None:
                n = crossgrowth_params[0]
                min_speed = crossgrowth_params[1]
            else:
                n = self.last_growth_trend_n
                min_speed = self.crossgrowth_speed_thr


            axon_dists = dists.iloc[i].dropna()
            last_values = axon_dists.iloc[-n:]
            last_growth_trend = linregress(last_values.index, last_values).slope
            
            crossgrowth = self.start_mask[y_coos[-n:], x_coos[-n:]]
            if last_growth_trend >min_speed and crossgrowth.any():
                axons_crossgrown.append(dists.index[i])
            
            if draw:
                panel_i, axes_i = divmod(i, 100)
                ax = panels[panel_i][1].flatten()[axes_i]
                ax.set_xlim(dists.columns.min(), dists.columns.max())
                if not axes_i:
                    ax.text(0.5, 0.02, 'DIV', fontsize=config.FONTS, transform=panels[panel_i][0].transFigure)
                    ax.text(0.02, 0.5, 'distance to outputchannel', rotation=90, fontsize=config.FONTS, transform=panels[panel_i][0].transFigure)
                x = axon_dists.index.values
                # ax.scatter(x, axon_dists, s=1)
                lbl = f'{axon_det.name}, l={len(axon_det.index.unique(0))} gt:{last_growth_trend:.2f}'
                
                if last_growth_trend >min_speed and crossgrowth.any():
                    mask_ax.scatter(x_coos, y_coos, s=30, marker='x', linewidth=.5)
                    [ax.spines[w].set_color(config.ORANGE) for w in ax.spines]
                    [ax.spines[w].set_linewidth(3) for w in ax.spines]
                    lbl += ' - CROSS-GROWTH!'
                    # [mask_ax.text(x, y, i, fontsize=8, color='gray') for i, (x, y) in enumerate(zip(x_coos, y_coos))]
                
                # if axon_det.name in ('Axon_146', 'Axon_081'):
                #     mask_ax.scatter(x_coos, y_coos, s=10, marker='.')

                if reached_goal.any():
                    mask_ax.scatter(x_coos, y_coos, s=40, marker='+', linewidth=.5)
                    [ax.spines[w].set_color(config.GREEN) for w in ax.spines]
                    [ax.spines[w].set_linewidth(3) for w in ax.spines]
                    lbl += ' - GOAL!'
                
                ax.text(.05,.05, lbl, fontsize=8, transform=ax.transAxes)

        if draw:
            mask_ax.imshow(mask, cmap='gray')
            for i, (fig, _) in enumerate(panels):
                fig.savefig(f'{self.dir}/single_destianations_{i}.png')
            mask_fig.savefig(f'{self.dir}/destinations.png')
        return axons_reached_target, axons_crossgrown

    def n_axons(self):
        return self.distances.shape[0]
    
    def get_delta_growth(self, dists, absolute=True):
        delta_growth = [dists.iloc[:,t+1] - dists.iloc[:,t].values for t in range(dists.shape[1]-1)]
        delta_growth = pd.concat(delta_growth, axis=1)
        if absolute:
            delta_growth = delta_growth.abs()
        return delta_growth

    def get_growth_speed(self, dists, absolute=True, average=True, per_time_unit='d'):
        delta_growth = self.get_delta_growth(dists, absolute)
        if per_time_unit == 'd':
            ddt_growth = delta_growth/ (self.dt/(60*24))
        elif per_time_unit == 'h':
            ddt_growth = delta_growth/ (self.dt/(60))
        if average:
            ddt_growth = ddt_growth.mean(1)
        return ddt_growth



    def growth_direction_start2end(self, dists):
        start_dist = dists.groupby(level=0).apply(lambda row: row.dropna(axis=1).iloc[:,0])
        end_dist = dists.groupby(level=0).apply(lambda row: row.dropna(axis=1).iloc[:,-1])
        
        # growth_direction = dists.notna().sum(1).rename('id_lifetime').to_frame()
        # growth_direction = (growth_direction*self.dt) /60
        growth_direction = pd.Series(end_dist.values-start_dist.values, 
                                        index=dists.index, name='dist_change')
        return growth_direction
    
    def detect_transitions(self, dists, draw=False):
        dists = self.subtract_initial_dist(dists)

        if draw:
            fig, axes = plt.subplots(10,10, figsize=(18,12), sharex=True, sharey=True)
            fig.subplots_adjust(wspace=0, hspace=0, top=.97, left=.03, bottom=.03, right=1)

        stagnating_axons = []
        good_trans_axons = []
        bad_trans_axons = []
        for i, (axon_id, row) in enumerate(dists.iterrows()):
            row = row.dropna()
            good_idxs, good_p = find_peaks(row, 
                                           distance = self.trans_det_distance, 
                                           prominence = self.trans_det_prominence, 
                                           threshold = self.trans_det_threshold, 
                                           width = self.trans_det_width)
            bad_idxs, bad_p = find_peaks(row*-1, 
                                         distance = self.trans_det_distance, 
                                         prominence = self.trans_det_prominence, 
                                         threshold = self.trans_det_threshold, 
                                         width = self.trans_det_width)
            std = np.std(row.iloc[-6:])
            
            if len(good_idxs):
                good_trans_axons.append(axon_id)
            if len(bad_idxs):
                bad_trans_axons.append(axon_id)
            if std <self.stagnating_axon_std_thr:
                stagnating_axons.append(axon_id)

            
            if draw:
                ax = axes.flatten()[i]
                x = np.array([int(x) for x in row.index.values])
                ax.scatter(x, row, s=4)
                
                if std<self.stagnating_axon_std_thr:
                    ax.scatter(x[-6:], row[-6:], color='black', s=4, facecolor='none')

                f = np.poly1d(np.polyfit(x, row, 5))
                x_new = np.linspace(x[0], x[-1], 50)
                ax.plot(x_new, f(x_new), color='purple', linestyle='dashed', alpha=.7)

                if len(good_idxs):
                    print(f'\green: {i}\n{good_p}')
                    ax.vlines(x[good_idxs], -1000,1000, color='green')
                
                if len(bad_idxs):
                    print(f'\nred: {i}\n{bad_p}')
                    ax.vlines(x[bad_idxs], -1000,1000, color='red')
        if draw:
            plt.show()
        return stagnating_axons, good_trans_axons, bad_trans_axons

    def subtract_initial_dist(self, dists):
        # get the initial timepoints distance
        init_dist = dists.groupby(level=0).apply(lambda row: row.dropna(axis=1).iloc[:,0])
        # subtract that distance
        return dists.apply(lambda col: col-init_dist.values)








    # for comparing to ground truth. Used to optimize paramters for MCF and here
    def _set_hacky_gt_properties(self, timelapse):
        # given a pad of 300 on left and right
        goal_mask = np.load(f'{OUTPUT_DIR}/mask_goal.npy')
        pad = np.zeros((goal_mask.shape[0],300))
        self.goal_mask = np.concatenate([pad, goal_mask, pad], axis=1).astype(bool)
        start_mask = np.load(f'{OUTPUT_DIR}/mask_start.npy')
        self.start_mask = np.concatenate([pad, start_mask, pad], axis=1).astype(bool)

        # this is the target of the training data
        timelapse.structure_outputchannel_coo = (550, 3000)

    def _get_gt_hacky(self, data_name):
        dets = pd.read_csv(f'{OUTPUT_DIR}/{data_name}_ax_detections_gt.csv', 
                           index_col=0, header=[0,1]).sort_index()
        dets.columns =  pd.MultiIndex.from_tuples([(int(t),which) for t, which in dets.columns])
        dists = pd.read_csv(f'{OUTPUT_DIR}/{data_name}_dists_gt.csv', 
                            index_col=0).sort_index()
        dists.columns = dists.columns.astype(int)
        return dets, dists
    
    def compare_to_gt_hacky(self, data_name):
        gt_detections, gt_distances = self._get_gt_hacky(data_name)
        draw=False

        # get the change in distance to outputchannel between each t, t+1
        pred_gr_directions = self.growth_direction(self.distances)
        pred_distr = np.histogram(pred_gr_directions, bins=40, density=True, 
                                  range=(-400,400))[0]
        gt_gr_directions = self.growth_direction(gt_distances)
        gt_distr = np.histogram(gt_gr_directions, bins=40, density=True, 
                                range=(-400,400))[0]
        hist_intersect_growth_direction = (np.minimum(pred_distr, gt_distr)*(800/40)).sum()
        if draw:
            plt.figure()
            plt.hist(pred_gr_directions, alpha=.5, bins=40, density=True, range=(-400,400), label='pred')
            plt.hist(gt_gr_directions, alpha=.5, bins=40, density=True, range=(-400,400), label='gt')
            plt.legend()

        # get the change in distance to outputchannel between each t0, t-1
        pred_axon_gr_directions = self.growth_direction_start2end(self.distances)
        pred_distr = np.histogram(pred_axon_gr_directions, bins=40, density=True, 
                                  range=(-2000,2000))[0]
        gt_axon_gr_directions = self.growth_direction_start2end(gt_distances)
        gt_distr = np.histogram(gt_gr_directions, bins=40, density=True, 
                                range=(-2000,2000))[0]
        hist_intersect_axon_growth_direction = (np.minimum(pred_distr, gt_distr)*(4000/40)).sum()
        if draw:
            plt.figure()
            plt.hist(pred_axon_gr_directions, alpha=.5, bins=40, range=(-2000,2000), density=True, label='pred')
            plt.hist(gt_axon_gr_directions, alpha=.5, bins=40, range=(-2000,2000), density=True, label='gt')
            plt.legend()


        stagnating_axons, good_trans_axons, bad_trans_axons = self.detect_transitions(self.distances, draw=draw)
        gt_stagnating_axons, gt_good_trans_axons, gt_bad_trans_axons = self.detect_transitions(gt_distances, draw=draw)
        stagnating_axons_err = (len(stagnating_axons)-len(gt_stagnating_axons))**2
        good_trans_axons_err = (len(good_trans_axons)-len(gt_good_trans_axons))**2
        bad_trans_axons_err = (len(bad_trans_axons)-len(gt_bad_trans_axons))**2
        
        reached_target, crossgrown = self.detect_axons_final_location(self.distances, self.all_dets, draw=draw)
        gt_reached_target, gt_crossgrown = self.detect_axons_final_location(gt_distances, gt_detections, draw=draw)
        reached_target_err = (len(reached_target)-len(gt_reached_target))**2
        crossgrown_err = (len(crossgrown)-len(gt_crossgrown))**2

        metrics = {'hist_intersect_growth_direction': hist_intersect_growth_direction,
                   'hist_intersect_axon_growth_direction': hist_intersect_axon_growth_direction,
                   'stagnating_axons_err ': stagnating_axons_err ,
                   'good_trans_axons_err': good_trans_axons_err,
                   'bad_trans_axons_err': bad_trans_axons_err,
                   'reached_target_err': reached_target_err,
                   'crossgrown_err': crossgrown_err,}
        return metrics
        