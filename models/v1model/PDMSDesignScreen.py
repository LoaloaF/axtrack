import pickle
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from scipy.stats import linregress

from config import OUTPUT_DIR

import matplotlib.pyplot as plt

class PDMSDesignScreen(object):
    def __init__(self, timelapse, directory, axon_detections, 
                 cache_target_distances='from'):
        self.sizet = timelapse.sizet
        self.dir = directory
        
        # when comparing with ground truth
        self._set_hacky_gt_properties(timelapse)

        axon_detections.compute_target_distances(cache=cache_target_distances)
        self.distances = axon_detections.to_target_dists
        self.all_dets = axon_detections.get_all_IDassiged_dets()
        
        # hyperparamters for metrics
        self.trans_det_distance = 4
        self.trans_det_prominence = 200
        self.trans_det_threshold = 4
        self.trans_det_width = (1,24)
        self.stagnating_axon_std_thr = 44
        self.crossgrowth_speed_thr = 40


    def __len__(self):
        return self.sizet

    def compute_metrics(self, draw=False):

        # get the change in distance to outputchannel between each t, t+1
        pred_gr_directions = self.growth_direction(self.distances)
        pred_axon_gr_directions = self.growth_direction_start2end(self.distances)
        
        if draw:
            plt.figure()
            plt.hist(pred_gr_directions, alpha=.5, bins=40, density=True, range=(-400,400), label='pred')
            plt.legend()
            
            plt.figure()
            plt.hist(pred_axon_gr_directions, alpha=.5, bins=40, range=(-2000,2000), density=True, label='pred')
            plt.legend()
        stagnating_axons, good_trans_axons, bad_trans_axons = self.detect_transitions(self.distances, draw=draw)
        reached_target, crossgrown = self.detect_axons_final_location(self.distances, self.all_dets, draw=draw)
        
        return metrics


    def detect_axons_final_location(self, dists, dets, draw=False):
        if draw:
            fig, axes = plt.subplots(10,10, figsize=(18,12), sharex=True, sharey=True)
            fig.subplots_adjust(wspace=0, hspace=0, top=.97, left=.03, bottom=.03, right=1)
            fig2, ax2 = plt.subplots()
            mask = self.goal_mask | self.start_mask

        axons_reached_target, axons_crossgrown = [], []
        for i in range(dists.shape[0]):
            axon_det = dets.iloc[i].dropna()
            dets_coos = axon_det.loc[slice(None),['anchor_y','anchor_x']].unstack().astype(int)
            print(dets_coos)
            x_coos = list(dets_coos.values[:,0])
            y_coos = list(dets_coos.values[:,1])

            reached_goal = self.goal_mask[y_coos, x_coos]
            if reached_goal.any():
                axons_reached_target.append(dists.index[i])

            axon_dists = dists.iloc[i].dropna()
            last_values = axon_dists.iloc[-6:]
            last_growth_trend = linregress(last_values.index.astype(int), last_values).slope
            crossgrowth = self.start_mask[y_coos[-6:], x_coos[-6:]]
            if last_growth_trend >self.crossgrowth_speed_thr and crossgrowth.any():
                axons_crossgrown.append(dists.index[i])
            
            if draw:            
                ax = axes.flatten()[i]
                x = axon_dists.index.astype(int).values
                ax.scatter(x, axon_dists, s=4)
                
                if last_growth_trend >self.crossgrowth_speed_thr and crossgrowth.any():
                    ax2.scatter(x_coos[-6:], y_coos[-6:], s=5)
                    ax.set_title(round(last_growth_trend, 2))

                if reached_goal.any():
                    ax2.scatter(x_coos, y_coos, s=5)
                    ax.set_title('GOAL!')
                
                f = np.poly1d(np.polyfit(x, axon_dists, 5))
                x_new = np.linspace(x[0], x[-1], 50)
                ax.plot(x_new, f(x_new), color='purple', linestyle='dashed', alpha=.7)

        if draw:
            ax2.imshow(mask, cmap='gray')
            plt.show()
        return axons_reached_target, axons_crossgrown

    def n_axons(self):
        return self.distances.shape[0]
    
    def growth_direction(self, dists):
        ddt_growth = [dists.iloc[:,t+1] - dists.iloc[:,t] for t in range(len(self)-1)]
        return [g for ddt_g in ddt_growth for g in ddt_g if not pd.isna([g])]

    def growth_direction_start2end(self, dists):
        start_dist = dists.groupby(level=0).apply(lambda row: row.dropna(axis=1).iloc[:,0])
        end_dist = dists.groupby(level=0).apply(lambda row: row.dropna(axis=1).iloc[:,-1])

        distr = end_dist.values-start_dist.values
        return distr
    
    def detect_transitions(self, dists, draw=False):
        dists = self._subtract_initial_dist(dists)

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

    def _subtract_initial_dist(self, dists):
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
        