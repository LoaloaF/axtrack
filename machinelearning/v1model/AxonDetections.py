import os
import pickle
from copy import deepcopy

from concurrent.futures import ThreadPoolExecutor
from itertools import repeat

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

import config
import cv2
from scipy import sparse
from sklearn import metrics

import pyastar2d
from libmot.data_association  import MinCostFlowTracker
import motmetrics as mm

class AxonDetections(object):
    """
    Bundles model inference output to data input. All conversions from model
    output to usable detections are implemented in this class.
    """
    def __init__(self, model, dataset, parameters, directory, 
                 timepoint_subset=None):
        """
        Create an AxonDetection object. 

        Attributes
        ----------
        model: YOLO_AXTrack
            The detector model to use for inference.
            
        dataset: Timelapse
            The input data.
        
        model: parameters
            The parameters used for training the model. Includes also parameters
            relevent for this class.
        
        directory: str
            The directory where cached data pieces are stored. No directory is 
            created/ used when None is passed. 
        
        timepoint_subset: iterable
            The frames to use. If None, use all timepoints. Defaults to None.

        """
        self.model = model
        self.dataset = dataset 
        self.dir = directory
        if self.dir:
            os.makedirs(self.dir, exist_ok=True)

        self.timepoint_subset = timepoint_subset if timepoint_subset is not None else range(self.dataset.sizet)
        
        # basic parameters
        self.device = parameters['DEVICE']
        self.Sx = parameters['SX']
        self.Sy = parameters['SY']
        self.tilesize = parameters['TILESIZE']
        
        # parameters for selecting detections
        self.nms_min_dist = parameters.get('NON_MAX_SUPRESSION_DIST')
        self.conf_thr = parameters['BBOX_THRESHOLD']
        self.all_conf_thrs = np.sort(np.append(np.arange(0.55, 1, .04), self.conf_thr)).round(2)
        self.max_px_assoc_dist = 500
        self.axon_box_size = 70     # only for visualization
        self.labelled = False if  hasattr(dataset, 'start_mask') else True

        # min cost flow hyperparamters 
        self.MCF_edge_cost_thr =  0.7
        self.MCF_entry_exit_cost = 2
        self.MCF_miss_rate =  0.6
        self.MCF_max_num_misses = 1
        self.MCF_min_flow = 5
        self.MCF_max_flow = 450
        self.MCF_max_conf_cost = 4.6
        self.MCF_vis_sim_weight = 0
        self.MCF_conf_capping_method = 'scale_to_max'
        self.MCF_min_ID_lifetime = 5

    def __len__(self):
        """
        The length of axon detections. Here, the number of frames.
        """
        return len(self.timepoint_subset)

    def detect_dataset(self, cache=None):
        """
        Use the model to detect growth cones. yolo_targets, pandas_tiled_dets,
            detections are created. Called from outside this class.

        Attributes
        ----------
        cache: str
            Whether to save the computed data to a file or use precomputed data
            or don't do any caching. Options: {'to', 'from', None}. Defaults to
            None, no caching.
        """
        self.dataset.construct_tiles(self.device, force_no_transformation=True)

        if cache == 'from':
            self.yolo_targets = self.from_cache('yolo_targets')
            self.pandas_tiled_dets = self.from_cache('pandas_tiled_dets')
            self.detections = self.from_cache('detections')

        else:
            self.yolo_targets = []
            self.pandas_tiled_dets = []
            self.detections = []
            print(f'Detecting axons in {self.dataset.name} data: ', end='\n')
            for t in self.timepoint_subset:
                print(f'frame {t}/{(len(self)-1)}', end='...', flush=True)
                # get a stack of tiles that makes up a frame
                # batch dim of X corresbonds to all tiles in frame at time t
                X, yolo_target = self.dataset.get_frametiles_stack(t, self.device)
                
                # use the model to detect axons, output here follows target.shape
                yolo_det = self.model.detect_axons(X)

                # convert the tensor to a tile-wise list of pd.DataFrames
                # conf threshold is very low here
                pandas_tiled_det = self._yolo_Y2pandas_det(yolo_det, conf_thr=self.all_conf_thrs.min())
                
                # stitch the list of pandas tiles to one pd.DataFrame
                pandas_frame_det, _ = self.dataset.stitch_tiles(pandas_tiled_det, reset_index=True)

                # set conf of strongly overlapping detections to 0, unique index needed!
                pandas_frame_det_nms = self._non_max_supression(pandas_frame_det)
            
                # save intermediate detection formats
                self.yolo_targets.append(yolo_target)
                self.pandas_tiled_dets.append(pandas_tiled_det)
                self.detections.append(pandas_frame_det_nms)
            print('Done.\n', flush=True)
        
        if cache == 'to':
            self.to_cache('yolo_targets', self.yolo_targets)
            self.to_cache('pandas_tiled_dets', self.pandas_tiled_dets)
            self.to_cache('detections', self.detections)
    
    def from_cache(self, which):
        """
        Load a pickled piece of data. Previously computed data, e.g. detections,
        are read as files and returned.

        Attributes
        ----------
        which: str
            Which data to get. Options: {'yolo_targets', 'pandas_tiled_dets',
            'detections', } 
        """
        fname = f'{self.dataset.name}_{which}.pkl'
        print(f'Getting from cache: {fname}', flush=True)
        cache_fname = f'{self.dir}/{fname}'
        with open(cache_fname, 'rb') as file:
            dat = pickle.load(file)
        return dat
        
    def to_cache(self, which, dat):
        """
        Save a computed piece of data as a pickle file.

        Attributes
        ----------
        which: str
            Which data to get. Options: {'yolo_targets', 'pandas_tiled_dets',
            'detections', }.

        dat:
            The data to be pickled. 
        """
        fname = f'{self.dataset.name}_{which}.pkl'
        print(f'Saving in cache: {fname}', flush=True)
        cache_fname = f'{self.dir}/{fname}'
        with open(cache_fname, 'wb') as file:
            pickle.dump(dat, file)

    def _yolo_Y2pandas_det(self, yolo_Y, conf_thr):
        """
        Convert the model output (YOLO format) to a tile-wise list of
        pd.DataFrames. pd.Dataframes have axon names as indices (eg. Axon_042)
        and three columns [conf, anchor_x, anchor_y].

        Attributes
        ----------
        yolo_Y: torch.Tensor
            The output of the detection model. Shape: [ntiles_framet, Sy, Sx, 3]

        conf_thr:
            The minimum detection confidence required to be included.
        """
        def _yolo_coo2tile_coo(yolo_Y):
            # mask that is false for all pos label entries, needed for true yolo_Y
            noanchor_mask = (yolo_Y == 0).all(-1)

            # these rescale vectors scale depending on the gridbox the label is in
            rescale_x = torch.arange(self.Sx).reshape(1, self.Sx, 1, 1).to(self.device)
            rescale_y = torch.arange(self.Sy).reshape(1, 1, self.Sy, 1).to(self.device)
                
            # this is 200 IQ: we add the gridbox (0-S) to the within-gridbox
            # coordinate (0-1). The coordiante is now represented in terms of
            # gridboxsize (here 64)
            yolo_Y[..., 1::3] = (yolo_Y[..., 1::3] + rescale_x)
            yolo_Y[..., 2::3] = (yolo_Y[..., 2::3] + rescale_y)
            # now we just muplitply by gridboxsize to scale to image coo
            yolo_Y[..., 1::3] = (yolo_Y[..., 1::3] * self.tilesize/self.Sx).round()
            yolo_Y[..., 2::3] = (yolo_Y[..., 2::3] * self.tilesize/self.Sy).round()
            # the rescale vector also adds to the 0-elements (unwanted). Mask them here
            yolo_Y[noanchor_mask] = 0
            return yolo_Y

        def _filter_yolo_det(yolo_Y, thr):
            # convert the target tensor to shape (batchsize, self.Sx*self.Sy, 3)
            batch_size, labelsize = yolo_Y.shape[0], yolo_Y.shape[-1]
            yolo_Y = yolo_Y.reshape((batch_size, -1, labelsize))
            # split batch dim (tiles) into list because of variable sizes 
            det_list = list(torch.tensor_split(yolo_Y, batch_size))

            # filter each example
            return [tile_det[tile_det[:,:,0] >= thr] for tile_det in det_list]

        def _torch2pandas(tile_det):
            # for true tile detection, use the 4. column (IDs) to define detection names 
            if tile_det.shape[1]>3:
                axons_ids = tile_det[:,3].to(int)  
            # for predicted detection, just count up new names 
            else:
                axons_ids = range(tile_det.shape[0])
            
            axon_lbls = [f'Axon_{i:0>3}' for i in axons_ids]
            cols = 'conf', 'anchor_x', 'anchor_y'
            tile_det = tile_det[:,:3].to('cpu').numpy()
            
            pd_tile_det = pd.DataFrame(tile_det, index=axon_lbls, columns=cols)
            return pd_tile_det.convert_dtypes().sort_values('conf')

        # convert the YOLO coordinates (0-1) to image coordinates (tilesize x tilesize) (inplace function)
        yolo_Y_imgcoo = _yolo_coo2tile_coo(yolo_Y.clone())

        # filter the (batch, self.Sx, self.Sy, 3) output into a list of length batchsize,
        # where the list elements are tensors of shape (#_boxes_above_thr x 3)
        # each element in the list represents a frame/tile
        det_list = _filter_yolo_det(yolo_Y_imgcoo, thr=conf_thr)
        # convert each frame from tensor to pandas
        for i in range(len(det_list)):
            det_list[i] = _torch2pandas(det_list[i])
        return det_list

    def _non_max_supression(self, pd_frame_det):
        """
        Drop strongly overlapping detections. Returns a pd.DataFrame with the
        non-maximum confidence detections suppressed. 

        Attributes
        ----------
        pd_frame_det: pd.Dataframe
            Detection of one frame. 
        """
        # order by confidence
        pd_frame_det = pd_frame_det.sort_values('conf', ascending=False)
        i = 0
        # get most confident anchor first
        while i < pd_frame_det.shape[0]:
            xs, ys = pd_frame_det[['anchor_x', 'anchor_y']].values.T
            x,y = pd_frame_det.iloc[i, 1:]

            # compute distances of that anchor to all others
            dists = np.sqrt(((xs-x)**2 + (ys-y)**2).astype(int))
            # check where its < thr, skip the first one (dist. with itself = 0)
            non_max_indices = np.where(dists<self.nms_min_dist)[0][1:]
            # drop those anchors
            pd_frame_det.drop(pd_frame_det.index[non_max_indices], inplace=True)
            i += 1

        axon_lbls = [f'Axon_{i:0>3}' for i in range(pd_frame_det.shape[0])]
        pd_frame_det.index = axon_lbls
        return pd_frame_det
    



    def get_frame_dets(self, which_dets, t, unstitched=False, return_ilocs=False):
        """
        Get detections of one frame as a pd.DataFrame. pd.Dataframe has axon
        names as indices (eg. Axon_042) and three columns [conf, anchor_x,
        anchor_y]. Requires to have previously called detect_dataset().

        Attributes
        ----------
        which_dets: str
            Which set of detections to get. Options: {'all', 'confident',
            'FP_FN', 'IDed'}. 'all' will return detections above a very low
            threshold set by the lowest value of self.all_conf_thrs, here 0.55.
            'confident' will return the detections above the predefined
            threshold set in parameters['BBOX_THRESHOLD']. 'FP_FN' will return
            two pd.DataFrames, instead of just one. These detections only
            include false-positives (FP) and false-negative (FN) detections.
            'IDed' will return detections with assigned identity. Requires to
            have previously run assign_ids()

        t: int
            Which frame to return the detections of.

        unstitched: bool
            Only implemented/ considered when which_dets == 'confident'. Return
            tile-wise list of pd.DataFrames (unstitched). Defaults to False.

        return_ilocs: bool
            Only implemented/ considered when which_dets == 'IDed'. Return the 
            ilocs of IDed detections in set of which_dets == 'all' detections.
        """
        # filter the detection set at this frame to confidence threshold 
        
        assert hasattr(self, 'detections'), "Run .detect_dataset() first!"
        if which_dets == 'all':
            return self.detections[t]
        
        elif which_dets == 'confident':
            if not unstitched:
                return self.detections[t][self.detections[t].conf>self.conf_thr]
            return [d[d.conf>self.conf_thr] for d in self.pandas_tiled_dets[t]]
        
        elif which_dets == 'FP_FN':
            assert self.labelled, 'Cannot compute FP FN on unlabelled data'
            dets = self.get_frame_dets('confident', t).copy()
            _, true_dets = self.get_frame_and_truedets(t)
            FP_mask, FN_mask = self.compute_TP_FP_FN('confident', t, 
                                                     return_FP_FN_mask=True)
            FP_dets = dets[FP_mask]
            FN_dets = true_dets[FN_mask]
            return FP_dets, FN_dets

        elif which_dets == 'IDed':
            assert hasattr(self, 'IDed_detections'), "Run .assign_IDs() first!"
            det = self.detections[t]
            if t not in self.IDed_detections.index:
                emtpy_IDed_det = pd.DataFrame([], columns=['conf','anchor_x','anchor_y'])
                if not return_ilocs:
                    return emtpy_IDed_det
                return emtpy_IDed_det, []
            IDed_det = self.IDed_detections.loc[t, ['X','Y']]
            half_boxs = self.axon_box_size/2
            
            valid_det = []
            ilocs = []
            # iter old detections (longer or same size as IDed detections)
            for i, (_, (conf, anchor_x, anchor_y)) in enumerate(det.iterrows()):
                # get the assigned ID by masking for matching box coordinates (hacky)
                coo_mask = (IDed_det.values==(anchor_x-half_boxs, anchor_y-half_boxs)).all(1)
                ID = IDed_det.index[coo_mask]
                # if the box coordinates were still in the IDed detections add them
                if not ID.empty:
                    ilocs.append(i)
                    valid_det.append(pd.Series([conf, anchor_x, anchor_y], 
                                                name=f'Axon_{ID[0]:0>3}', 
                                                index=('conf', 'anchor_x', 'anchor_y')))
            IDed_det = pd.concat(valid_det, axis=1).T
            if not return_ilocs:
                return IDed_det
            return IDed_det, ilocs

    # def get_confident_det(self, t, unstitched=False):
    #     # filter the detection set at this frame to confidence threshold 
    #     if not unstitched:
    #         return self.detections[t][self.detections[t].conf>self.conf_thr]
    #     return [d[d.conf>self.conf_thr] for d in self.pandas_tiled_dets[t]]

    # def get_FP_FN_detections(self, t):
    #     # FP_dets = self.get_confident_det(t).copy()
    #     FP_dets = self.get_frame_dets('confident', t).copy()
    #     FN_dets = self.get_frame_and_truedets(t)[1]

    #     FP_mask, FN_mask = self.compute_TP_FP_FN('confident', t, return_FP_FN_mask=True)
    #     FP_dets = FP_dets[FP_mask]
    #     FN_dets = FN_dets[FN_mask]
    #     return FP_dets, FN_dets

    def get_frame_and_truedets(self, t, unstitched=False):
        """
        Get a drawable frame and the matching groundtruth detections. Returns a
        np.array of shape [channels, img_sizey, img_sizex] and pd.DataFrame of
        true detections. Channels is 1 if the detector is not set to use
        precomputed motion (default: 1 channel)

        Attributes
        ----------
        t: int
            Which frame and matching detections to return. 
        
        unstitched: bool
            Return the unstitched versions (tile-wise) list of img + detections. 
        """
        img_tiled, _ = self.dataset.get_frametiles_stack(t, self.device)
        pd_tiled_true_det = self._yolo_Y2pandas_det(self.yolo_targets[t], conf_thr=1)
        pd_frame_true_det, img = self.dataset.stitch_tiles(pd_tiled_true_det, img_tiled)

        if not unstitched:
            return img, pd_frame_true_det
        return img_tiled, pd_tiled_true_det

    def get_detection_metrics(self, which_dets, t, return_all_conf_thrs=False):
        """
        Get the detection metrics of one-frame-detections: precision, recall,
        and F1 score. Returns either just the metrics of the preset threshold or
        metrics for all thresholds. Called outside the class after training.

        Attributes
        ----------
        which_dets:
            Which set of detections to use. Options: {'all', 'confident',
            'IDed'}

        t: int
            Detection metric of which frame to return.
        
        return_all_conf_thrs: bool
            Whether to return metrics of all confidence thresholds or only the
            one matching parameters['BBOX_THRESHOLD']. If True, return np.array
            of shape [3xlen(self.all_conf_thrs)], by default [3x13]. If False
            return np.array of shape [3]. Defaults to False.
        """
        if not self.labelled:
            return None, None, None
        # for labelled dataset compute prediction metrics 
        cnfs_mtrx = self.compute_TP_FP_FN(which_dets, t)
        prc_rcl_f1 = self.compute_prc_rcl_F1(cnfs_mtrx)
        if not return_all_conf_thrs:
            idx = np.where(self.all_conf_thrs==self.conf_thr)[0][0]
            return prc_rcl_f1[:, idx]
        return prc_rcl_f1

    def compute_TP_FP_FN(self, which_dets, t, return_FP_FN_mask=False):
        """
        On the detections of frame t, count true-positives (TP), false-positives
        (FP), and false-negatives (FN). Returns an int np.array of shape
        [3xlen(self.all_conf_thrs)], by default [3x13]. Rows represent TP, FP, FN
        counts, columns increasing confident thresholds. Iteratively called for
        in-training performance logging.

        Attributes
        ----------
        which_dets:
            Which set of detections to use. Options: {'all', 'confident',
            'IDed'}

        t: int
            TP, FP, FN of which frame detection to return.
        
        return_FP_FN_mask: bool
            Instead of returning the confusion matrix, return FP, FN masks
            fitting the detections.
        """
        print(f'Calculating FP, TP, and FN at t{t}...', end='', flush=True)
        det = self.get_frame_dets(which_dets, t)

        _, true_det = self.get_frame_and_truedets(t)
        if det.shape[0] == 0:
            det = pd.DataFrame([0,0,0], ['conf', 'anchor_x', 'anchor_y']).T
        if true_det.shape[0] == 0:
            true_det = pd.DataFrame([0,0,0], ['conf', 'anchor_x', 'anchor_y']).T
        # calculate distances between all groundtruth-, and predicted detections
        d = metrics.pairwise.euclidean_distances(true_det.iloc[:,1:], det.iloc[:,1:])

        TP_masks, FP_masks, FN_masks = [], [], []
        for thr in self.all_conf_thrs:
            TPs, FNs = [], []
            for i in range(len(true_det)):
                gt_x, gt_y = true_det.iloc[i,1:]
                dist_to_gt_det = d[i]
                TP_idx = np.where((dist_to_gt_det < self.nms_min_dist) & (det.conf>thr))[0]
                if len(TP_idx) > 1:
                    TP_idx = [TP_idx[np.argmin(dist_to_gt_det[TP_idx])]]
                if len(TP_idx) == 1 and TP_idx not in TPs:
                    TPs.append(TP_idx[0])
                else:
                    FNs.append(i)

            TP_mask = np.zeros(len(det), dtype=bool)
            TP_mask[TPs] = True
            FP_mask = np.where(~TP_mask & (det.conf>thr), True, False)
            FN_mask = np.zeros(len(true_det), dtype=bool)
            FN_mask[FNs] = True 
            TP_masks.append(TP_mask)
            FP_masks.append((FP_mask))
            FN_masks.append(FN_mask)
        
        TP = [tp.sum() for tp in TP_masks]
        FP = [fp.sum() for fp in FP_masks]
        FN = [fn.sum() for fn in FN_masks]
        cnfs_mtrx = np.array([TP, FP, FN])
        if return_FP_FN_mask:
            # select the mask corresbonding to the conf threshold
            idx = np.where(self.all_conf_thrs==self.conf_thr)[0][0]
            return FP_masks[idx], FN_masks[idx]
        print('Done.', flush=True)
        return cnfs_mtrx
        
    def compute_prc_rcl_F1(self, cnfs_mtrx, return_dataframe=False):
        """
        From a passed confusion matrix, calculate precision, recall and F1
        score. Returns a np.array of the same shape but rows now represent
        precision, recall, and F1 score.

        Attributes
        ----------
        which_dets:
            Which set of detections to use. Options: {'all', 'confident',
            'IDed'}

        cnfs_mtrx: np.array [3xlen(self.all_conf_thr)]
            Rows represent TP, FP, FN counts, columns increasing confident
            thresholds. 
        
        return_dataframe: bool
            Convenience argument for use-case during training. Instead of
            returning a np.array, return a pd.DataFrame.
        """
        prc = cnfs_mtrx[0] / (cnfs_mtrx[0]+cnfs_mtrx[1]+1e-6)
        rcl = cnfs_mtrx[0] / (cnfs_mtrx[0]+cnfs_mtrx[2]+1e-6)
        f1 =  2*(prc*rcl)/((prc+rcl) +1e-6)
        metric = np.array([prc, rcl, f1]).round(3)

        if return_dataframe:
            index = pd.MultiIndex.from_product([('precision', 'recall', 'F1'), 
                                                self.all_conf_thrs])
            return pd.Series(metric.flatten(), index=index)
        return metric




















    # def get_IDed_det(self, t, return_ilocs=False):
    #     det = self.detections[t]
    #     if t not in self.IDed_detections.index:
    #         emtpy_IDed_det = pd.DataFrame([], columns=['conf','anchor_x','anchor_y'])
    #         if not return_ilocs:
    #             return emtpy_IDed_det
    #         return emtpy_IDed_det, []
    #     IDed_det = self.IDed_detections.loc[t, ['X','Y']]
    #     half_boxs = self.axon_box_size/2
        
    #     valid_det = []
    #     ilocs = []
    #     # iter old detections (longer or same size as IDed detections)
    #     for i, (_, (conf, anchor_x, anchor_y)) in enumerate(det.iterrows()):
    #         # get the assigned ID by masking for matching box coordinates (hacky)
    #         coo_mask = (IDed_det.values==(anchor_x-half_boxs, anchor_y-half_boxs)).all(1)
    #         ID = IDed_det.index[coo_mask]
    #         # if the box coordinates were still in the IDed detections add them
    #         if not ID.empty:
    #             ilocs.append(i)
    #             valid_det.append(pd.Series([conf, anchor_x, anchor_y], 
    #                                         name=f'Axon_{ID[0]:0>3}', 
    #                                         index=('conf', 'anchor_x', 'anchor_y')))
    #     IDed_det = pd.concat(valid_det, axis=1).T
    #     if not return_ilocs:
    #         return IDed_det
    #     return IDed_det, ilocs

    def assign_ids(self, cache):
        self.dets_astar_paths = self._compute_detections_astar_paths(cache=cache)
        self.IDed_detections = self._assign_IDs_to_detections()

    def _compute_detections_astar_paths(self, cache='to'):
        cache_fname = f'{self.dir}/{self.dataset.name}_astar_dets_paths.pkl'
        if cache == 'from':
            print('Getting A* between-detection paths from cached file...', end='')
            with open(cache_fname, 'rb') as file:
                dets_astar_paths = pickle.load(file)

        else:
            print('Computing A* detection paths between detections...', end='')
            # make the cost matrix for finding the astar path
            dets_astar_paths = {}
            for t in range(len(self)):
                lbl = f'{self.dataset.name}_t:{t:0>3}'
                print('\n')

                weights = np.where(self.dataset.mask[t].todense()==1, 1, 2**16).astype(np.float32)

                # get current detections
                t_dets = self.detections[t]
                # ids (placeholders)
                t_ids = [int(idx[-3:]) for idx in t_dets.index]
                
                # from matplotlib import pyplot as plt
                # plt.subplots(1,1, figsize=(16,9))
                # plt.imshow(weights, cmap='gray')
                # plt.scatter(t_dets.anchor_x, t_dets.anchor_y, s=50, alpha=.6, )
                # plt.figure(16,9)
                # plt.save(f'/home/ssteffens/frame{t:0>2}.png')
                # plt.close()
                
                t_dets = np.stack([t_ids, t_dets.anchor_y, t_dets.anchor_x], -1)
                
                # predecessor timestamps
                for t_bef in range(t-1, t-(self.MCF_max_num_misses+2), -1):
                    if t_bef < 0:
                        continue
                    lbl_t_before = f'{lbl}-t:{t_bef:0>3}'
                    print('\t', lbl_t_before, end=': ')
                    
                    # get detections of a timestep before
                    t_bef_dets = self.detections[t_bef]
                    t_bef_ids = [int(idx[-3:]) for idx in t_bef_dets.index]
                    t_bef_dets = np.stack([t_bef_ids, t_bef_dets.anchor_y, t_bef_dets.anchor_x], -1)

                    # iterate predecessor detections, compute its distance to all detections at t
                    astar_paths = []
                    for i, t_bef_det in enumerate(t_bef_dets):
                        print(f'{i+1}/{len(t_bef_dets)}', end='...', flush=True)

                        with ThreadPoolExecutor() as executer:
                            args = repeat(t_bef_det), t_dets, repeat(weights)
                            as_paths = executer.map(self._get_path_between_dets, *args)
                        
                        astar_paths.append([p for p in as_paths])
                    dets_astar_paths[lbl_t_before] = astar_paths
                
            if cache == 'to':
                print('caching...', end='')
                with open(cache_fname, 'wb') as file:
                    pickle.dump(dets_astar_paths, file)
        print('Done.')
        return dets_astar_paths

    def _get_path_between_dets(self, t1_det, t0_det, weights):
        t1id, t1y, t1x = t1_det
        t0id, t0y, t0x = t0_det
        
        # compute euclidean distance 
        dist = np.sqrt((t1y-t0y)**2 + (t1x-t0x)**2)
        # path = None
        # only if it's lower than thr, put in the work to compute A* paths
        if dist < self.max_px_assoc_dist:
            path, dist = self._compute_astar_path((t1y,t1x), (t0y,t0x), weights, True, self.max_px_assoc_dist)
        # cap all distances to thr, A* may be longer than Eucl.
        elif dist >= self.max_px_assoc_dist:
            path = None
        return path

    def _compute_astar_path(self, source, target, weights, return_dist=True, max_path_length=10000):
        path_coo = pyastar2d.astar_path(weights, source, target, max_path_length)
        if path_coo is not None:
            ones, row, cols = np.ones(path_coo.shape[0]), path_coo[:,0], path_coo[:,1]
            path = sparse.coo_matrix((ones, (row, cols)), shape=weights.shape, dtype=bool)
            if return_dist:
                return path, path_coo.shape[0]
            else:
                return path
        elif return_dist:
            return None, None
        else:
            return None





















    def _assign_IDs_to_detections(self):
        def observation_model(**kwargs):
            """Compute Observation Cost for one frame

            Parameters
            ------------
            scores: array like
                (N,) matrix of detection's confidence

            Returns
            -----------
            costs: ndarray
                (N,) matrix of detection's observation costs
            """
            scores = np.array(kwargs['scores']).astype(np.float)
            
            # why not like in paper where beta (uncertainty) is used instead of 
            # confidence? Maybe relaxation because conf == 1 gives neg infinity
            scores = (scores-1) *-1   # conf to beta
            scores = (np.log(scores/(1-scores)))
            scores[scores>self.MCF_max_conf_cost] = self.MCF_max_conf_cost
            scores[scores<-self.MCF_max_conf_cost] = -self.MCF_max_conf_cost
            return scores
            # return np.log(-0.446 * np.array(kwargs['scores']).astype(np.float) + 0.456)

        def feature_model(**kwargs):
            """Compute each object's K features in a frame

            Parameters
            ------------
            image: ndarray
                bgr image of ndarray
            boxes: array like
                (N,4) matrix of boxes in (x,y,w,h)

            Returns
            ------------
            features: array like
                (N,K,1) matrix of features
            """
            assert 'image' in kwargs and 'boxes' in kwargs, 'Parameters must contail image and boxes'

            boxes = kwargs['boxes']
            image = kwargs['image']
            if len(boxes) == 0:
                return np.zeros((0,))

            boxes = np.atleast_2d(deepcopy(boxes))
            features = np.zeros((boxes.shape[0], 180, 1), dtype=np.float32)

            for i, roi in enumerate(boxes):
                x1 = max(roi[1], 0)
                x2 = max(roi[0], 0)
                x3 = max(x1 + 1, x1 + roi[3])
                x4 = max(x2 + 1, x2 + roi[2])
                cropped = image[x1:x3, x2:x4]
                # cropped = np.moveaxis(image[:, x1:x3, x2:x4], 0, -1)[:,:,0]
                hist = cv2.calcHist([cropped], [0], None, [180], [0, 1])
                cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
                features[i] = deepcopy(hist)
            return features

        def transition_model(**kwargs):
            """Compute costs between track and detection

            Parameters
            ------------
            miss_rate: float
                the similarity for track and detection will be multiplied by miss_rate^(time_gap - 1)
            time_gap: int
                number of frames between track and detection
            predecessor_boxes: ndarray
                (N,4) matrix of boxes in track's frame
            boxes: ndarray
                (M,4) matrix of boxes in detection's frame
            predecessor_features: ndarray
                (N,180, 256) matrix of features in track's frame
            features: ndarray
                (M,180, 256) matrix of features in detection's frame
            frame_idx: int
                frame id, begin from 1

            Returns
            ------------
            costs: ndarray
                (N,M) matrix of costs between track and detection
            """
            miss_rate = kwargs['miss_rate']
            time_gap = kwargs['time_gap']
            boxes = kwargs['boxes']
            predecessor_boxes = kwargs['predecessor_boxes']
            features = kwargs['features']
            predecessor_features = kwargs['predecessor_features']
            frame_idx = kwargs['frame_idx']

            # A* distances component of cost
            lbl = f'{self.dataset.name}_t:{frame_idx:0>3}-t:{frame_idx-(time_gap):0>3}'
            distances = ((astar_dists[lbl] /self.max_px_assoc_dist) -1) *-1
            inf_dist = distances == 0

            # visual similarity component of cost
            vis_sim = []
            for f1 in predecessor_features:
                vis_sim_f1 = []
                for f2 in features:
                    vis_sim_f1.append(1 - cv2.compareHist(f1, f2, cv2.HISTCMP_BHATTACHARYYA))
                vis_sim.append(vis_sim_f1)
            vis_sim = np.array(vis_sim)
            
            costs = -np.log((1-self.MCF_vis_sim_weight) * distances*(miss_rate ** (time_gap-1)) \
                            + self.MCF_vis_sim_weight * vis_sim \
                            + 1e-5)
            costs[inf_dist] = np.inf
            return costs
        
        print('Assigning axon IDs using min cost flow...', end='')
        # get input data (A* distances and detections in libmot format)
        astar_dists = self._get_astar_path_distances(self.dets_astar_paths)
        dets = [self.det2libmot_det(self.detections[t], t, empty_id=True) 
                for t in range(len(self))]
        dets = np.concatenate(dets)

        # cap confidence to 1
        if self.MCF_conf_capping_method == 'ceil':
            dets[:, -1][dets[:, -1]>1] = 1
        if self.MCF_conf_capping_method == 'scale_to_max':
            dets[:, -1] /= dets[:, -1].max()

        # setup the tracker model
        track_model = MinCostFlowTracker(observation_model = observation_model,
                                        transition_model = transition_model, 
                                        feature_model = feature_model,
                                        entry_exit_cost = self.MCF_entry_exit_cost, 
                                        min_flow = self.MCF_min_flow,
                                        max_flow = self.MCF_max_flow, 
                                        miss_rate = self.MCF_miss_rate, 
                                        max_num_misses = self.MCF_max_num_misses,
                                        cost_threshold = self.MCF_edge_cost_thr)

        for i in range(len(self)):
            print(f'frame {i}/{(len(self)-1)}', end='...', flush=True)
            det = dets[(dets[:, 0] == i), :]
            img, gt = self.get_frame_and_truedets(i)
            track_model.process(boxes=det[:, 2: 6].astype(np.int32),
                                scores=det[:, 6], image=img[0].numpy(),
                                frame_idx=i)

        # trajectory is a list [IDs] of list of boxes (in frames) with format
        # frame_idx, box_idx, box_coo (4 values)
        trajectory = track_model.compute_trajectories()
        
        # iterate IDs
        record = []
        for i, t in enumerate(trajectory):
            # check how long the ID lives
            if len(t) < self.MCF_min_ID_lifetime:
                continue
            # iterate over lifetime timepoints
            for j, box in enumerate(t):
                record.append([box[0], i, box[2][0], box[2][1], box[2][2], box[2][3]])
        print(f'-> {i} axon IDs. Done.')

        track = np.array(record)
        track = track[np.argsort(track[:, 0])]

        IDed_detections = pd.DataFrame(track, columns=['FrameId', 'Id', 'X', 'Y', 'Width', 'Height'])
        return IDed_detections.set_index(['FrameId', 'Id'])



        # # for target hack (uncomment line in compute dets path to target as well)
        # self.to_target_paths = self._compute_dets_path_to_target(cache='nooo')
        # true_to_target_dists = self._get_astar_path_distances(self.to_target_paths)
        # true_to_target_dists = [pd.Series(true_to_target_dists[t], index=self.get_frame_and_truedets(t, return_tiled=False)[1].index, name=t) 
        #                         for t in range(len(true_to_target_dists))]
        # self.to_target_dists = pd.concat(true_to_target_dists, axis=1)
        # self.to_target_dists.to_csv(f'{self.dir}/train_data_dists_gt.csv')

    # def get_dets_in_libmot_format(self):
    #     libmot_dets = []
    #     for t in range(len(self)):
    #         conf, x, y = self.detections[t].values.T
    #         x_topleft = x-self.axon_box_size//2
    #         y_topleft = y-self.axon_box_size//2
    #         frame_id = np.full(conf.shape, t)
    #         empty_id = np.full(conf.shape, -1)
    #         boxs = np.full(conf.shape, self.axon_box_size)

    #         libmot_dets.append(np.stack([frame_id, empty_id, x_topleft, y_topleft, 
    #                                      boxs, boxs, conf]).T)
    #     return np.concatenate(libmot_dets)

    def _get_astar_path_distances(self, astar_paths, to_array=True):
        def rec_get_distance(path_list):
            if isinstance(path_list, list):
                # unpack the next depth level
                dists_list = [rec_get_distance(el) for el in path_list]
            else:
                # path_list fully unpacked, elements are either None or scipy.coo
                dist = self.max_px_assoc_dist if path_list is None else path_list.getnnz()
                # built up nested list of distances
                return dist
            # final exit
            return dists_list
        
        dictinput = isinstance(astar_paths, dict)
        if dictinput:
            keys, astar_paths = astar_paths.keys(), list(astar_paths.values())
        
        dists = rec_get_distance(astar_paths)
        if to_array:
            dists = [np.array(ds) for ds in dists]
        
        # reconstruct dict input format
        if dictinput:
            dists = dict(zip(keys, dists))
        return dists












    def get_axon_reconstructions(self, t, interpolate_missed_dets=True):
        weights = np.where(self.dataset.mask[t].todense(), 1, 2**16).astype(np.float32)
        if not t:
            return pd.DataFrame([])
        # get the paths from t0 to t before (t0-1)
        lbl = f'{self.dataset.name}_t:{t:0>3}-t:{t-1:0>3}'
        paths = self.dets_astar_paths[lbl]

        # get the IDed (filtered) detections
        dets_t, ilocs_t = self.get_frame_dets('IDed', t, return_ilocs=True)
        dets_t_bef, ilocs_t_bef = self.get_frame_dets('IDed', t-1, return_ilocs=True)
        if interpolate_missed_dets and t>1:
            dets_2t_bef = self.get_frame_dets('IDed', t-2, return_ilocs=False)
        
        reconstructions = [pd.DataFrame([])]
        # iterate the IDed (filtered) detections at t=0
        for i, axon_t0 in enumerate(dets_t.index):
            # check which iloc this det had in the unfiltered t0 dets
            t0_unfilt_dets_idx = ilocs_t[i]
            
            # from the IDed (filtered) t0-1 dets, get the iloc of its t0 match
            t_bef_IDed_dets_matched_idx = np.where(dets_t_bef.index.values == axon_t0)[0]
            # if the matching name was in the filtered t0-1 dets
            if len(t_bef_IDed_dets_matched_idx) == 1:
                # check which iloc this t0-1 det had in the unfiltered t0-1 dets
                t_bef_unfilt_dets_idx = ilocs_t_bef[t_bef_IDed_dets_matched_idx[0]]
                # finally use the indices corresbonding to unfiltered dets to 
                # index the paths (which were computed on the unfiltered dets...)
                p = paths[t_bef_unfilt_dets_idx][t0_unfilt_dets_idx]
            
            # interpolate one frame detection by calculating A* from t0 to t-2
            elif interpolate_missed_dets and t>1 and axon_t0 in dets_2t_bef.index:
                yx_det_2t_before = dets_2t_bef.loc[axon_t0][['anchor_y','anchor_x']]
                yx_det = dets_t.loc[axon_t0][['anchor_y','anchor_x']]
                p = self._compute_astar_path(yx_det.astype(int), 
                                             yx_det_2t_before.astype(int), 
                                             weights, False)
            else:
                continue 
            cols = pd.MultiIndex.from_product([[axon_t0],['X','Y'],[t]])
            reconstructions.append(pd.DataFrame([p.col, p.row], index=cols).T)
        return pd.concat(reconstructions, axis=1)







    def det2libmot_det(self, detection, t, empty_id=False, drop_conf=False, to_pandas=False):
        conf, x, y = detection.values.T
        x_topleft = x-self.axon_box_size//2
        y_topleft = y-self.axon_box_size//2
        frame_id = np.full(conf.shape, t)
        boxs = np.full(conf.shape, self.axon_box_size)
        if empty_id:
            axon_id = np.full(conf.shape, -1)
        else:
            axon_id = np.array([int(idx[-3:]) for idx in detection.index])

        det_libmot = np.stack([frame_id, axon_id, x_topleft, y_topleft, 
                               boxs, boxs, conf]).T

        cols = ['FrameId', 'Id', 'X', 'Y', 'Width', 'Height', 'conf']
        if drop_conf:
            cols = cols[:-1]
            det_libmot = det_libmot[:, :-1]
        if to_pandas:
            det_libmot = pd.DataFrame(det_libmot, columns=cols)
            det_libmot = det_libmot.set_index(['FrameId', 'Id'])
        return det_libmot













    def get_all_IDassiged_dets(self):
        all_detections = []
        for t in range(len(self)):
            # det = self.get_IDed_det(t)
            det = self.get_frame_dets('IDed', t)
            det.columns = pd.MultiIndex.from_product([[t], det.columns])
            all_detections.append(det)
        return pd.concat(all_detections, axis=1)

    def compute_target_distances(self, cache):
        self.to_target_paths = self._compute_dets_path_to_target(cache=cache)
        to_target_dists = self._get_astar_path_distances(self.to_target_paths)
        idx = self.get_frame_dets('IDed', t).index
        to_target_dists = [pd.Series(to_target_dists[t], index=idx, name=t) 
                           for t in range(len(self))]
        self.to_target_dists = pd.concat(to_target_dists, axis=1)
        
    def _compute_dets_path_to_target(self, cache='to'):
        cache_fname = f'{self.dir}/{self.dataset.name}_astar_target_paths.pkl'
        if cache == 'from':
            print('Getting A* target paths from cache file...', end='')
            with open(cache_fname, 'rb') as file:
                target_astar_paths = pickle.load(file)
            
        else:
            print('Computing A* paths to target...', end='')
            target_yx = self.dataset.structure_outputchannel_coo
            
            target_astar_paths = []
            for t in range(len(self)):
                print(f't:{t:0>3}', end='...', flush=True)
                yx = self.get_frame_dets('IDed', t).loc[:, ['anchor_y', 'anchor_x']].values.astype(int)
                # yx = self.get_IDed_det(t).loc[:, ['anchor_y', 'anchor_x']].values.astype(int)
                # yx = self.get_frame_and_truedets(t, return_tiled=False)[1].loc[:, ['anchor_y', 'anchor_x']].values
                
                weights = np.where(self.dataset.mask[t].todense(), 1, 2**16).astype(np.float32)

                with ThreadPoolExecutor() as executer:
                    args = repeat(target_yx), yx, repeat(weights), repeat(False)
                    as_paths = executer.map(self._compute_astar_path, *args)
                target_astar_paths.append([p for p in as_paths])
            
            if cache == 'to':
                print('caching...', end='')
                with open(cache_fname, 'wb') as file:
                    pickle.dump(target_astar_paths, file)
        print('Done.')
        return target_astar_paths





















    # dirty zone
    def search_MCF_params(self, show_results=False, dest_dir=None):
        # if below ran once, viz results
        if show_results or dest_dir:
            ec = self.MCF_edge_cost_thr
            eec = self.MCF_entry_exit_cost
            mr = self.MCF_miss_rate
            vsw = self.MCF_vis_sim_weight
            ccm = self.MCF_conf_capping_method
            print(f'Current params: ec:{ec} eec:{eec} mr:{mr} vsw:{vsw} ccm:{ccm}')

            metrics = pd.read_csv(f'{self.dir}/MCF_param_search.csv', index_col=0)
            # print(metrics.iloc[:,[1169, 1029, 813, 914, 780, 483, 316, 252, 32, 1, 0, 18]].T.to_string())
            metrics.sort_values(['mota', 'idf1'], axis=1, inplace=True, ascending=False)
            prc = metrics.loc['idp']
            rcl = metrics.loc['idr']
            mota = metrics.loc['mota']
            f1 = metrics.loc['idf1']

            n = metrics.loc['num_unique_objects'][0]
            mostly_tr = metrics.loc['mostly_tracked'][0] / n
            part_tr = metrics.loc['partially_tracked'][0] / n
            mostly_lost = metrics.loc['mostly_lost'][0] / n

            print()
            print(metrics.iloc[:,:10].columns)
            print(metrics.iloc[:,:10].T.reset_index(drop=True).T)
            print()


            fig, axes = plt.subplots(1, 2, figsize=config.MEDIUM_FIGSIZE)
            fig.subplots_adjust(bottom=.01, left=.1, wspace=.3, right=.95)
            ticks = np.arange(0.5, .91, 0.2)
            lims = (.3, .94)

            # setup both axes
            for i, ax in enumerate(axes):
                ax.set_ylim(*lims) 
                ax.set_yticks(ticks)
                ax.set_yticklabels([f'{t:.1f}' for t in ticks], fontsize=config.SMALL_FONTS)

                ax.set_xlim(*lims)
                ax.set_xticks(ticks)
                ax.set_xticklabels([f'{t:.1f}' for t in ticks], fontsize=config.SMALL_FONTS)

                ax.set_aspect('equal')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.grid()
                if i == 0:
                    ax.set_xlabel('Identity precision', fontsize=config.FONTS)
                    ax.set_ylabel('Identity recall', fontsize=config.FONTS)
                else:
                    ax.set_xlabel('MOT accuracy', fontsize=config.FONTS)
                    ax.set_ylabel('Identity F1', fontsize=config.FONTS)

            colors = []
            sizes = []
            for i in range(len(mota)):
                if i == 0:
                    colors.append(config.DEFAULT_COLORS[0])
                    sizes.append(250)
                else:
                    colors.append('k')
                    sizes.append(3)
            axes[0].scatter(prc, rcl, alpha=1, edgecolor='k', color=colors, s=sizes)
            axes[1].scatter(mota, f1, alpha=1, edgecolor='k', color=colors, s=sizes)


            bottom, left = .85, .1
            width, height = .85, .1
            ax = fig.add_axes([left, bottom, width, height])
            ax.set_ylim(.2,.8)
            ax.set_xlim(0,1)
            ax.set_xticks(np.arange(0,1.01,.1))
            ax.tick_params(left=False, labelleft=False, labelsize=config.SMALL_FONTS)

            ax.text(mostly_tr/2, .5, 'mostly tracked', clip_on=False, 
                    ha='center', va='center', fontsize=config.SMALL_FONTS)
            ax.text(mostly_tr + part_tr/2, .5, 'partially tracked', clip_on=False, 
                    ha='center', va='center', fontsize=config.SMALL_FONTS)
            ax.text(mostly_tr + part_tr + mostly_lost/2, .5, 'mostly lost', clip_on=False, 
                    ha='center', va='center', fontsize=config.SMALL_FONTS)
            ax.set_title('Proportion of axons', fontsize=config.FONTS)

            ax.barh(0.5, mostly_tr, color=config.DEFAULT_COLORS[0], alpha=.8)
            ax.barh(0.5, part_tr, left=mostly_tr, color=config.DEFAULT_COLORS[2], alpha=.8)
            ax.barh(0.5, mostly_lost, left=mostly_tr+part_tr, color=config.DEFAULT_COLORS[1], alpha=.8)
            
            if show_results:
                plt.show()
            elif dest_dir:
                plt.savefig(f'{dest_dir}/MCF_param_search_results.{config.FIGURE_FILETYPE}')



        else:
            # get the targets to compute MOT metrics later
            target_dets_IDed = []
            for t in range(len(self)):
                _, frame_target = self.get_frame_and_truedets(t)
                target_dets_IDed.append(self.det2libmot_det(frame_target, t, 
                                                    to_pandas=True, drop_conf=True))
            target_dets_IDed = pd.concat(target_dets_IDed)

            # hyperparams to search over
            edge_cost_thr_values =  [.4, .6, .7, .8, .9, 1, 1.2, 3]
            entry_exit_cost_values = [ .2, .8, .9, 1, 1.1,  2]
            miss_rate_values =  [0.9, 0.6]
            vis_sim_weight_values = [0, 0.1]
            conf_capping_method_values = ['ceil' , 'scale_to_max']
            # edge_cost_thr_values =  [.5, .6, .7, .8, .9, 1, 1.2, 1.6]
            # entry_exit_cost_values = [.4, .8, .9, 1, 1.1, 1.5]
            # miss_rate_values =  [0.7]
            # vis_sim_weight_values = [0]
            # conf_capping_method_values = ['scale_to_max']

            results = []
            # # uncomment below to search over small preslected param combinations
            # for (ec, eec, mr, vsw, ccm) in prom_combs:
            #     self.MCF_edge_cost_thr = ec
            #     self.MCF_entry_exit_cost = eec
            #     self.MCF_miss_rate = mr
            #     self.MCF_vis_sim_weight = vsw
            #     self.MCF_conf_capping_method = ccm
            
            # brute search all param combinations 
            for ec in reversed(edge_cost_thr_values):
                self.MCF_edge_cost_thr = ec
                for eec in reversed(entry_exit_cost_values):
                    self.MCF_entry_exit_cost = eec
                    for mr in miss_rate_values:
                        self.MCF_miss_rate = mr
                        for vsw in vis_sim_weight_values:
                            self.MCF_vis_sim_weight = vsw
                            for ccm in conf_capping_method_values:
                                self.MCF_conf_capping_method = ccm

                                key = f'ec:{ec} eec:{eec} mr:{mr} vsw:{vsw} ccm:{ccm}'
                                print('\n\n\n', key, flush=True)

                                # using MCF params defined above, assign IDs
                                self.assign_ids(cache='from')
                                
                                # compute MOT benchmark
                                acc = mm.utils.compare_to_groundtruth(target_dets_IDed, self.IDed_detections, 'euclidean', distth=(self.nms_min_dist)**2)
                                mh = mm.metrics.create()
                                summary = mh.compute(acc, metrics=mm.metrics.motchallenge_metrics, \
                                                    return_dataframe=True).iloc[0]
                                
                                # # compute precision and recall my way (tiny difference with MOT Prc, Rcl?)
                                # prc_rcl = np.array([(self.get_detection_metrics(t, 'ID_assigned')) for t in range(len(self))])
                                # prc_rcl = pd.Series(prc_rcl.mean(0)[:-1], index=['MyPrc', 'MyRcl'])
                                # summary = summary.T.append(prc_rcl)

                                # # compute the distance to target metrics
                                # design_screen = StructureScreen(self.dataset, self.dir, 
                                #                                 axon_detections=self, 
                                #                                 cache_target_distances='no')
                                # # compare if metrics like n axons crossgrown match ground truth
                                # metrics = pd.Series(design_screen.compare_to_gt_hacky(self.dataset.name))
                                # summary = summary.append(metrics)
                                
                                print(summary)
                                summary.name = key
                                results.append(summary)

                results_csv = pd.concat(results, axis=1)
                results_csv.to_csv(f'{self.dir}/MCF_param_search.csv')