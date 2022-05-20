import os
import pickle
from itertools import repeat
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import torch

from sklearn import metrics
from libmot.data_association import MinCostFlowTracker
import motmetrics as mm

from .utils import _compute_astar_path
from .mincostflow_models import observation_model, transition_model, feature_model

class AxonDetections(object):
    """
    Bundles model inference output to data input. All conversions from model
    output to usable detections are implemented in this class.
    """
    def __init__(self, model, dataset, parameters, directory, 
                 timepoint_subset=None):
        """
        Create an AxonDetection object. 

        Arguments
        ---------
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
        self.name = dataset.name 
        self.dir = directory
        if self.dir:
            os.makedirs(self.dir, exist_ok=True)

        self.timepoint_subset = timepoint_subset if timepoint_subset is not None else range(self.dataset.sizet)
        
        # basic parameters
        self.device = parameters['DEVICE']
        self.Sx = parameters['SX']
        self.Sy = parameters['SY']
        self.tilesize = parameters['TILESIZE']
        
        # min cost flow hyperparamters 
        self.MCF_edge_cost_thr = parameters['MCF_EDGE_COST_THR']
        self.MCF_entry_exit_cost = parameters['MCF_ENTRY_EXIT_COST']
        self.MCF_miss_rate = parameters['MCF_MISS_RATE']
        self.MCF_max_num_misses = parameters['MCF_MAX_NUM_MISSES']
        self.MCF_min_flow = parameters['MCF_MIN_FLOW']
        self.MCF_max_flow = parameters['MCF_MAX_FLOW']
        self.MCF_max_conf_cost = parameters['MCF_MAX_CONF_COST']
        self.MCF_vis_sim_weight = parameters['MCF_VIS_SIM_WEIGHT']
        self.MCF_conf_capping_method = parameters['MCF_CONF_CAPPING_METHOD']
        
        # parameters for selecting detections
        self.nms_min_dist = parameters.get('NON_MAX_SUPRESSION_DIST')
        self.conf_thr = parameters['BBOX_THRESHOLD']
        self.all_conf_thrs = np.sort(np.append(np.arange(0.55, 1, .04), self.conf_thr)).round(2)
        self.max_px_assoc_dist = 500
        self.axon_box_size = 70     # only for visualization
        self.labelled = dataset.target.empty

    def __len__(self):
        """
        The length of axon detections. Here, the number of frames.
        """
        return len(self.timepoint_subset)

    def detect_dataset(self, cache=None):
        """
        Use the model to detect growth cones. yolo_targets, pandas_tiled_dets,
            detections are created. Called from outside this class.

        Arguments
        ---------
        cache: str
            Whether to save the computed data to a file or use precomputed data
            or don't do any caching. Options: {'to', 'from', None}. Defaults to
            None, no caching.
        """
        self.dataset.construct_tiles(self.device, force_no_transformation=True)

        if cache == 'from':
            self._yolo_targets = self.from_cache('_yolo_targets')
            self._pandas_tiled_dets = self.from_cache('_pandas_tiled_dets')
            self._detections = self.from_cache('_detections')

        else:
            self._yolo_targets = []
            self._pandas_tiled_dets = []
            self._detections = []
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
                self._yolo_targets.append(yolo_target)
                self._pandas_tiled_dets.append(pandas_tiled_det)
                self._detections.append(pandas_frame_det_nms)
            print('Done.\n', flush=True)
        
        if cache == 'to':
            self.to_cache('_yolo_targets', self._yolo_targets)
            self.to_cache('_pandas_tiled_dets', self._pandas_tiled_dets)
            self.to_cache('_detections', self._detections)
    
    def from_cache(self, which):
        """
        Load a pickled piece of data. Previously computed data, e.g. detections,
        are read as files and returned.

        Arguments
        ---------
        which: str
            Which data to get. Options: {'yolo_targets', 'pandas_tiled_dets',
            'detections', 'astar_dets_paths', 'IDed_detections', ...} 
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

        Arguments
        ---------
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

        Arguments
        ---------
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

        # convert the YOLO coordinates (0-1) to image coordinates 
        # (tilesize x tilesize) (inplace function)
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

        Arguments
        ---------
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

    def get_frame_dets(self, which_dets, t, libmot=False, unstitched=False):
        """
        Get detections of one frame as a pd.DataFrame. pd.Dataframe has axon
        names as indices (eg. Axon_042) and three columns [conf, anchor_x,
        anchor_y]. Requires to have previously called detect_dataset(). May also
        return all detections if t is None.

        Arguments
        ---------
        which_dets: str
            Which set of detections to get. Options: {'all', 'confident',
            'FP_FN', 'IDed', 'groundtruth'}. 'all' will return detections above
            a very low threshold set by the lowest value of self.all_conf_thrs,
            here 0.55. 'confident' will return the detections above the
            predefined threshold set in parameters['BBOX_THRESHOLD']. 'FP_FN'
            will return two pd.DataFrames, instead of just one. These detections
            only include false-positives (FP) and false-negative (FN)
            detections. 'IDed' will return detections with assigned identity.
            'groundtruth' will return the labelled detections. Requires to have
            previously run assign_ids()

        t: int
            Which frame to return the detections of. if None, return all
            detections concatenated in a pd.DataFrame.

        libmot: bool
            Return the detection in libmot format.
        
        unstitched: bool
            Only implemented/ considered when which_dets == 'confident' or
            'all'. Return tile-wise list of pd.DataFrames (unstitched). Defaults
            to False.
        """
        if t is None:
            # recursively get framewise detections, concat
            all_dets = [self.get_frame_dets(which_dets, t, libmot) for t in range(len(self))]
            # libmot is merged by-row, all others by-column
            return pd.concat(all_dets, axis=not libmot)
        
        # filter the detection set at this frame to confidence threshold 
        assert hasattr(self, '_detections'), "Run .detect_dataset() first!"
        if which_dets == 'all':
            if not unstitched:
                det = self._detections[t]
            else:
                det = self._pandas_tiled_dets[t]
        
        elif which_dets == 'confident':
            if not unstitched:
                det = self._detections[t][self._detections[t].conf>self.conf_thr]
            else:
                det = [d[d.conf>self.conf_thr] for d in self._pandas_tiled_dets[t]]

        elif which_dets == 'IDed':
            assert hasattr(self, '_IDed_detections') and self._IDed_detections, "Run .assign_IDs() first!"
            det = self._IDed_detections[t]
        
        elif which_dets == 'groundtruth':
            assert self.labelled, 'No labels exist for this detection object!'
            det = self.get_frame_and_truedets(t)[1]

        elif which_dets == 'FP_FN':
            assert self.labelled, 'Cannot compute FP FN on unlabelled data'
            dets = self.get_frame_dets('confident', t).copy()
            true_dets = self.get_frame_dets('groundtruth', t)
            FP_mask, FN_mask = self.compute_TP_FP_FN('confident', t, 
                                                     return_FP_FN_mask=True)
            FP_dets = dets[FP_mask]
            FN_dets = true_dets[FN_mask]
            return FP_dets, FN_dets

        if libmot:
            return self.det2libmot_det(det, t)
        return det.copy()

    def get_frame_and_truedets(self, t, unstitched=False):
        """
        Get a drawable frame and the matching groundtruth detections. Returns a
        np.array of shape [cfizex] and pd.DataFrame of
        true detections. Channels is 1 if the detector is not set to use
        precomputed motion (default: 1 channel)

        Arguments
        ---------
        t: int
            Which frame and matching detections to return. 
        
        unstitched: bool
            Return the unstitched versions (tile-wise) list of img + detections. 
        """
        img_tiled, _ = self.dataset.get_frametiles_stack(t, self.device)
        pd_tiled_true_det = self._yolo_Y2pandas_det(self._yolo_targets[t], conf_thr=1)
        pd_frame_true_det, img = self.dataset.stitch_tiles(pd_tiled_true_det, img_tiled)

        if not unstitched:
            return img, pd_frame_true_det
        return img_tiled, pd_tiled_true_det

    def get_detection_metrics(self, which_dets, t, return_all_conf_thrs=False):
        """
        Get the detection metrics of one-frame-detections: precision, recall,
        and F1 score. Returns either just the metrics of the preset threshold or
        metrics for all thresholds. Called outside the class after training.

        Arguments
        ---------
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

        Arguments
        ---------
        which_dets:
            Which set of detections to use. Options: {'all', 'confident',
            'IDed'}

        t: int
            TP, FP, FN of which frame detection to return.
        
        return_FP_FN_mask: bool
            Instead of returning the confusion matrix, return FP, FN masks
            fitting the detections.
        """
        print(f'\nCalculating FP, TP, and FN at t{t}...', end='', flush=True)
        det = self.get_frame_dets(which_dets, t)
        true_det = self.get_frame_dets('groundtruth', t)
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

        Arguments
        ---------
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

    def assign_ids(self, astar_paths_cache=None, assigedIDs_cache=None):
        """
        Associate detections with one another to create identities through time.
        Called from outside the class. 

        Arguments
        ---------
        astar_paths_cache: str
            Whether to save the computed A* paths to a file or use precomputed
            A* paths between detections. When None, no caching. Options: {'to',
            'from', None}. Defaults to None, no caching. 

        assigedIDs_cache: str
            Whether to use precomputed identity assignment. Options: {'to',
            'from', None}. Defaults to None, no caching. 

        """
        self.astar_dets_paths = self._compute_detections_astar_paths(cache=astar_paths_cache)
        self._IDed_detections = self._assign_IDs_to_detections(cache=assigedIDs_cache)
        self.IDed_dets_all = self._agg_all_IDed_dets()

    def _compute_detections_astar_paths(self, cache='to'):
        """
        Compute the A* paths between detections of adjacent frames. Returns a
        dictionary where single entries represent all distances between two
        frames detections. Keys contain the dataset name and frame index, eg.
        'test_t:001-t:000'. Values are a 2-level nested list representing the A*
        paths between single detections in frame t (outer list) and single
        detections at frame t-j, with j = (0,1,...) by default (0,1). A* paths
        or elements in the list may be None if the euclidian distance between
        two detections is >max_px_assoc_dist, by default 500 (reducing
        computational load). A* paths are scipy.sparse.coo_matrix object
        computed on the dataset.mask.

        Arguments
        ---------
        cache: str
            Whether to save the computed A* paths to a file or use precomputed
            A* paths between detections. When None, no caching. Options: {'to',
            'from', None}. Defaults to None, no caching. 
        """
        if cache == 'from':
            astar_dets_paths = self.from_cache('astar_dets_paths')

        else:
            print('\nComputing A* detection paths between detections...', end='')
            # make the cost matrix for finding the astar path
            astar_dets_paths = {}
            for t in range(len(self)):
                lbl_t = f'{self.dataset.name}_t:{t:0>3}'
                # get current detections
                t_dets = self.get_frame_dets('all', t)
                weights = self._get_maskweights(t)
                
                # predecessor timestamps (could be more than one)
                for t_bef in range(t-1, t-(self.MCF_max_num_misses+2), -1):
                    if t_bef < 0:
                        continue
                    lbl = f'{lbl_t}-t:{t_bef:0>3}'
                    print('\t', lbl, end=': ')
                    # get detections of a timestep before
                    t_bef_dets = self.get_frame_dets('all', t_bef)

                    # iterate predecessor detections, compute its distance to all detections at t
                    astar_paths = []
                    for i, (_, t_bef_det) in enumerate(t_bef_dets.iterrows()):
                        print(f'{i+1}/{len(t_bef_dets)}', end='...', flush=True)

                        with ThreadPoolExecutor() as executer:
                            args = repeat(t_bef_det.values), t_dets.values, repeat(weights)
                            as_paths = executer.map(self._get_path_between_dets, 
                                                    *args)
                        
                        astar_paths.append([p for p in as_paths])
                    astar_dets_paths[lbl] = astar_paths
                print('\n')
            print('Done.')
            
            if cache == 'to':
                self.to_cache('astar_dets_paths', astar_dets_paths)
        return astar_dets_paths

    def _get_maskweights(self, t):
        """
        Get the segmentation mask with the dataset object at frame t. Convert
        the sparse bool mask to a weight matrix where True -> 1, False -> 2^16.
        Returns a np.array of shape [frame_height, frame_width]

        Arguments
        ---------
        t: int
            The frame index to which the returned mask corresbonds to. 
        """
        return np.where(self.dataset.mask[t].todense()==1, 1, 2**16).astype(np.float32)

    def _get_path_between_dets(self, t1_det, t0_det, weights):
        """
        Calculate the A* path between two detections (originating from different
        frames). Returns the A* path or None.

        Arguments
        ---------
        t1_det: np.array
            Detection at frame t=1 - np.array with elements [confidence, x, y].
        
        t0_det: np.array
            Detection at frame t=0 - np.array with elements [confidence, x, y].
        
        weights: np.array
            Weight matrix on which the A* paths are calculated. Weight matrix
            follows segmentation mask of the dataset.
        """
        _, t1x, t1y = t1_det
        _, t0x, t0y = t0_det
        # compute euclidean distance 
        dist = np.sqrt((t1y-t0y)**2 + (t1x-t0x)**2)
        # path = None
        # only if it's lower than thr, put in the work to compute A* paths
        if dist < self.max_px_assoc_dist:
            path, dist = _compute_astar_path((t1y,t1x), (t0y,t0x), weights, 
                                                  max_path_length=self.max_px_assoc_dist)
        # cap all distances to thr, A* may be longer than Eucl.
        elif dist >= self.max_px_assoc_dist:
            path = None
        return path

    def _assign_IDs_to_detections(self, cache=None):
        """
        Link the detections to identities over frames. Detections are converted
        to the libmot format (https://github.com/nightmaredimple/libmot) and
        associated using the min cost flow approach. This libmot format is also
        returned. Rows mark frame-axonID pairs, columns X, Y top left bounding
        box anchor, and bbox width and height (4). Height and width are
        controlled by self.axon_box_size.

        Arguments
        ---------

        cache: str
            Whether to save the assigned detection in libmot format to a file or
            use precomputed assignemnts or don't do any caching. Options: {'to',
            'from', None}. Defaults to None, no caching.
        """
        if cache == 'from':
            _IDed_detections = self.from_cache('_IDed_detections')

        else:
            print('\nAssigning axon IDs using min cost flow...', end='')
            dets = self.get_frame_dets('all', None, libmot=True).reset_index().values

            # cap confidence to 1
            if self.MCF_conf_capping_method == 'ceil':
                dets[:, -1][dets[:, -1]>1] = 1
            if self.MCF_conf_capping_method == 'scale_to_max':
                dets[:, -1] /= dets[:, -1].max()

            # setup the tracker model with paramters and a* distances
            astar_dists = self._get_astar_path_distances(self.astar_dets_paths)
            track_model = MinCostFlowTracker(observation_model = observation_model,
                                            transition_model = transition_model, 
                                            feature_model = feature_model,
                                            astar_dists = astar_dists,
                                            dataset_name = self.dataset.name,
                                            max_px_assoc_dist = self.max_px_assoc_dist,
                                            max_conf_cost = self.MCF_max_conf_cost,
                                            vis_sim_weight = self.MCF_vis_sim_weight,
                                            entry_exit_cost = self.MCF_entry_exit_cost, 
                                            min_flow = self.MCF_min_flow,
                                            max_flow = self.MCF_max_flow, 
                                            miss_rate = self.MCF_miss_rate, 
                                            max_num_misses = self.MCF_max_num_misses,
                                            cost_threshold = self.MCF_edge_cost_thr)

            # iterate of the frames and have them processed by tracker model
            for i in range(len(self)):
                print(f'frame {i}/{(len(self)-1)}', end='...', flush=True)
                det = dets[(dets[:, 0] == i), :]
                img = self.get_frame_and_truedets(i)[0]
                track_model.process(boxes=det[:, 2: 6].astype(np.int32),
                                    scores=det[:, 6], image=img[0].numpy(),
                                    frame_idx=i)

            # solve the graph output trajectory is a list [IDs] of list of boxes (in
            # frames) with format frame_idx, box_idx, box_coo (4 values)
            print('Finding trajectories...', end=' ')
            trajectory = track_model.compute_trajectories()
            if not trajectory:
                print('Could not solve the graph for identity association; -> '
                      'no IDed detections. Try narrowing expected identities by'
                      ' updating parameters[`MCF_MIN_FLOW`, `MCF_MAX_FLOW`]. '
                      f'Currently: {self.MCF_min_flow} to {self.MCF_max_flow}.')
                return None

            # iterate IDs and bring them in to more convinient format again
            record = []
            for i, t in enumerate(trajectory):
                # iterate over lifetime timepoints
                for j, box in enumerate(t):
                    record.append([box[0], i, box[2][0], box[2][1], box[2][2], box[2][3]])
            print(f'-> {len(trajectory)} axon IDs. Done.')
            track = np.array(record)
            track = track[np.argsort(track[:, 0])]

            cols = ['FrameId', 'Id', 'X', 'Y', 'Width', 'Height']
            _IDed_detections_libmot = pd.DataFrame(track, columns=cols)
            _IDed_detections_libmot.set_index(['FrameId', 'Id'], inplace=True)
            _IDed_detections = self.libmot_det2det(_IDed_detections_libmot)
            
        if cache == 'to':
            self.to_cache('_IDed_detections', _IDed_detections)
        return _IDed_detections

    def _get_astar_path_distances(self, astar_paths):
        """
        Get the lengths of a set of A* paths. Inputs may be lists, nested lists
        or dictionaries. Returns a np.array where dimensions match the input
        type and shape.

        Arguments
        ---------
        astar_paths: {dict, list}
            The set of A* paths (scipy.sparse.coo_matrix) either within a
            dictionary (for between-detection use case) or lists (or nested
            lists) for general use case.
        """
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
        dists = [np.array(ds) for ds in dists]
        
        # reconstruct dict input format
        if dictinput:
            dists = dict(zip(keys, dists))
        return dists

    def det2libmot_det(self, detection, t, empty_id=False, drop_conf=False, 
                       to_pandas=True):
        """
        Convert the frame t detection format used throughout the class (Axon042:
        (conf, x, y)) to the format libmot solvers expect: Rows mark
        frame-axonID pairs, columns X, Y top left bounding box anchor, and bbox
        width and height (4). Height and width are controlled by
        self.axon_box_size. Returns pd.DataFrame detections in libmot format. 

        Arguments
        ---------
        detection: pd.DataFrame
            Detections at frame t. Index holds Axon names, columns (confidence,
            anchor_x, anchor_y).

        t: int
            The frame index.
        """
        conf, x, y = detection.values.T
        x_topleft = x-self.axon_box_size//2
        y_topleft = y-self.axon_box_size//2
        frame_id = np.full(conf.shape, t)
        boxs = np.full(conf.shape, self.axon_box_size)

        axon_id = np.array([int(idx[-3:]) for idx in detection.index])
        det_libmot = np.stack([frame_id, axon_id, x_topleft, y_topleft, 
                               boxs, boxs, conf]).T

        cols = ['FrameId', 'Id', 'X', 'Y', 'Width', 'Height', 'conf']
        det_libmot = pd.DataFrame(det_libmot, columns=cols)
        return det_libmot.set_index(['FrameId', 'Id'])

    def libmot_det2det(self, IDed_detections_libmot):
        """
        Convert all IDed detections in libmot format back into format used
        throughout the class (Axon042: (conf, x, y)). Returns a list of
        pd.DataFrames of this format where each element matches a frame t.

        Arguments
        ---------
        IDed_detections_libmot: pd.DataFrame
            Detetions across all frames in libmot format.
        """
        # convert from top left to centor coordinate again, slice out hieght+width
        IDed_detections = (IDed_detections_libmot + self.axon_box_size//2).iloc[:,:2]
        
        # get the confidence values lost during ID assignment
        missing_confs = []
        for t in IDed_detections.index.unique(0):
            # get all detection at frame t
            conf, det_x, det_y = self.get_frame_dets('all', t).values.T
            for IDed_det_x, IDed_det_y  in IDed_detections.loc[t].values:
                # For each IDed detection, based on matching anchor values, get the confidence
                conf_match = conf[(IDed_det_x == det_x) & (IDed_det_y == det_y)]
                missing_confs.append(conf_match[0])
        IDed_detections['conf'] = missing_confs

        # Reorder columns and rename the indices back to normal
        IDed_detections = IDed_detections[['conf', 'X', 'Y']]
        IDed_detections.columns = ['conf', 'anchor_x', 'anchor_y']
        IDed_detections_list = []
        for t in range(len(self)):
            if t in IDed_detections.index.unique(0):
                det = IDed_detections.loc[t]
                det.index = [f'Axon_{i:0>3}' for i in det.index]
            else:
                # if frame empty, no detections
                det = pd.DataFrame([])
            IDed_detections_list.append(det.sort_index())
        return IDed_detections_list

    def _agg_all_IDed_dets(self):
        """
        Aggregate all frame-wise IDed detections into one pd.Dataframe.
        DataFrame columns has frame index at level one, and [anchor_x, anchor_y,
        conf] at level two; rows represent axons.
        """
        IDed_dets_all = self.get_frame_dets('IDed', t=None)
        # add temporal frame index
        t_cols = [(i//3, c) for i, c in enumerate(IDed_dets_all.columns)]
        IDed_dets_all.columns = pd.MultiIndex.from_tuples(t_cols)
        # fill up timeopints without detections
        missing_ts = [t for t in range(len(self)) if t not in IDed_dets_all.columns.unique(0)]
        fill_dfs = [pd.DataFrame([], columns=pd.MultiIndex.from_product(([t], 
                    ['anchor_x', 'anchor_y', 'conf']))) for t in missing_ts]
        IDed_dets_all = pd.concat([IDed_dets_all, *fill_dfs], axis=1).sort_index(axis=1)
        IDed_dets_all.index.rename('axonID', inplace=True)
        IDed_dets_all.columns.rename(('frameID', 'detInfo'), inplace=True)
        return IDed_dets_all
        

    def search_MCF_params(self, edge_cost_thr_values = [.4, .6, .7, .8, .9, 1, 1.2, 3], 
                          entry_exit_cost_values = [ .2, .8, .9, 1, 1.1,  2], 
                          miss_rate_values =  [0.9, 0.6], vis_sim_weight_values = [0, 0.1], 
                          conf_capping_method_values = ['ceil' , 'scale_to_max']):
        """
        Test the performance of different hyperparamter combinations on data
        assocication. This functions searches over all possible combinations.

        Arguments
        ---------
        edge_cost_thr_values: [float, ...]
            The maxmimum cost for an edge to still be considered. Low values
            translate to lower detection-confidence thresholds.
        
        entry_exit_cost_values: [float, ...]
            The cost to create or kill an identity through an edge to the source
            and drain node. Lower values translate to lower detection-confidence
            thresholds, and potentially more fragmented trajectories.
        
        miss_rate_values: [float, ...]
            The cost impact of missing detections over gap frames. A low value
            will bias the association to kill and create tracks, a higher value
            will more often link detections even if a gap-frame doesn't contain
            a matching detections.
            
        vis_sim_weight_values: [float, ...]
            To which degree visual similarity should be considered besides
            spatial distance. Computed with histgram matching. Between 0 and 1.
        
        conf_capping_method_values: [str, ...]
            Capping behaviour on confidence values above one. Options: {'ceil',
            'scale_to_max'}. 'Ceil' will set all values above 1 to 1, 'scale'
            will divide all confidence scores by the max confidence.
        """
        def test_params():
            # using MCF params defined above, assign IDs
            self.assign_ids(astar_paths_cache='from')
            pred_dets_IDed = self.get_frame_dets('IDed', t=None, libmot=True)

            # compute MOT benchmark
            acc = mm.utils.compare_to_groundtruth(target_dets_IDed, pred_dets_IDed, 
                                                  dist='euclidean', 
                                                  distth=(self.nms_min_dist)**2)
            mh = mm.metrics.create()
            res = mh.compute(acc, metrics=mm.metrics.motchallenge_metrics,
                             return_dataframe=True).iloc[0]
            param_names = ('edge_cost_thr', 'entry_exit_cost', 'miss_rate', 
                           'vis_sim_weight', 'conf_capping_method')
            param_vals = (self.MCF_edge_cost_thr, self.MCF_entry_exit_cost, 
                          self.MCF_miss_rate, self.MCF_vis_sim_weight, 
                          self.MCF_conf_capping_method)
            return pd.Series(param_vals, param_names).append(res)

        target_dets_IDed = self.get_frame_dets('groundtruth', None, libmot=True)
        
        length = len(edge_cost_thr_values) * len(entry_exit_cost_values) * \
                 len(miss_rate_values) * len(vis_sim_weight_values) * \
                 len(conf_capping_method_values)
        results, i = [], 0
        # search all param combinations 
        for ec in edge_cost_thr_values:
            self.MCF_edge_cost_thr = ec
            for eec in entry_exit_cost_values:
                self.MCF_entry_exit_cost = eec
                for mr in miss_rate_values:
                    self.MCF_miss_rate = mr
                    for vsw in vis_sim_weight_values:
                        self.MCF_vis_sim_weight = vsw
                        for ccm in conf_capping_method_values:
                            self.MCF_conf_capping_method = ccm
                            print(f'{i}/{length}', flush=True)

                            results.append(test_params())
                            print(results[-1])
                            i += 1

        results = pd.concat(results, axis=1).T
        results.to_csv(f'{self.dir}/MCF_params_results.csv')

def _reconstruct_axons(self, ):
    """-- Not implemented --"""
    # indicate what path was interpolated, option to get that version or the one
    # without interpolation 
    all_dets = self.get_frame_dets('IDed', None)
    print(all_dets)
    pass

def get_axon_reconstructions(self, t=None, axon_name=None, include_history=True, 
                             interpolate_missing=True, ymin=0, ymax=0):
    """-- Not implemented --"""
    pass