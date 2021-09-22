import os
import pickle
from copy import deepcopy

from concurrent.futures import ThreadPoolExecutor
from itertools import repeat

import numpy as np
import pandas as pd
import torch

import cv2
from scipy import sparse
# from scipy.optimize import linear_sum_assignment

import pyastar2d

from libmot.data_association  import MinCostFlowTracker
import motmetrics as mm



class AxonDetections():
    def __init__(self, model, dataset, parameters, directory=None, 
                 detect_at_init=True, assign_id_at_init=False, calc_target_dist_at_init=False):
        
        self.model = model
        self.dataset = dataset 
        self.P = parameters
        self.dir = directory
        os.makedirs(self.dir, exist_ok=True) if self.dir is not None else None
        
        self.device = parameters['DEVICE']
        self.Sx = parameters['SX']
        self.Sy = parameters['SY']
        self.tilesize = parameters['TILESIZE']
        self.device = parameters['DEVICE']
        self.num_workers = parameters['NUM_WORKERS']
        self.conf_thr = parameters['BBOX_THRESHOLD']

        self.nms_min_dist = parameters.get('NON_MAX_SUPRESSION_DIST')
        if not self.nms_min_dist:
            self.nms_min_dist = 18
        
        self.max_px_assoc_dist = 500
        self.alpha = 0.015
        self.beta = 3
        self.axon_box_size = 70
        self.max_num_misses = 2

        if detect_at_init:
            out = self.detect_dataset()
            self.yolo_targets, self.yolo_dets, self.pandas_tiled_dets = out[:3]
            self.pandas_image_dets, self.detections = out[3:]

        if assign_id_at_init:
            self.dets_astar_paths = self._compute_detections_astar_paths(cache='from')
            self.all_detections = self._assign_IDs_to_detections()
        
        if calc_target_dist_at_init:
            self.target_astar_paths = self._compute_dets_path_to_target(cache='to')
            dists = self._get_astar_path_distances(self.target_astar_paths)


    def __len__(self):
        return self.dataset.sizet

    def detect_dataset(self):
        self.dataset.construct_tiles(self.device, force_no_transformation=True)

        yolo_targets = []
        yolo_dets = []
        pandas_tiled_dets = []
        pandas_image_dets = []
        detections = []
        for t in range(len(self)):
            print(f'detecting {t}', end='...')
            # get a stack of tiles that makes up a frame
            # batch dim of X corresbonds to all tiles in frame at time t
            X, yolo_target = self.dataset.get_stack('image', t, self.device)
            
            # use the model to detect axons, output here follows target.shape
            yolo_det = self.model.detect_axons(X)

            # non max supression on the inital yolo output 
            yolo_det_nms = self.non_max_supress_yolo_det(yolo_det)

            # convert the tensor to a list of tiles of type pd.DataFrames
            pandas_tiled_det = self._yolo_Y2pandas_det(yolo_det_nms)
            
            # stitch the list of pandas tiles to one pd.DataFrame
            pandas_image_det, _ = self.dataset.stitch_tiles(pandas_tiled_det, reset_index=True)

            # set conf of strongly overlapping detections to 0
            pandas_image_det_nms = self._non_max_supression_pandas(pandas_image_det)
        
            # save all intermediate detection formats
            yolo_targets.append(yolo_target)
            yolo_dets.append(yolo_det_nms)
            pandas_tiled_dets.append(pandas_tiled_det)
            pandas_image_dets.append(pandas_image_det)
            detections.append(pandas_image_det_nms)
        return yolo_targets, yolo_dets, pandas_tiled_dets, pandas_image_dets, detections

    def update_conf_thr_to_best_F1(self):
        cnfs_mtrx = np.zeros((3,51))
        for t in range(len(self)):
            cnfs_mtrx += self.compute_TP_FP_FN(self.yolo_dets[t], self.yolo_targets[t])
        prc_rcl_f1 = self.compute_prc_rcl_F1(cnfs_mtrx)
        self.conf_thr = 0.5 + 0.1*np.argmax(prc_rcl_f1[2])

    def get_det_image_and_target(self, t, return_tiled=True):
        img_tiled, _ = self.dataset.get_stack('image', t, self.device, 
                                              X2drawable_img=False)
        pd_tiled_true_det = self._yolo_Y2pandas_det(self.yolo_targets[t])
        pd_image_true_det, img = self.dataset.stitch_tiles(pd_tiled_true_det, img_tiled)
        if return_tiled:
            return img_tiled, img, pd_tiled_true_det, pd_image_true_det
        return img, pd_image_true_det
    
    def get_detection_metrics(self, t, return_all_conf_thrs=False):
        # for labelled dataset compute prediction metrics 
        cnfs_mtrx = self.compute_TP_FP_FN(self.yolo_dets[t], self.yolo_targets[t])
        prc_rcl_f1 = self.compute_prc_rcl_F1(cnfs_mtrx)
        if not return_all_conf_thrs:
            conf_thresholds = np.arange(0.5, 1.01, 0.01)
            idx = np.argmin(np.abs(conf_thresholds-self.conf_thr))
            return prc_rcl_f1[:, idx]
        return prc_rcl_f1

    def get_FP_FN_detections(self, t):
        FP_mask, FN_mask = self.compute_TP_FP_FN(self.yolo_dets[t], 
                                                 self.yolo_targets[t], 
                                                 return_FP_FN_mask=True)

        FP_dets, FN_dets = self.yolo_dets[t].clone(), self.yolo_dets[t].clone()
        conf_mask = torch.tensor((True,False,False))

        # index mask (FP/TP) and column mask (just select confidence column)
        FP_dets[FP_mask, conf_mask] = 1
        FP_dets[~FP_mask, conf_mask] = 0
        FN_dets[FN_mask, conf_mask] = 1
        FN_dets[~FN_mask, conf_mask] = 0

        FP_dets = self._yolo_Y2pandas_det(FP_dets)
        FP_dets, _ = self.dataset.stitch_tiles(FP_dets)
        FN_dets = self._yolo_Y2pandas_det(FN_dets)
        FN_dets, _ = self.dataset.stitch_tiles(FN_dets)
        return FP_dets, FN_dets


        


    def _yolo_coo2tile_coo(self, yolo_Y):
        # nobatch dim
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







    def _filter_yolo_det(self, yolo_Y):
        # convert the target tensor to shape (batchsize, self.Sx*self.Sy, 3)
        batch_size, labelsize = yolo_Y.shape[0], yolo_Y.shape[-1]
        yolo_Y = yolo_Y.reshape((batch_size, -1, labelsize))
        # split batch dim (tiles) into list because of variable sizes 
        det_list = list(torch.tensor_split(yolo_Y, batch_size))

        # filter each example
        return [tile_det[tile_det[:,:,0] >= self.conf_thr] for tile_det in det_list]

    def _torch2pandas(self, tile_det):
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

    def _yolo_Y2pandas_det(self, yolo_Y):
        # convert the YOLO coordinates (0-1) to image coordinates (tilesize x tilesize) (inplace function)
        yolo_Y_imgcoo = self._yolo_coo2tile_coo(yolo_Y.clone())

        # filter the (batch, self.Sx, self.Sy, 3) output into a list of length batchsize,
        # where the list elements are tensors of shape (#_boxes_above_thr x 3)
        # each element in the list represents a frame/tile
        det_list = self._filter_yolo_det(yolo_Y_imgcoo)
        # convert each frame from tensor to pandas
        for i in range(len(det_list)):
            det_list[i] = self._torch2pandas(det_list[i])
        return det_list

    def _non_max_supression_pandas(self, pd_img_det):
        # order by confidence
        pd_img_det = pd_img_det.sort_values('conf', ascending=False)
        i = 0
        # get most confident anchor first
        while i < pd_img_det.shape[0]:
            xs, ys = pd_img_det[['anchor_x', 'anchor_y']].values.T
            x,y = pd_img_det.iloc[i, 1:]

            # compute distances of that anchor to all others
            dists = np.sqrt(((xs-x)**2 + (ys-y)**2).astype(int))
            # check where its < thr, skip the first one (dist. with itself = 0)
            non_max_indices = np.where(dists<self.nms_min_dist)[0][1:]
            # drop those anchors
            pd_img_det.drop(pd_img_det.index[non_max_indices], inplace=True)
            i += 1

        axon_lbls = [f'Axon_{i:0>3}' for i in range(pd_img_det.shape[0])]
        pd_img_det.index = axon_lbls
        return pd_img_det



    def non_max_supress_yolo_det(self, yolo_det):
        batchs = yolo_det.shape[0]
        # take the raw flat model output and reshape to standard yolo target shape
        yolo_det = yolo_det.reshape((batchs, self.Sx, self.Sy, 3))
        # convert coordinates to tile coordinates 
        pred_tilecoo = self._yolo_coo2tile_coo(yolo_det.clone())

        # iterate the tiles/ batch
        for tile_i in range(batchs):
            # index to the tile and flatten the SxSy dim (12x12) -> 144 anchors
            tile_det = pred_tilecoo[tile_i]
            tile_det_flat = tile_det.flatten(0,1)

            # get the indices where the anchor confidence is at least above 0.5
            pos_pred_idx = torch.where(tile_det_flat[:,0] > 0.5)[0]
            # sort these by their associated confidences 
            pos_pred_idx = pos_pred_idx[tile_det_flat[pos_pred_idx, 0].argsort(descending=True)]
            
            i = 0
            # iterate the conf-sorted anchor indices that were above 0.5
            while i < len(pos_pred_idx):
                # get the coordianates of all the anchors above 0.5
                xs, ys = tile_det_flat[pos_pred_idx, 1:].T
                # and the specifc anchor being tested in this iteration
                x,y = tile_det_flat[pos_pred_idx[i], 1:]

                # compute euclidian distances of that anchor to all others
                dists = torch.sqrt(((xs-x)**2 + (ys-y)**2))
                    
                # check where it is < thr, skip the first one (dist. with itself = 0)
                non_max_indices = torch.where(dists<self.nms_min_dist)[0][1:]
                # if any of the conf-above-0.5 anchors was smaller than distance:
                if len(non_max_indices):
                    # set the confidence at those indices to 0
                    tile_det_flat[pos_pred_idx[non_max_indices], 0] = 0
                    # remove those anchor indices, they shouldn't be checked after being dropped
                    pos_pred_idx = torch.tensor([pos_pred_idx[idx] for idx in range(len(pos_pred_idx)) if idx not in non_max_indices])
                i += 1

            # undo the flattening and set the tile-coo confidences to the yolo box confs.
            tile_det = tile_det.reshape(self.Sx, self.Sy, 3)
            yolo_det[tile_i, :, :, 0] = tile_det.reshape(self.Sx, self.Sy, 3)[:, :, 0]
        return yolo_det
        
    def compute_TP_FP_FN(self, yolo_det, yolo_true_det, return_FP_FN_mask=False):
        obj_exists = yolo_true_det[..., 0].to(bool)
        pred_obj_exists = yolo_det[..., 0]
        
        TP_masks, FP_masks, FN_masks = [], [], []
        conf_thresholds = np.arange(0.5, 1.01, 0.01)
        for thr in conf_thresholds:
            pos_pred = torch.where(pred_obj_exists>thr, 1, 0).to(bool)
            TP_masks.append((pos_pred & obj_exists))
            FP_masks.append((pos_pred & ~obj_exists))
            FN_masks.append((~pos_pred & obj_exists))
        
        TP = [tp.sum().item() for tp in TP_masks]
        FP = [fp.sum().item() for fp in FP_masks]
        FN = [fn.sum().item() for fn in FN_masks]
        cnfs_mtrx = np.array([TP, FP, FN])
        if return_FP_FN_mask:
            # select the mask corresbonding to the threshold closest to self.conf_thr
            idx = np.argmin(np.abs(conf_thresholds-self.conf_thr))
            return FP_masks[idx], FN_masks[idx]
        return cnfs_mtrx

    def compute_prc_rcl_F1(self, confus_mtrx, epoch=None, which_data=None):
        prc = confus_mtrx[0] / (confus_mtrx[0]+confus_mtrx[1]+0.000001)
        rcl = confus_mtrx[0] / (confus_mtrx[0]+confus_mtrx[2]+0.000001)
        f1 =  2*(prc*rcl)/(prc+rcl)
        metric = np.array([prc, rcl, f1]).round(3)

        if epoch is not None and which_data is not None:
            conf_thresholds = np.arange(0.5, 1.01, 0.01)
            index = pd.MultiIndex.from_product([[epoch],[which_data],conf_thresholds])
            return pd.DataFrame(metric, columns=index, index=('precision', 'recall', 'F1'))
        return metric







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

    def _compute_astar_path(self, source, target, weights, return_dist=True):
        path_coo = pyastar2d.astar_path(weights, source, target)
        ones, row, cols = np.ones(path_coo.shape[0]), path_coo[:,0], path_coo[:,1]
        path = sparse.coo_matrix((ones, (row, cols)), shape=weights.shape, dtype=bool)
        if return_dist:
            return path, path_coo.shape[0]
        return path

    def _get_path_between_dets(self, t1_det, t0_det, weights):
        t1id, t1y, t1x = t1_det
        t0id, t0y, t0x = t0_det
        
        # compute euclidean distance 
        dist = np.sqrt((t1y-t0y)**2 + (t1x-t0x)**2)
        # only if it's lower than thr, put in the work to compute A* paths
        if dist < self.max_px_assoc_dist:
            path, dist = self._compute_astar_path((t1y,t1x), (t0y,t0x), weights)
        # cap all distances to thr, A* may be longer than Eucl.
        if dist >= self.max_px_assoc_dist:
            path = None
        return path
            
    def _compute_detections_astar_paths(self, cache='to'):
        cache_fname = f'{self.dir}/{self.dataset.name}_astar_dets_paths.pkl'
        if cache == 'from':
            print('Getting A* between-detection paths from cached file...', end='')
            with open(cache_fname, 'rb') as file:
                dets_astar_paths = pickle.load(file)

        else:
            print('Computing A* detection paths between detections...', end='')
            # make the cost matrix for finding the astar path
            weights = np.where(self.dataset.mask==1, 1, 2**16).astype(np.float32)

            dets_astar_paths = {}
            for t in range(len(self)):
                lbl = f'{self.dataset.name}_t:{t:0>3}'
                print('\n')
                
                # get current detections
                t_dets = self.detections[t]
                # ids (placeholders)
                t_ids = [int(idx[-3:]) for idx in t_dets.index]
                t_dets = np.stack([t_ids, t_dets.anchor_y, t_dets.anchor_x], -1)
                
                # predecessor timestamps
                for t_bef in range(t-1, t-(self.max_num_misses+1), -1):
                    if t_bef < 0:
                        continue
                    lbl_t_before = f'{lbl}-t:{t_bef:0>3}'
                    print(lbl_t_before, end=': ')
                    
                    # get detections of a timestep before
                    t_bef_dets = self.detections[t_bef]
                    t_bef_ids = [int(idx[-3:]) for idx in t_bef_dets.index]
                    t_bef_dets = np.stack([t_bef_ids, t_bef_dets.anchor_y, t_bef_dets.anchor_x], -1)

                    # iterate predecessor detections, compute its distance to all detections at t
                    astar_paths = []
                    for i, t_bef_det in enumerate(t_bef_dets):
                        print(f'{i+1}/{len(t_bef_dets)}', end='...')

                        with ThreadPoolExecutor(self.num_workers) as executer:
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

    def _compute_dets_path_to_target(self, cache='to'):
        cache_fname = f'{self.dir}/{self.dataset.name}_astar_target_paths.pkl'
        if cache == 'from':
            print('Getting A* target paths from cache file...', end='')
            with open(cache_fname, 'rb') as file:
                target_astar_paths = pickle.load(file)
            
        else:
            print('Computing A* paths to target...', end='')
            weights = np.where(self.dataset.mask, 1, 2**16).astype(np.float32)
            target_yx = self.dataset.structure_outputchannel_coo
            
            target_astar_paths = []
            for t in range(len(self)):
                print(f't:{t:0>3}', end='...')
                yx = self.detections[t].loc[:, ['anchor_y', 'anchor_x']].values

                with ThreadPoolExecutor(self.num_workers) as executer:
                    args = repeat(target_yx), yx, repeat(weights), repeat(False)
                    as_paths = executer.map(self._compute_astar_path, *args)
                target_astar_paths.append([p for p in as_paths])
            
            if cache == 'to':
                print('caching...', end='')
                with open(cache_fname, 'wb') as file:
                    pickle.dump(target_astar_paths, file)
        print('Done.')
        return target_astar_paths


    def get_dists_to_target(self, window_size=3):
        all_dists = self._get_astar_path_distances(self.target_astar_paths)
        all_dists = [pd.Series(all_dists[t], index=self.detections[t].index, name=t) for t in range(len(self))]
        all_dists = pd.concat(all_dists, axis=1)

        # get the initial timepoints distance
        init_dist = all_dists.groupby(level=0).apply(lambda row: row.dropna(axis=1).iloc[:,0])
        # subtract that distance
        all_dists = all_dists.apply(lambda col: col-init_dist.values)

        # smooth over window_size timpoints
        all_dists = all_dists.rolling(window_size, axis=1, center=True).mean()
        return all_dists
        
    def get_dets_in_libmot_format(self):
        libmot_dets = []
        for t in range(len(self)):
            conf, x, y = self.detections[t].values.T
            x_topleft = x-self.axon_box_size//2
            y_topleft = y-self.axon_box_size//2
            frame_id = np.full(conf.shape, t)
            empty_id = np.full(conf.shape, -1)
            boxs = np.full(conf.shape, self.axon_box_size)

            libmot_dets.append(np.stack([frame_id, empty_id, x_topleft, y_topleft, 
                                         boxs, boxs, conf]).T)
        return np.concatenate(libmot_dets)





    def _assign_IDs_to_detections(self):#, data, model, params, run_dir):
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
            print(scores)
            scores = (np.log(scores/(1-scores)))
            scores[scores>max_conf_cost] = max_conf_cost
            scores[scores<-max_conf_cost] = -max_conf_cost
            print(scores)
            print(np.log(-0.446 * np.array(kwargs['scores']).astype(np.float) + 0.456))
            return scores
            return np.log(-0.446 * np.array(kwargs['scores']).astype(np.float) + 0.456)

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
            print(image.shape)
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
            
            costs = -np.log((1-vis_sim_weight) * distances*(miss_rate ** (time_gap-1)) \
                            + vis_sim_weight * vis_sim \
                            + 1e-5)
            costs[inf_dist] = np.inf
            return costs


        def make_video(tracks):
            from matplotlib.animation import ArtistAnimation
            from matplotlib.patches import Rectangle
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots()
            boxs = 70
            framewise_artists = []
            for frame in range(len(self)):

                img, gt = self.get_det_image_and_target(frame, return_tiled=False)
                im = ax.imshow(img[0].numpy(), animated=True)

                boxes = tracks[tracks[:,0] == frame]
                recs = []
                # for box in boxes:
                for axon_id, (conf, x, y) in self.detections[frame].iterrows():

                    conf = min(1, conf)
                    ax_id_col = plt.cm.get_cmap('hsv', 20)(int(axon_id[-3:])%20)
                    # text_artists.append(ax.text(x-boxs/2, y-boxs/1.5, axon_id.replace('on_',''), 
                    #                     fontsize=5.5, color=ax_id_col))
                    axon_box = Rectangle((x-boxs/2, y-boxs/2), boxs, boxs, facecolor='none', edgecolor=ax_id_col)
                    recs.append(axon_box)


                    
                    # ID, x, y, width, height = box[1:]
                    # col = plt.cm.get_cmap('hsv', 10)(ID%10)
                    # recs.append(Rectangle((x,y), width, height, edgecolor=col, facecolor='none'))
                [ax.add_patch(rec) for rec in recs]
                t = ax.text(0,0, frame, zorder=1)
                
                framewise_artists.append([im, *recs, t])
            an = ArtistAnimation(fig, framewise_artists, blit=True, interval=1000)
            plt.show()

        print('Assigning axon IDs ...', end='')
        # setup parameters 
        
        # default = {'entry_exit_cost': 1, 'thresh': 2.8,
        default = {'entry_exit_cost': 2.5, 'thresh': .4,
                'miss_rate': 0.8, 'max_num_misses': 1}
        min_flow = 1
        max_flow = 120
        max_conf_cost = 4.6
        vis_sim_weight = 0.1
        # conf_capping_method = 'scale_to_max'   # 'scale_to_max'
        conf_capping_method = 'ceil'   # 'scale_to_max'
        
        entry_exit_cost = default['entry_exit_cost']
        miss_rate = default['miss_rate']
        max_num_misses = default['max_num_misses']
        thresh = default['thresh']
        
        
        # get data (distances and detections in libmot format)
        astar_dists = self._get_astar_path_distances(self.dets_astar_paths)
        dets = self.get_dets_in_libmot_format()
        print(dets)
        print(dets.shape)

        # cap confidence to 1
        if conf_capping_method == 'ceil':
            dets[:, -1][dets[:, -1]>1] = 1
        if conf_capping_method == 'scale_to_max':
            dets[:, -1] /= dets[:, -1].max()


        # do the association using min cost flow
        record = []
        track_model = MinCostFlowTracker(observation_model=observation_model,
                                        transition_model=transition_model, feature_model=feature_model,
                                        entry_exit_cost=entry_exit_cost, min_flow=min_flow,
                                        max_flow=max_flow, miss_rate=miss_rate, max_num_misses=max_num_misses,
                                        cost_threshold=thresh)

        for i in range(len(self)):
            print('Processing frame: ', i, end='...')
            det = dets[(dets[:, 0] == i), :]

            img, gt = self.get_det_image_and_target(i, return_tiled=False)
            track_model.process(boxes=det[:, 2: 6].astype(np.int32),
                                scores=det[:, 6], image=img[0].numpy(),
                                frame_idx=i)
            print('Done.\n\n\n')

        trajectory = track_model.compute_trajectories()
        # trajectory is a list [IDs] of list of boxes (in frames) with format
        # frame_idx, box_idx, box_coo (4 values)
        for i, t in enumerate(trajectory):
            for j, box in enumerate(t):
                record.append([box[0], i, box[2][0], box[2][1], box[2][2], box[2][3]])
        track = np.array(record)
        track = track[np.argsort(track[:, 0])]

        IDed_detections = pd.DataFrame(track, columns=['FrameId', 'Id', 'X', 'Y', 'Width', 'Height'])
        IDed_detections = IDed_detections.set_index(['FrameId', 'Id'])

        # finally update self.detections with correct IDs
        boxs = self.axon_box_size
        for t in range(len(self)):
            # get detections at frame t, old ones and ones with ID
            det = self.detections[t]
            IDed_det = IDed_detections.loc[t, ['X','Y']]
            valid_det = []
            # iter old detections (longer the ones ones usually)
            for _, (conf, anchor_x, anchor_y) in det.iterrows():
                # get the assigned ID by masking for matching box coordinates
                ID = IDed_det.index[(IDed_det.values==(anchor_x-boxs/2, anchor_y-boxs/2)).all(1)]
                # if the box coordinates where still in the IDed detections, add
                if not ID.empty:
                    valid_det.append(pd.Series([conf, anchor_x, anchor_y], 
                                               name=f'Axon_{ID[0]:0>3}', 
                                               index=('conf', 'anchor_x', 'anchor_y')))
            self.detections[t] = pd.concat(valid_det, axis=1).T
        return IDed_detections

        # all_detections = []
        # for frame_idx in range(len(self)):
        #     frame_track_dets = track[track[:,0]==frame_idx]
        #     print()
        #     print()
        #     print()
        #     print(self.detections[frame_idx])
        #     frame_dets = dets[dets[:,0]==frame_idx]
        #     print(frame_dets)
        #     print(frame_dets.shape)
        #     print()
        #     print(frame_track_dets)
        #     print(frame_track_dets.shape)

        #     order = [np.where((frame_track_dets[:, [2,3]] == xy).all(1))[0] for xy in frame_dets[:, [2,3]]]
        #     order = np.array([i[0] for i in order if len(i)])
        #     print(order)





        #     ids = [f'Axon_{int(ID):0>3}' for ID in track[track[:,0]==frame_idx][:, 1]]
        #     all_dets = self.detections[frame_idx].iloc[order].copy()
        #     all_dets.index = ids
        #     all_dets.columns = pd.MultiIndex.from_product([[frame_idx],['conf', 'anchor_x', 'anchor_y']])
        #     print(all_dets)

        #     self.detections[frame_idx] = all_dets[frame_idx]
        #     all_detections.append(all_dets)
        #     # exit()
        # print(pd.concat(all_detections, axis=1))
        # print(self.detections)
        # make_video(track)
        # return pd.concat(all_detections, axis=1)
        # # return track







































        # print(astar_dists['test_t:001-t:000'])
        # print(len(astar_dists['test_t:001-t:000']))
        # print(len(astar_dists['test_t:001-t:000'][0]))

        # astar_dists_libmot = {}
        # for t in range(len(astar_dists)):
        #     for t_bef in range(len(astar_dists[t])):
        #         dists = np.array(astar_dists[t][t_bef]).T
        #         astar_dists_libmot.update({f't{t:0>3}-t_before{t-(t_bef+1):0>3}': dists})

        # [print(key, astar.shape) for key,astar in astar_dists_libmot.items()]
        # pickle.dump(astar_dists_libmot, open('libmot/axon_test_dists.pkl', 'wb'))
        # exit()




        # cntd_costs = self._compute_cntn_costs_from_dists()

        # # glue cost and detection to one dataframe (lin. assignm method below expects this)
        # cntd_costs_pd_detections = []
        # for t in range(len(self)):
        #     if t == 0:
        #         det_t0 = self.detections[t]
        #         cntd_costs_pd_detections.append(det_t0)
        #         continue
        #     det_t1 = self.detections[t]
        #     cntd_cost_pd = pd.DataFrame(cntd_costs[t-1], index=det_t1.index, columns=det_t0.index)
        #     cntd_costs_pd_detections.append(pd.concat([det_t1, cntd_cost_pd], axis=1))
        #     det_t0 = det_t1
        #     print(cntd_costs_pd_detections)

        # assigned_ids = self._hungarian_ID_assignment(cntd_costs_pd_detections)

        # all_detections = []
        # for t in range(len(self)):
        #     self.detections[t].index = assigned_ids[t].values

        #     all_detections.append(self.detections[t].copy())
        #     all_detections[-1].columns = pd.MultiIndex.from_product([[t],['conf', 'anchor_x', 'anchor_y']])
        # all_detections = pd.concat(all_detections, axis=1)
        # print('Done.')
        # return all_detections





    # # ID ASSIGNMENT - to be redone....
    
    # def _cntn_id_cost(self, t1_conf, t0_conf, dist):
    #     return ((self.alpha*dist) * (1/(1 + np.e**(self.beta*(t1_conf+t0_conf-1))))).astype(float)

    # def _create_id_cost(self, conf):
    #     return 1/(1 + np.e**(self.beta*(conf-.5))).astype(float)

    # def _generate_axon_id(self, ):
    #     axon_id = 0
    #     while True:
    #         yield axon_id
    #         axon_id += 1

    # def _hungarian_ID_assignment(self, cntd_cost_detections):
    #     t0_conf = pd.DataFrame([])
    #     axon_id_generator = self._generate_axon_id()

    #     # iterate detections: [(conf+anchor coo) + distances to prv frame dets]
    #     assigned_ids = []
    #     for t, cntn_c_dets in enumerate(cntd_cost_detections):
    #         # unpack confidences of current t 
    #         t1_conf = cntn_c_dets.iloc[:,0]
    #         # each detections will have a new ID assigned to it, create empty here
    #         ass_ids = pd.Series(-1, index=cntn_c_dets.index, name=t)
            
    #         if t == 0:
    #             # at t=0 every detections is a new one, special case
    #             ass_ids[:] = [next(axon_id_generator) for _ in range(len(ass_ids))]
    #         else:
    #             # t1 and t0 have different lengths, save shorter n of the two
    #             nt1 = cntn_c_dets.shape[0]
    #             ndet_min = min((nt1, cntn_c_dets.shape[1]-3))

    #             # get the costs for creating t1 ID and killing t0 ID (dist indep.)
    #             create_t1_c = self._create_id_cost(t1_conf).values
    #             kill_t0_c = self._create_id_cost(t0_conf).values
                
    #             # get the costs for continuing IDs
    #             cntn_c_matrix = cntn_c_dets.iloc[:,3:].values
    #             # hungarian algorithm
    #             t1_idx_cntn, t0_idx_cntn = linear_sum_assignment(cntn_c_matrix)
    #             # index costs to assignments, associating t1 with t0, dim == ndet_min
    #             cntn_t1_c = cntn_c_matrix[t1_idx_cntn, t0_idx_cntn]

    #             # an ID is only NOT continued if the two associated detections both 
    #             # have lower costs on their own (kill/create)
    #             cntn_t1_mask = (cntn_t1_c<create_t1_c[:ndet_min]) | (cntn_t1_c<kill_t0_c[:ndet_min])
    #             # pad t1 mask to length of t1 (in case t1 was shorter than t0)
    #             cntn_t1_mask = np.pad(cntn_t1_mask, (0, nt1-ndet_min), constant_values=False)
    #             created_t1_mask = ~cntn_t1_mask

    #             # created IDs simply get a new ID from the generator
    #             ass_ids[created_t1_mask] = [next(axon_id_generator) for _ in range(created_t1_mask.sum())]
    #             # for cntn ID, mask the t0_idx (from hung. alg.) and use it as 
    #             # indexer of previous assignments timpoint
    #             assoc_id = t0_idx_cntn[cntn_t1_mask[:ndet_min]]
    #             ass_ids[cntn_t1_mask] = assigned_ids[-1].iloc[assoc_id].values

    #         t0_conf = t1_conf
    #         assigned_ids.append(ass_ids)
    #     assigned_ids = pd.concat(assigned_ids, axis=1).stack().swaplevel().sort_index()
    #     assigned_ids = assigned_ids.apply(lambda ass_id: f'Axon_{int(ass_id):0>3}')
    #     return assigned_ids

    # def _compute_cntn_costs_from_dists(self):
    #     t1t0_cntd_costs = []
    #     astar_dists = self._get_astar_path_distances(self.dets_astar_paths)
    #     for t in range(len(self)):
    #         if t == 0:
    #             t0_conf = self.detections[t].conf
    #             continue
    #         t1_conf = self.detections[t].conf
    #         # get the distance between t0 and t1 detections 
    #         t1_t0_dists_matrix = astar_dists[t-1]

    #         # t1 detection confidence vector to column-matrix
    #         t1_conf_matrix = np.tile(t1_conf, (len(t0_conf),1)).T
    #         # t0 detection confidence vector to row-matrix
    #         t0_conf_matrix = np.tile(t0_conf, (len(t1_conf),1))
            
    #         # pass distances and anchor conf to calculate assoc. cost between detections
    #         t1t0_cntd_costs.append(self._cntn_id_cost(t1_conf_matrix, t0_conf_matrix, 
    #                                                   t1_t0_dists_matrix))
    #         t0_conf = t1_conf
    #     return t1t0_cntd_costs
