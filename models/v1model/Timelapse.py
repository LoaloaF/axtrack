import os
import pickle
from copy import copy

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix

from tifffile import imread
from skimage.util import img_as_float32
from skimage.exposure import adjust_log
from skimage.filters import gaussian

import torch
from torch.utils.data import Dataset
from torch.nn import ZeroPad2d

from data_utils import apply_transformations, sprse_scipy2torch

class Timelapse(Dataset):
    def __init__(self, imseq_path, labels_csv, mask_path, timepoints,
                log_correct, standardize_framewise, standardize, name, 
                use_motion_filtered, use_sparse, use_transforms, contrast_llim, 
                plot, pad, Sy, Sx, tilesize, cache, from_cache, temporal_context):
        
        super().__init__()
        self.name = name
        print(f'Data: {name}')
        # load cached object if from_cache true
        self._caching(from_cache=from_cache)
        # irrespective of cache, update these two bois
        self.use_motion_filtered = use_motion_filtered
        self.transform_configs = dict.fromkeys(use_transforms,0)
        
        if not from_cache:
            # agg preprocessing steps data in this dict
            self.plot_data = {}

            # which input data to use
            self.timepoints = timepoints
            self.pad = pad  
            self.use_sparse = use_sparse
            self.use_motion_filtered = use_motion_filtered #'both', 'only', 'exclude'
            self.temporal_context = temporal_context
            self.motion_gaussian_filter_std = 3
            self.motion_lowerlim = .1

            # image augmentation setup (actually performed during training)
            self.transform_configs = dict.fromkeys(use_transforms,0)

            # X: here, still a numpy array, later becomes scipy.sparse
            # read tiff file
            self.imseq, self.mask = self._read_tiff(imseq_path, mask_path, plot)
            
            # define basic attributes of X
            self.sizet = self.imseq.shape[0]
            self.sizey = self.imseq.shape[1]
            self.sizex = self.imseq.shape[2]
            self.size_chnls, self.size_colchnls = self._get_channelsizes() 

            # for formatting the label Y (anchors) in YOLO format
            self.Sy = Sy
            self.Sx = Sx
            self.tilesize = tilesize
            self.xtiles = np.ceil((self.sizex/tilesize)).astype(int).item() 
            self.ytiles = np.ceil((self.sizey/tilesize)).astype(int).item() 

            # clip image values to lower bound
            self._clip_image_values(contrast_llim, plot)

            # log stretch low intensity image data
            self._log_adjust_image(log_correct, plot)

            # X: make self.imseq a list of scipy.sparse coo matrices
            self.imseq = self._sparsify_data()

            # standardize data to mean=1, std=1
            self.stnd_scaler = self._standardize(standardize, standardize_framewise, plot)

            # compute motion by subtracting t-1 from t
            self.p_motion_seq, self.n_motion_seq = self._compute_motion(standardize, plot)

            # Y: format ylabel, target is a pd.DataFrame
            self.target = self._load_bboxes(labels_csv)

            # slice to timeinterval
            self.timepoints_indices, self.sizet, self.target, self.imseq, self.p_motion_seq, self.n_motion_seq = self._slice_timepoints()

            # convert scipy.COO (imseq, p_motion & n_motion) to sparse tensor
            self.X = self._construct_X_tensor()

            # final data to use is constructed before each epoch by calling 
            # construct_tiles(). empty tiles mask for reconstructing image from 
            # filtered (non-empty) tiles
            self.X_tiled, self.target_tiled, self.tile_info = None, None, None

            # save dataset object as .pkl file
            self._caching(cache=cache)

    def __getitem__(self, idx):
        t_idx, tile_idx = self.unfold_idx(idx)
        t_idx = self.timepoints_indices[t_idx]
        tstart, tstop = t_idx-self.temporal_context, t_idx+self.temporal_context+1

        if self.use_motion_filtered == 'include':
            X = self.X_tiled[tstart:tstop, tile_idx].flatten(end_dim=1)
        elif self.use_motion_filtered == 'exclude':
            X = self.X_tiled[tstart:tstop, tile_idx, :1].flatten(end_dim=1)
        elif self.use_motion_filtered == 'only':
            X = self.X_tiled[tstart:tstop, tile_idx, 1:].flatten(end_dim=1)
        
        elif self.use_motion_filtered == 'temp_context':
            X = self.X_tiled[tstart:tstop, tile_idx, :1].flatten(end_dim=1)
        return X, self.target_tiled[t_idx, tile_idx]
    
    def __len__(self):
        assert self.X_tiled is not None, "No tiles yet. Run dataset.construct_tiles() before iterating."
        length = self.sizet * self.tile_info[..., 0].all(-1).sum()
        return length

    def unfold_idx(self, idx):
        t_idx, tile_idx = divmod(idx, self.X_tiled.shape[1])
        return t_idx, tile_idx

    def fold_idx(self, idx):
        t_idx, tile_idx = idx
        folded_idx = t_idx*self.X_tiled.shape[1] + tile_idx
        return folded_idx

    def flat_tile_idx2yx_tile_idx(self, tile_idx):
        folded_idxs = torch.arange(self.ytiles*self.xtiles)
        folded_idxs_2d = folded_idxs.reshape((self.ytiles, self.xtiles))
        
        non_empty_tile_mask = self.tile_info[..., 0].all(-1)
        folded_idxs_non_empty = folded_idxs_2d[non_empty_tile_mask].flatten()
        ycoo, xcoo = divmod(folded_idxs_non_empty[tile_idx].item(), self.xtiles)
        return ycoo, xcoo
    
    def get_stack(self, which, major_index, device=None, X2drawable_img=False):
        X, target = [], []
        if which == 'image':
            n_imagetiles = self.X_tiled.shape[1]
            iter_over = range(n_imagetiles)

        elif which == 'time':
            n_timepoints = self.sizet
            iter_over = range(n_timepoints)

        for minor_idx in iter_over:
            if which == 'image':
                # major index is the timpoint, minor the tile 
                indexer = (major_index, minor_idx)
            elif which == 'time':
                # major index is the tile, minor the timepoint
                indexer = (minor_idx, major_index)

            x, tar = self[self.fold_idx(indexer)]
            if X2drawable_img:
                x = x[self.get_tcenter_idx()]
            X.append(x)
            target.append(tar)
        if device:
            return torch.stack(X, 0).to(device), torch.stack(target, 0).to(device)
        return torch.stack(X, 0), torch.stack(target, 0)
        
    def get_tcenter_idx(self):
        return [list(range(i, i+self.size_colchnls)) for i in range(0, 
                self.size_chnls, self.size_colchnls)][self.temporal_context]

    def _read_tiff(self, path, mask_path, plot):
        print('Loading .tif image...', end='', flush=True)
        imseq = img_as_float32(imread(path))
        imseq -= imseq.min() # usually min is 0, but some datasets are offset
        if mask_path:
            print('masking...', end='', flush=True)
            mask = np.load(mask_path).astype(bool)
            imseq[:,~mask] = 0
        if any(self.pad):
            print('padding...', end='', flush=True)
            top, right, bottom, left = self.pad
            H = imseq.shape[1] + top + bottom
            W = imseq.shape[2] + right + left
            padded = np.zeros((imseq.shape[0], H, W), dtype=np.float32)
            padded[:, top:top+imseq.shape[1], left:left+imseq.shape[2]] = imseq
            padded_mask = np.zeros((H, W), dtype=int)
            padded_mask[top:top+imseq.shape[1], left:left+imseq.shape[2]] = mask
            imseq = padded
            mask = padded_mask
        print('Done.')
        if plot:
            t0 = imseq[self.timepoints[0]].copy()
            tn1 = imseq[self.timepoints[-1]].copy() 
            self.plot_data['Original'] = t0, tn1
        return imseq, mask
        
    def _clip_image_values(self, lower, plot):
        if lower:
            self.imseq[self.imseq<lower] = 0
            print(f'Image clipped to min value: {lower:.4f}')
        if plot:
            t0 = self.imseq[self.timepoints[0]].copy()
            tn1 = self.imseq[self.timepoints[-1]].copy() 
            self.plot_data['Clipped'] = t0, tn1

    def _log_adjust_image(self, log_correct, plot):
        if log_correct:
            print('Log-adjusting image values...', end='', flush=True)
            self.imseq = adjust_log(self.imseq, log_correct)
            print('Done.')
            if plot:
                t0 = self.imseq[self.timepoints[0]].copy()
                tn1 = self.imseq[self.timepoints[-1]].copy() 
                self.plot_data['Log-Adjusted'] = t0, tn1

    def _sparsify_data(self):
        print('Sparsifying data...', end='', flush=True)
        imseq_sparse = []
        sparseness = 0
        for t in range(self.sizet):
            sparseness += (self.imseq[t]==0).sum()/(self.sizey*self.sizex)
            imseq_sparse.append(coo_matrix(self.imseq[t]))
        print(f' Avg sparseness: {sparseness/self.sizet:.3f} - Done.')
        return imseq_sparse
    
    def _standardize(self, standardize, standardize_framewise, plot):
        stnd_scalar = (None, None)
        if standardize[0]:
            print(standardize, standardize_framewise)
            print(f'Standardizing image values ({standardize[0]})...', end='', flush=True)
            # use passed scaler (eg train scaler for standardizing test data)
            # only applies for non-framewise standardization
            if standardize[1] is not None:
                var_scalars, mean_scalars = standardize[1]
                print('using passed scalers...', end='')
            elif standardize[0] == 'zscore':
                mean_scalars = [np.mean(frame.data) for frame in self.imseq]
                var_scalars = [np.std(frame.data) for frame in self.imseq]
            elif standardize[0] == '0to1':
                mean_scalars = np.zeros(self.sizet)
                var_scalars = [np.max(frame.data) for frame in self.imseq]

            # one for all, collapsed std and mean
            if not standardize_framewise:
                var_scalar = np.mean(var_scalars) if standardize == 'zscore' else np.max(var_scalars)
                mean_scalar = np.mean(mean_scalars)
                stnd_scalar = (standardize[0], (var_scalar, mean_scalar))
            else:
                stnd_scalar = (standardize[0], None)

            for t in range(self.sizet):
                frame_dense = self.imseq[t].todense()
                zeros = frame_dense == 0

                # framewise
                if standardize_framewise:
                    var_scalar = var_scalars[t]
                    # mean_scalar = mean_scalars[t]
                
                # to_mean = 1 if standardize[0] == 'zscore' else 0
                # frame_dense = (frame_dense -mean_scalar)/var_scalar +to_mean
                frame_dense = frame_dense/var_scalar
                frame_dense[zeros] = 0
                self.imseq[t] = coo_matrix(frame_dense, dtype=np.float32)

            if plot:
                t0 = np.array(self.imseq[self.timepoints[0]].todense())
                tn1 = np.array(self.imseq[self.timepoints[-1]].todense())
                lbl = f'Standardized (frame-wize: {standardize_framewise})'
                self.plot_data[lbl] = t0, tn1
            print('Done.')
        return stnd_scalar

    def _compute_motion(self, standardize, plot):
        blur_strength = self.motion_gaussian_filter_std
        lowerlim = self.motion_lowerlim
        print(f'Calculating motion (clip lower lim: {lowerlim}) + '
              f'Gaussian filtering (std: {blur_strength})...', end='', flush=True)

        # first timepoint has no t-1, set to all 0. Recommend to not use t=0
        pos_motion_seq = [coo_matrix((self.sizey, self.sizex))]
        neg_motion_seq = [coo_matrix((self.sizey, self.sizex))]
        for t in range(1, self.sizet):
            motion_frame = self.imseq[t] - self.imseq[t-1]
            motion_frame = gaussian(motion_frame.todense(), blur_strength)
            motion_frame[np.abs(motion_frame)<lowerlim] = 0

            pos_motion = np.where(motion_frame>0, motion_frame, 0)
            neg_motion = np.where(motion_frame<0, motion_frame*-1, 0)
            pos_motion_seq.append(coo_matrix(pos_motion, dtype=np.float32))
            neg_motion_seq.append(coo_matrix(neg_motion, dtype=np.float32))
        print('Done.')
            
        if standardize:
            # motion data is always normed with a global scalar
            print('Standardizing motion values...', end='', flush=True)
            pos_scaler = np.mean([np.std(f.data) for f in pos_motion_seq[1:]])
            neg_scaler = np.mean([np.std(f.data) for f in neg_motion_seq[1:]])
            
            pos_motion_seq = [f/pos_scaler for f in pos_motion_seq]
            neg_motion_seq = [f/neg_scaler for f in neg_motion_seq]
            print('Done.')

        if plot:
            t0 = np.array(pos_motion_seq[self.timepoints[0]].todense())
            tn1 = np.array(pos_motion_seq[self.timepoints[-1]].todense())
            self.plot_data['Positive Motion'] = t0, tn1
            t0 = np.array(neg_motion_seq[self.timepoints[0]].todense())
            tn1 = np.array(neg_motion_seq[self.timepoints[-1]].todense())
            self.plot_data['Negative Motion'] = t0, tn1
        return pos_motion_seq, neg_motion_seq
        
    def _load_bboxes(self, labels_csv):
        if labels_csv is None:
            # for unlabelled data, create an empty dummy pd.DataFrame
            cols = pd.MultiIndex.from_product([['Ax-00'],['anchor_y','anchor_x']], 
                                               names=('axon','prop'))
            bboxes = pd.DataFrame([], index=range(self.sizet), columns=cols)
        else:
            bboxes = pd.read_csv(labels_csv, index_col=0, header=[0,1])
            bboxinfo_cols = ['anchor_x', 'anchor_y']
            bboxes = bboxes.loc[:, (slice(None), bboxinfo_cols)].sort_index()
            bboxes = bboxes.reset_index(drop=True)
            if self.pad[0] or self.pad[3]:
                for col in bboxinfo_cols:
                    bboxes.loc[:, (slice(None), col)] += self.pad[0] if 'y' in col else self.pad[3]
        return bboxes

    def _slice_timepoints(self):
        print(f'Slicing timepoints from t=[0...{self.sizet}] to t=' \
              f'{self.timepoints} (n={len(self.timepoints)})')
        tps = self.timepoints
        if self.temporal_context:
            # get neighbouring timepoints 
            tps = [[t-tpad,t,t+tpad] for t in self.timepoints for tpad in range(1,self.temporal_context+1)]
            # unpack list and drop duplicates
            tps = sorted(list(set([t for tp in tps for t in tp])))
        # temporal padding makes things a bit messy... the list below holds the
        # timepoints to actually feed into the model. This will be equal to
        # self.timepoints if the passed timepoints are continuous, eg (2,3,4,5)
        # in contrast to 2,3,34,35. For these cases below is needed for indexing
        timepoints_indices = [tps.index(tp) for tp in self.timepoints]

        imseq = [self.imseq[t] for t in tps]
        p_motion_seq = [self.p_motion_seq[t] for t in tps]
        n_motion_seq = [self.n_motion_seq[t] for t in tps]
        target = self.target.iloc[tps]
        sizet = len(self.timepoints)
        return timepoints_indices, sizet, target, imseq, p_motion_seq, n_motion_seq
    
    def _get_channelsizes(self):
        if self.use_motion_filtered == 'exclude':
            ncol_chnls = 1
        elif self.use_motion_filtered == 'only':
            ncol_chnls = 2
        elif self.use_motion_filtered == 'include':
            ncol_chnls = 3
        col_block_size = self.temporal_context*2 +1
        nchnls = col_block_size * ncol_chnls
        return nchnls, ncol_chnls
    
    def _construct_X_tensor(self):
        print('Constructing sparse Tensor...', end='', flush=True)
        imseq = torch.stack([sprse_scipy2torch(coo_frame) for coo_frame in self.imseq])
        p_mot_seq = torch.stack([sprse_scipy2torch(coo_frame) for coo_frame in self.p_motion_seq])
        n_mot_seq = torch.stack([sprse_scipy2torch(coo_frame) for coo_frame in self.n_motion_seq])
        X = torch.stack([imseq, p_mot_seq, n_mot_seq], 1)
        print('Done.')
        return X

    def _caching(self, cache=False, from_cache=False):
        if from_cache:
            print('Loading data from cache.\n')
            dataset_file = f'{from_cache}/{self.name}_dataset_cached.pkl'
            assert os.path.exists(dataset_file), f'\n\nNo cached dataset found: {dataset_file}'
            with open(dataset_file, 'rb') as file:
                cached_object = pickle.load(file)
                [self.__setattr__(n, v) for n, v in cached_object.items()]

        elif cache:
            with open(f'{cache}/{self.name}_dataset_cached.pkl', 'wb') as file:
                print('Serializing datset for caching...', end='')
                pickle.dump(self.__dict__, file, protocol=4)
                print('Done.\n\n')
        
    def tiled_target2yolo_format(self, target_tiled):
        # yolo target switches dim order from y-x to x-y!
        # dims: ytile x xtile x timepoint x yolo_x_grid x yolo_y_grid x label(conf, x_anchor, y_anchor)
        yolo_target = torch.zeros((*target_tiled.shape[:-2], self.Sx, self.Sy, 4))

        # scale x and y anchors to tilesize (0-1)
        target_tiled = target_tiled/self.tilesize
        # get the indices of each dimension where a label exists
        target_tiled_idx = torch.where(target_tiled >= 0)
        # construct a sparse tensor using these indices - useful because the 
        # number of axons per frame differs
        vals = target_tiled[target_tiled_idx]
        target_tiled = torch.sparse_coo_tensor(torch.stack(target_tiled_idx), 
                                               vals, size=target_tiled.shape)

        # coordinate here is in range 0-1 wrt the tile (512x512 px) - unpack
        y_anchors = target_tiled[..., 0].coalesce()
        x_anchors = target_tiled[..., 1].coalesce()
        # get the tile-specific coordinates that have positive labels
        ytile_coo, xtile_coo, t_idx, axID_idx = y_anchors.indices()
        
        # this converts the values of the sparse tensors x_anchors, y_anchors
        # from 0-1 to 0-S (0-12)
        yolo_y = (self.Sy*y_anchors).values()
        # the integer component of this variable gives the yolo box the label 
        # belongs to (0-11)
        yolo_y_box = yolo_y.to(int)
        # the float component gives the position relative to the yolo-box size (0-1)
        yolo_y_withinbox = yolo_y - yolo_y_box
        # same for x
        yolo_x = (self.Sx*x_anchors).values()
        yolo_x_box = yolo_x.to(int)
        yolo_x_withinbox = yolo_x - yolo_x_box
        
        # now finally set the values at the respective coordinates
        yolo_target[ytile_coo, xtile_coo, t_idx, yolo_x_box, yolo_y_box, 0] = 1
        yolo_target[ytile_coo, xtile_coo, t_idx, yolo_x_box, yolo_y_box, 1] = yolo_x_withinbox
        yolo_target[ytile_coo, xtile_coo, t_idx, yolo_x_box, yolo_y_box, 2] = yolo_y_withinbox
        yolo_target[ytile_coo, xtile_coo, t_idx, yolo_x_box, yolo_y_box, 3] = axID_idx.to(torch.float32)
        return yolo_target

    def construct_tiles(self, device, force_no_transformation=False, nolabel=False):
        # # lazy mode
        # if not torch.cuda.is_available():
        #     self.tile_info = torch.load(f'/home/loaloa/.cache/axtrack/{self.name}_tileinfo.pth')
        #     self.X_tiled = torch.load(f'/home/loaloa/.cache/axtrack/{self.name}_X_tiled.pth')
        #     self.target_tiled = torch.load(f'/home/loaloa/.cache/axtrack/{self.name}_target_tiled.pth')
        #     return
        if self.transform_configs and not force_no_transformation:
            X, target = apply_transformations(self.transform_configs, self.X, 
                                              self.target, self.sizey, 
                                              self.sizex, device)
        else:
            X, target = self.X.to_dense(), self.target
        print(f'Tiling data...', end='', flush=True)

        # split X into tiles using .split function
        ts = self.tilesize
        tile_list = [tile for col in torch.split(X, ts, -2) 
                     for tile in torch.split(col, ts, -1)]

        # splitting the target is a bit more work. see below
        target = target.fillna(-1).astype(int, copy=False).swaplevel(0,1,1)
        anchor_y = target.anchor_y.values.astype(np.int16)
        anchor_x = target.anchor_x.values.astype(np.int16)

        
        # iterate over the tiles that got unpacked, alos build an aray that 
        # stores the number of positive labels and non-empty tiles
        # no_label = anchor_x == -1
        tile_labels = []
        for tile_i in range(len(tile_list)):
            # X
            tile_shape = np.array(list(tile_list[tile_i].shape[-2:]))
            ypad_needed, xpad_needed = ts - tile_shape
            if ypad_needed or xpad_needed:
                padder = ZeroPad2d((0, xpad_needed, 0, ypad_needed))
                tile_list[tile_i] = padder(tile_list[tile_i])

            # target
            ytile_coo, xtile_coo = divmod(tile_i, self.xtiles)
            in_tile_row = (anchor_y >= ytile_coo*ts) & (anchor_y < (ytile_coo+1)*ts)
            in_tile_col = (anchor_x >= xtile_coo*ts) & (anchor_x < (xtile_coo+1)*ts)
            in_tile = in_tile_row & in_tile_col
            y_anchor_filtd = np.where(in_tile, anchor_y-ytile_coo*ts, -1)
            x_anchor_filtd = np.where(in_tile, anchor_x-xtile_coo*ts, -1)
            tile_labels.append(np.stack([y_anchor_filtd, x_anchor_filtd], -1))

        X = torch.stack(tile_list)
        target = np.stack(tile_labels)

        # dims: ytile x xtile x timepoint x colorchannels x tile_ysize x tile_xsize
        X = X.reshape((self.ytiles, self.xtiles, *X.shape[ 1:]))
        # dims: ytile x xtile x timepoint x axonID x anchor_yx
        target = target.reshape(self.ytiles, self.xtiles, *tile_labels[0].shape)

        # next transform the labels to YOLO format
        target_tiled = self.tiled_target2yolo_format(torch.from_numpy(target))

        # get an array that indicates empty tiles and n pos labels 
        non_empty_tiles = (X[:,:,:, 0] > 0).any(-1).any(-1)
        n_pos_labels = target_tiled[..., 0].sum((-1,-2))
        self.tile_info = torch.stack([non_empty_tiles, n_pos_labels], -1)

        # collapse the x- & y tile coordinates into one flat dim that has only 
        # those tiles that are not empty across all timepoints
        # Finally swap the first two dims to have (timepoints, tile, ...)
        self.X_tiled = X[non_empty_tiles.all(-1)].swapaxes(0,1)
        self.target_tiled = target_tiled[non_empty_tiles.all(-1)].swapaxes(0,1)

        # if not torch.cuda.is_available():
        #     # lazy mode
        #     torch.save(self.tile_info, f'/home/loaloa/.cache/axtrack/{self.name}_tileinfo.pth')
        #     torch.save(self.X_tiled, f'/home/loaloa/.cache/axtrack/{self.name}_X_tiled.pth')
        #     torch.save(self.target_tiled, f'/home/loaloa/.cache/axtrack/{self.name}_target_tiled.pth')
        print('Done', flush=True) 

    def stitch_tiles(self, pd_tiled_det, img_tiled=None, reset_index=False):
        ts = self.tilesize
        # get the yx coordinates of the tiles  
        tile_coos = torch.tensor([self.flat_tile_idx2yx_tile_idx(tile) 
                                  for tile in range(len(pd_tiled_det))])

        # make an empty img, then iter over tile grid 
        img = torch.zeros((self.size_colchnls, self.sizey, self.sizex)) if img_tiled is not None else None
        pd_det = [tile_det.copy() for tile_det in pd_tiled_det]
        for tile_ycoo in range(self.ytiles):
            for tile_xcoo in range(self.xtiles):
                non_empty_tile = (tile_coos[:,0] == tile_ycoo) & (tile_coos[:,1] == tile_xcoo)
                flat_tile_idx = torch.where(non_empty_tile)[0]

                if len(flat_tile_idx) == 1:
                    # when there is a nonempty tile, place its values in img_new
                    yslice = slice(ts*tile_ycoo, ts*(tile_ycoo+1))
                    xslice = slice(ts*tile_xcoo, ts*(tile_xcoo+1))
                    if img is not None:
                        # these values are always =ts, unless at the edge (needs pad)
                        ts_x, ts_y = img[:, yslice, xslice].shape[1:]
                        img[:, yslice, xslice] = img_tiled[flat_tile_idx, self.get_tcenter_idx(), :ts_x, :ts_y]

                    # and transform the anchor coordinates into img coordinates
                    pd_det[flat_tile_idx].anchor_y += tile_ycoo*ts
                    pd_det[flat_tile_idx].anchor_x += tile_xcoo*ts
                    
        pd_det = pd.concat(pd_det)
        if reset_index:
            pd_det.index = [f'Axon_{i:0>3}' for i in range(len(pd_det))]
        return pd_det, img

    def get_target_in_libmot_format(self, axon_boxs):
        libmot_targets = []
        for t in range(self.sizet):
            target = self.target.iloc[self.timepoints_indices[t]].unstack().dropna()
            x, y = target.values.T
            x_topleft = target.anchor_x-axon_boxs//2
            y_topleft = target.anchor_y-axon_boxs//2
            frame_id = np.full(target.shape[0], t)
            axon_id = np.array([int(idx[-3:]) for idx in target.index])
            boxs = np.full(target.shape[0], axon_boxs)

            libmot_target = np.stack([frame_id, axon_id, x_topleft, y_topleft, 
                                    boxs, boxs]).T.astype(int)
            libmot_target = pd.DataFrame(libmot_target, columns=['FrameId', 'Id', 'X', 'Y', 'Width', 'Height'])
            libmot_targets.append(libmot_target.set_index(['FrameId', 'Id']))
        return pd.concat(libmot_targets)