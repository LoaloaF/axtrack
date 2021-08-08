import os
import pickle
import time

from config import RAW_DATA_DIR, OUTPUT_DIR, CODE_DIR, PYTORCH_DIR
from pt_utils import rotate_indices, flip_indices, coo2tensor, gen_denseTensorChunk
from plotting import draw_tile
from utils import Y2pandas_anchors

import numpy as np
import pandas as pd
from tifffile import imread
from skimage.util import img_as_float32
from skimage.exposure import adjust_log
from skimage.filters import gaussian

from scipy.sparse import coo_matrix
from scipy.sparse import load_npz
from scipy.sparse import save_npz

import matplotlib.pyplot as plt
import seaborn as sns

import torch
from torch.utils.data import Dataset
from torch.nn import ZeroPad2d

import torchvision.transforms.functional as TF
from skimage.transform import rotate, AffineTransform, warp


class Timelapse(Dataset):
    def __init__(self, imseq_path=None, labels_csv=None, mask_path=None, timepoints=None,
                log_correct=True, retain_temporal_var=True, standardize=True, name='', use_motion_filtered='only', 
                use_sparse=False, use_transforms = ['vflip', 'hflip', 'rot', 'translateY', 'translateX'],
                contrast_llim=43/2**16, plot=True, pad=[0,300,0,300], Sy=12, Sx=12, tilesize=512,
                cache=False, from_cache=False, temporal_context=2):
        
        super().__init__()
        self.name = name
        print(f'Data: {name}')
        # load cached object if from_cache true
        self._caching(from_cache=from_cache)
        # irrespective of cache, update these two bois
        self.use_motion_filtered = use_motion_filtered
        self.transform_configs = dict.fromkeys(use_transforms,0)

        # self.t_center, self.ncolorchannels = self._get_center_time_index()
        # self.imseq, self.mask = self._read_tiff(imseq_path, mask_path, plot)
        # self.ID_assoc = self._construct_axon_identity_association_matrix()

        if not from_cache:
            # agg preprocessing steps data in this dict
            self.plot_data = {}

            # read tiff file
            self.timepoints = timepoints
            self.pad = pad
            # X: here, still a numpy array, later becomes scipy.sparse
            self.imseq, self.mask = self._read_tiff(imseq_path, mask_path, plot)

            # which input data to use
            self.use_sparse = use_sparse
            self.use_motion_filtered = use_motion_filtered #, 'only', 'exclude'
            self.motion_gaussian_filter_std = 3
            self.motion_lowerlim = .1
            self.temporal_context = temporal_context
            
            # image augmentation setup (actually performed only when training)
            self.transform_configs = dict.fromkeys(use_transforms,0)

            # define basic attributes
            self.sizet = self.imseq.shape[0]
            self.sizey = self.imseq.shape[1]
            self.sizex = self.imseq.shape[2]
            self.t_center, self.ncolorchannels = self._get_center_time_index()

            # for formatting the label Y (bboxes) in YOLO format
            self.Sy = Sy
            self.Sx = Sx
            self.tilesize = tilesize
            self.xtiles = np.ceil((self.sizex/tilesize)).astype(int).item() 
            self.ytiles = np.ceil((self.sizey/tilesize)).astype(int).item() 

            # clip image values to lower bound
            self._clip_image_values(contrast_llim, plot)

            # log stretch image data
            self._log_adjust_image(log_correct, plot)

            # X: make self.imseq a list of scipy.sparse coo matrices
            self.imseq = self._sparsify_data()

            # standardize data to mean=1, std=1
            self.scaler = self._standardize(standardize, retain_temporal_var, plot)

            # compute motion by subtracting t-1 from t
            self.p_motion_seq, self.n_motion_seq = self._compute_motion(standardize, plot)

            # Y: format ylabel, target is a pd.DataFrame
            self.target = self._load_bboxes(labels_csv)

            # slice to timeinterval
            self.sizet, self.target, self.imseq, self.p_motion_seq, self.n_motion_seq = self._slice_timepoints()

            # convert scipy.COO (imseq, p_motion & n_motion) to sparse tensor
            self.X = self._construct_X_tensor()
            self.ID_assoc = self._construct_axon_identity_association_matrix()

            # final data to use is constructed before each epoch by calling 
            # construct_tiles(). empty tiles mask for reconstructing image from 
            # filtered (non-empty) tiles
            self.X_tiled, self.target_tiled, self.tile_info = None, None, None

            # save dataset object as .pkl file
            self._caching(cache=cache)

    def __getitem__(self, idx):
        t_idx, tile_idx = self.unfold_idx(idx)
        # print('Getting t: ', t_idx, end='')
        t_idx += 2
        tstart, tstop = t_idx-2, t_idx+3
        # print('   but really t, tstart, tstop: ', t_idx, tstart, tstop)

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
    
    def get_stack(self, which, major_index):
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
            X.append(x)
            target.append(tar)
        return torch.stack(X, 0), torch.stack(target, 0)
        

    def _read_tiff(self, path, mask_path, plot):
        print('Loading .tif image...', end='', flush=True)
        imseq = img_as_float32(imread(path))
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
    
    def _standardize(self, standardize, retain_temporal_var, plot):
        scaler = None
        if standardize:
            print('Standardizing image values...', end='', flush=True)
            std = [np.std(frame.data) for frame in self.imseq]
            mean = [np.mean(frame.data) for frame in self.imseq]

            for t in range(self.sizet):
                frame_dense = self.imseq[t].todense()
                zeros = frame_dense == 0

                # framewise
                if not retain_temporal_var:
                    s = std[t]
                    m = mean[t]
                # one for all, collapsed std and mean
                else:
                    s = np.mean(std)
                    m = np.mean(mean)

                frame_dense = (frame_dense -m)/s +1
                frame_dense[zeros] = 0
                self.imseq[t] = coo_matrix(frame_dense, dtype=np.float32)
            scaler = (std, mean) if not retain_temporal_var else np.mean(std), np.mean(mean)

            if plot:
                t0 = np.array(self.imseq[self.timepoints[0]].todense())
                tn1 = np.array(self.imseq[self.timepoints[-1]].todense())
                lbl = f'Standardized (retain temp. var: {retain_temporal_var})'
                self.plot_data[lbl] = t0, tn1
            print('Done.')
        return scaler

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
        bboxes = pd.read_csv(labels_csv, index_col=0, header=[0,1])
        bboxinfo_cols = ['anchor_x', 'anchor_y']
        # bboxinfo_cols = ['extend', 'anchor_x', 'anchor_y', 'topleft_x', 
        #                 'topleft_y', 'bottomright_x', 'bottomright_y']
        bboxes = bboxes.loc[:, (slice(None), bboxinfo_cols)].sort_index()
        bboxes = bboxes.reset_index(drop=True)
        if self.pad[0] or self.pad[3]:
            for col in bboxinfo_cols:
                bboxes.loc[:, (slice(None), col)] += self.pad[0] if 'y' in col else self.pad[3]
        return bboxes

    def _slice_timepoints(self):
        print(f'Slicing timepoints from t=[0...{self.sizet}] to t=' \
              f'{self.timepoints} (n={len(self.timepoints)})')
        t0, tn1 = self.timepoints[0], self.timepoints[-1]
        padded_tps = [t0-2, t0-1, *self.timepoints, tn1+1, tn1+2]
        imseq = [self.imseq[t] for t in padded_tps]
        p_motion_seq = [self.p_motion_seq[t] for t in padded_tps]
        n_motion_seq = [self.n_motion_seq[t] for t in padded_tps]
        target = self.target.iloc[padded_tps]
        sizet = len(self.timepoints)
        return sizet, target, imseq, p_motion_seq, n_motion_seq
    
    def _get_center_time_index(self):
        if self.use_motion_filtered == 'exclude':
            t_center = slice(2,3)
            ncchannels = 1
        elif self.use_motion_filtered == 'only':
            t_center = slice(4,6)
            ncchannels = 2
        elif self.use_motion_filtered == 'include':
            t_center = slice(6,9)
            ncchannels = 3
        return t_center, ncchannels
    
    def _construct_X_tensor(self):
        print('Constructing sparse Tensor...', end='', flush=True)
        imseq = torch.stack([coo2tensor(coo_frame) for coo_frame in self.imseq])
        p_mot_seq = torch.stack([coo2tensor(coo_frame) for coo_frame in self.p_motion_seq])
        n_mot_seq = torch.stack([coo2tensor(coo_frame) for coo_frame in self.n_motion_seq])
        X = torch.stack([imseq, p_mot_seq, n_mot_seq], 1)
        print('Done.')
        return X

    def _construct_axon_identity_association_matrix(self):
        IDs = self.target.swaplevel(0,1,1).anchor_x.fillna(0).astype(bool).values
        IDs = IDs[self.temporal_context:self.sizet+self.temporal_context]
        # pad virtual t0-1 timepoint, and tlast+1 with zeros
        IDs = np.pad(IDs, ((1,1),(0,0)), 'constant', constant_values=(0,0))

        id_capacity = IDs.shape[1]
        filler_block = np.zeros((id_capacity,id_capacity))
        ID_assoc = []
        for t in range(self.sizet+1):
            t0, t1 = IDs[t:t+2]
            ID_continues = t0 & t1
            ID_killed = t0 & ~t1
            ID_created = ~t0 & t1
            ID_t0isNone = ~(ID_continues|ID_killed)
            ID_t1isNone = ~(ID_continues|ID_created)

            ID_continues = np.diagflat(ID_continues)
            ID_killed = np.diagflat(ID_killed)
            ID_created = np.diagflat(ID_created)
            ID_t0isNone = np.diagflat(ID_t0isNone)
            ID_t1isNone = np.diagflat(ID_t1isNone)

            t0t1_assoc = np.block([[ID_continues, ID_killed,    ID_t0isNone],
                                   [ID_created,   filler_block, filler_block],
                                   [ID_t1isNone,  filler_block, filler_block]])
            ID_assoc.append(t0t1_assoc)
        return np.stack(ID_assoc)
            
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
                pickle.dump(self.__dict__, file)
                print('Done.\n\n')
    
    def construct_tiles(self, device):
        if self.transform_configs:
            X, target = self.apply_transformations(device)
            print(f'Tiling data...', end='', flush=True)
        elif self.X_tiled is None:
            print(f'Tiling data...', end='', flush=True)
            X = self.X.to_dense()
            target = self.target
        else:
            print('Using tiled data from previous Epoch.', flush=True)
            return

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
        print('Done', flush=True) 

    def apply_transformations(self, device):
        def transform_X(X, tchunksize, angle, flip_dims, dy, dx):
            # translating
            if dx or dy:
                X_coal = X.coalesce()
                indices, vals = X_coal.indices(), X_coal.values()
                oof = torch.ones(vals.shape, dtype=bool) # ~out of frame

                if dy:
                    indices[2] = (indices[2] + dy).to(torch.long)
                    oof[(indices[2]>=self.sizey) | (indices[2]<0)] = False
                if dx:
                    indices[3] = (indices[3] + dx).to(torch.long)
                    oof[(indices[3]>=self.sizex) | (indices[3]<0)] = False
                
                # torch doens't broadcast like numpy?
                oof_brdcst = torch.stack([oof]*4)
                indices = indices[oof_brdcst].reshape((4, oof.sum()))
                X = torch.sparse_coo_tensor(indices, vals[oof], X.shape)
            
            if not flip_dims and not angle:
                return X.to('cpu').to_dense()
            else:
                X_dense = []
                for X_chunk in gen_denseTensorChunk(X, tchunksize):
                   
                    # flipping
                    if flip_dims:
                        X_chunk = torch.flip(X_chunk, flip_dims)
                    # rotating
                    if angle:
                        X_chunk = TF.rotate(X_chunk, angle)
                    X_dense.append(X_chunk.to('cpu'))

                return torch.cat(X_dense)

        def transform_Y(target, angle, flip_dims,  dy, dx):
            # make a new dataframe with reset index (y-axis from 0 - last timepoint)
            target_transf = target.copy()
            # get the anchor points of all the bboxes (center point)
            anchors = target.loc[:,(slice(None),['anchor_y', 'anchor_x'])].sort_index(1).copy()

            # translating
            if dy or dx:
                y_anchor = anchors.loc[:, (slice(None), 'anchor_y')]
                x_anchor = anchors.loc[:, (slice(None), 'anchor_x')]
                if dy:
                    y_anchor += dy
                    oof = ((1 >= y_anchor) | (y_anchor >= self.sizey-1)).values
                    y_anchor = y_anchor.where(~oof)
                if dx:
                    x_anchor += dx
                    oof = ((1 >= x_anchor) | (x_anchor >= self.sizex-1)).values
                    x_anchor = x_anchor.where(~oof)
                # ----- check which bbox is lost after translation? -----
                anchors.loc[:, (slice(None), 'anchor_y')] = y_anchor
                anchors.loc[:, (slice(None), 'anchor_x')] = x_anchor

            # flipping
            if flip_dims:
                anchors = flip_indices(anchors, (self.sizey, self.sizex), flip_dims)
            
            # rotating
            if self.transform_configs.get('rot',0) > .5:
                # ----- check which bbox is lost after rotation? -----
                anchors = rotate_indices(anchors, (self.sizey, self.sizex), angle)
            
            # get the new x,y anchors and mask extend in case bboxes got lost in rotation
            y_anchor = anchors.loc[:, (slice(None), 'anchor_y')].round()
            x_anchor = anchors.loc[:, (slice(None), 'anchor_x')].round()
            target_transf.loc[:, (slice(None),'anchor_y')] = y_anchor
            target_transf.loc[:, (slice(None),'anchor_x')] = x_anchor
            return target_transf

        # make a transformation config: rotate, hflip, vflip translate - or not
        self.transform_configs = {key: round(torch.rand(1).item(), 2)
                                  for key in self.transform_configs}
        print(f'New transform config set: {self.transform_configs}\n'
               'Transforming data...', end='', flush=True)

        # translating parameter
        dy, dx = 0, 0
        # dy and dx are in [-.25, +.25]
        if self.transform_configs.get('translateY', 0) >.5:
            dy = round(self.sizey *(self.transform_configs.get('translateY', 0)-.75))
        if self.transform_configs.get('translateX', 0) >.5:
            dx = round(self.sizex *(self.transform_configs.get('translateY', 0)-.75))

        # flipping parameter
        flip_dims = []
        if self.transform_configs.get('hflip',0) > .5:
            flip_dims.append(2)
        if self.transform_configs.get('vflip',0) > .5:
            flip_dims.append(3)

        # rotating parameter
        angle = None
        if self.transform_configs.get('rot',0) > .5:
            angle = ((self.transform_configs['rot'] * 40) -20)
        
        # apply the transformation using paramters above, do on GPU, return on CPU
        tchunksize = 2
        X = transform_X(self.X.to(device), tchunksize, angle, flip_dims, dy, dx)
        # now transform the bboxes with the same paramters
        target_transf = transform_Y(self.target, angle, flip_dims, dy, dx)
        print('Done.')
        torch.cuda.empty_cache()
        return X, target_transf
    
    def tiled_target2yolo_format(self, target_tiled):
        # yolo target switches dim order from y-x to x-y!
        # dims: ytile x xtile x timepoint x yolo_x_grid x yolo_y_grid x label(conf, x_anchor, y_anchor)
        yolo_target = torch.zeros((*target_tiled.shape[:-2], self.Sx, self.Sy, 4))

        # scale x and y anchors to tilesize (0-1)
        target_tiled = target_tiled/self.tilesize
        # get the idices of each dimension where a label exists
        target_tiled_idx = torch.where(target_tiled >= 0)
        # construct a sparse tensor using these indices - useful because the 
        # number of axons per differs
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

    def stitch_tiles(self, img, target, pred_target=None):
        ts = self.tilesize
        # get the yx coordinates of the tiles  
        tile_coos = torch.tensor([self.flat_tile_idx2yx_tile_idx(tile) 
                                  for tile in range(img.shape[0])])

        # make an empty img, then iter of tile grid 
        img_new = torch.zeros((1, self.ncolorchannels, self.sizey, self.sizex))
        target_new, pred_target_new = [], []
        for tile_ycoo in range(self.ytiles):
            for tile_xcoo in range(self.xtiles):
                non_empty_tile = (tile_coos[:,0] == tile_ycoo) & (tile_coos[:,1] == tile_xcoo)
                flat_tile_idx = torch.where(non_empty_tile)[0]

                if flat_tile_idx.any():
                    # when there is a nonempty tile, place its values in img_new
                    yslice = slice(ts*tile_ycoo, ts*(tile_ycoo+1))
                    xslice = slice(ts*tile_xcoo, ts*(tile_xcoo+1))
                    img_new[0, :, yslice, xslice] = img[flat_tile_idx, self.t_center]

                    # and transform the anchor coordinates into img coordinates
                    target[flat_tile_idx].anchor_y += tile_ycoo*ts
                    target[flat_tile_idx].anchor_x += tile_xcoo*ts
                    target_new.append(target[flat_tile_idx])
                    if pred_target is not None:
                        pred_target[flat_tile_idx].anchor_y += tile_ycoo*ts
                        pred_target[flat_tile_idx].anchor_x += tile_xcoo*ts
                        pred_target_new.append(pred_target[flat_tile_idx])
        
        target_new = pd.concat(target_new)
        if pred_target:
            pred_target_new = pd.concat(pred_target_new, ignore_index=True)
        return img_new, target_new, pred_target_new
