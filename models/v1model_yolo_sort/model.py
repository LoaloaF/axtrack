import torch
import torch.nn as nn

import pandas as pd
import numpy as np
from scipy.optimize import linear_sum_assignment

import matplotlib.pyplot as plt
from matplotlib import animation

from utils import Y2pandas_anchors, compute_astar_distances, generate_axon_id
from plotting import draw_frame

from torchvision import datasets, models, transforms

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(CNNBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels, out_channels, bias=True, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.LeakyReLU = nn.LeakyReLU(.1)
    
    def forward(self, x):
        conv_out = self.conv(x)
        # print(f'CNN in size: {x.shape}, \n\t\tout: {conv_out.shape}\n')
        return self.LeakyReLU(self.batchnorm(conv_out)) 

class CNNFeature_collector(nn.Module):
    def __init__(self, architecture, in_c):
        super(CNNFeature_collector, self).__init__()
        
        self.feature_blocks = []
        for layer in range(len(architecture)):
            out_c = architecture[layer][1]
            fblock = CNNBlock(in_channels = in_c, 
                              out_channels = out_c, 
                              padding = (1,1),
                              kernel_size = architecture[layer][0], 
                              stride = architecture[layer][2])
            self.feature_blocks.append(fblock)
            in_c = out_c
        self.out_channels = out_c

    def forward(self, x):
        features = []
        for block in self.feature_blocks:
            x = block(x)
            features.append(x.flatten(1))
        return torch.cat(features, 1)

class YOLO_AXTrack(nn.Module):
    def __init__(self, initial_in_channels, architecture_config, img_dim, Sy, Sx, which_pretrained=None, **kwargs):
        super(YOLO_AXTrack, self).__init__()
        self.architecture = architecture_config
        self.Sy = Sy
        self.Sx = Sx
        self.B = 1
        self.initial_in_channels = initial_in_channels
        self.tilesize = img_dim[0]

        if which_pretrained in ['mobilenet', 'alexnet', 'resnet']:
            self.PreConvNet = self._from_pretrained(self.initial_in_channels, which_pretrained)
            self.PostConvNet = None
            self.get_CNN_outdim()
        else:
            # create preCNN
            self.PreConvNet, out_channels = self._create_PreConvNet(self.architecture[0],
                                                                    self.initial_in_channels)
            # create postCNN
            self.PostConvNet = CNNFeature_collector(self.architecture[1], 
                                                    out_channels)
        # create FC
        self.fcs = self._create_fcs()
    
    def to_device(self, device):
        self.to(device)
        if self.PostConvNet is not None:
            [fb.to(device) for fb in self.PostConvNet.feature_blocks]

    def get_CNN_outdim(self):
        x = torch.zeros((1, self.initial_in_channels, self.tilesize, self.tilesize))
        x = self.PreConvNet(x)
        if self.PostConvNet is not None:
            cnn_features = self.PostConvNet(x)
        else:
            cnn_features = x.flatten(start_dim=1)
        return cnn_features.shape[1]

    def forward(self, x):
        x = self.PreConvNet(x)
        if self.PostConvNet is not None:
            cnn_features = self.PostConvNet(x)
        else:
            cnn_features = x.flatten(start_dim=1)
        return self.fcs(cnn_features)

    def _from_pretrained(self, in_c, which_pretrained):
        if which_pretrained == 'mobilenet':
            out_c = 16
            padding = (1,1)
            model = models.mobilenet_v3_small(True)
            features = model.features
        elif which_pretrained == 'alexnet':
            out_c = 64
            padding = (2,2)
            model = models.alexnet(pretrained=True)
            features = model.features
        elif which_pretrained == 'resnet':
            out_c = 64
            padding = (2,2)
            model = models.resnet18(pretrained=True)
            features = [model.conv1, model.bn1, model.relu, model.maxpool, 
                        model.layer1, model.layer2, model.layer3, model.layer4]

        ConvNet = nn.Sequential()
        for i, module in enumerate(features):
            if i == 0:
                module = nn.Conv2d(in_c, out_c, bias=True, kernel_size=(3,3), 
                                   stride=(2,2), padding=padding)
            if i == 3:
                module.stride = (2,2)
            if i == 6:
                module.padding = (2,2)
            ConvNet.add_module(f'block_{i}', module)
        return ConvNet
    
    def _create_PreConvNet(self, architecture, in_c):
        PreConvNet = nn.Sequential()
        for layer in range(len(architecture)):
            if architecture[layer] != 'M':
                groups = architecture[layer][3]
                out_c = architecture[layer][1]
                block = CNNBlock(in_channels = in_c, 
                                 out_channels = out_c, 
                                 padding = (1,1),
                                 groups = groups,
                                 kernel_size = architecture[layer][0], 
                                 stride = architecture[layer][2])
                in_c = out_c
            else:
                block = nn.MaxPool2d(2,2)
            PreConvNet.add_module(f'PreConvBlock_{layer}', block)
        return PreConvNet, out_c

    def _create_fcs(self, FC_size=1024):
        features_in = self.get_CNN_outdim()
        features_out = self.Sy * self.Sx *self.B*3
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(features_in, FC_size),
            # nn.LeakyReLU(),
            nn.Sigmoid(),
            # nn.Dropout(.15),
            nn.Linear(FC_size, FC_size),
            # nn.LeakyReLU(),
            nn.Sigmoid(),
            nn.Linear(FC_size, features_out),
        )

    def detect_axons_in_frame(self, X):
        self.eval()
        n_tiles = X.shape[0]
        with torch.no_grad():
            pred = self(X).reshape(n_tiles, self.Sx, self.Sy, -1)
        self.train()
        return pred
    
    def get_detections_ids(self, dets, t, run_dir, data_name):
        assigned_ids = pd.read_csv(f'{run_dir}/{data_name}_assigned_ids.csv', index_col=(0,1))
        dets.index = assigned_ids.loc[(t, dets.index),:].values[:,0]
        return dets

    def get_frame_detections(self, data, t, device, anchor_conf_thr, assign_ids=None):
        # get a stack of tiles that makes up a frame
        # batch dim of X corresbonds to all tiles in frame at time t
        X, target = data.get_stack('image', t)
        X = X.to(device)
        # use the model to detect axons, output here follows target.shape
        pred_target = self.detect_axons_in_frame(X)

        # convert the detected anchors in YOLO format to normal ones
        pred_dets = Y2pandas_anchors(pred_target, self.Sx, self.Sy, 
                                     self.tilesize, device, anchor_conf_thr) 
        if target is not None:
            true_dets = Y2pandas_anchors(target.to(device), self.Sx, self.Sy, 
                                         self.tilesize, device, 1) 

        # stitch the prediction from tile format back to frame format
        X_stch, pred_dets_stch, true_dets_stch = data.stitch_tiles(X, pred_dets, true_dets)
        if assign_ids:
            run_dir, data_name = assign_ids
            pred_dets_stch = self.get_detections_ids(pred_dets_stch, t, run_dir, data_name)
        return X,X_stch, pred_dets,pred_dets_stch, true_dets,true_dets_stch


    def predict_all(self, data, params, dest_dir=None, show=False, 
                    save_single_tiles=True, animated=False, assign_ids=False, **kwargs):
        print('Drawing [eval] model output...', end='')
        device, anchor_conf_thr = params['DEVICE'], params['BBOX_THRESHOLD']
        data.construct_tiles(device, force_no_transformation=True)
        n_tiles = data.X_tiled.shape[1]
        if animated:
            anim_frames = []
            animated = plt.subplots(1, figsize=(data.sizex/350, data.sizey/350), 
                                    facecolor='#242424')

        # iterate the timepoints
        for t in range(data.sizet):
            lbl = f'{data.name}_t{t:0>2}|{data.sizet:0>2}_({data.timepoints[t]})'
            print(lbl, end='...', flush=True)

            # get detections and stitch tiles to frame 
            if assign_ids:
                assign_ids = (dest_dir, data.name)
            out = self.get_frame_detections(data, t, device, anchor_conf_thr, 
                                            assign_ids=assign_ids)
            X,X_stch, pred_dets,pred_dets_stch, true_dets,true_dets_stch = out

            # draw stitched frame
            frame_artists = draw_frame(X_stch, true_dets_stch, pred_dets_stch, 
                                       animation=animated, dest_dir=dest_dir, 
                                       fname=lbl, show=show, **kwargs)
            if animated:
                anim_frames.append(frame_artists)

            # also save the single tile images
            if save_single_tiles:
                # for the current timepoints, iter non-empty tiles
                for tile in range(n_tiles):
                    draw_frame(X[tile, data.get_tcenter_idx()], true_dets[tile],
                               pred_dets[tile], draw_YOLO_grid=(self.Sx, self.Sy),
                               dest_dir=dest_dir, show=False, 
                               fname=f'{lbl}_tile{tile:0>2}|{n_tiles:0>2}')

        if animated:
            anim_frames = [anim_frames[0]*2] + anim_frames + [anim_frames[-1]*2]
            ani = animation.ArtistAnimation(animated[0], anim_frames, interval=1000, 
                                            blit=True, repeat_delay=3000)
            if show:
                plt.show()
            Writer = animation.writers['imagemagick'](fps=1)
            print('encoding animation...')
            ani.save(f'{dest_dir}/{data.name}_assoc_dets.mp4', writer=Writer)
            print(f'{data.name} animation saved.')
        print(' - Done.')















    def cntn_id_cost(self, t1_conf, t0_conf, dist, alpha, beta):
        return ((alpha*dist) * (1/(1 + np.e**(beta*(t1_conf+t0_conf-1))))).astype(float)

    def create_id_cost(self, conf, alpha, beta):
        return 1/(1 + np.e**(beta*(conf-.5))).astype(float)

    def dist2cntn_cost(self, dists_detections, alpha, beta):
        for i in range(len(dists_detections)):
            dist = dists_detections[i]
            if i == 0:
                t0_conf = dist.conf
                continue
            t1_conf = dist.conf

            # t1 detection confidence vector to column-matrix
            t1_conf_matrix = np.tile(t1_conf, (len(t0_conf),1)).T
            # t0 detection confidence vector to row-matrix
            t0_conf_matrix = np.tile(t0_conf, (len(t1_conf),1))
            # get the distance between t0 and t1 detections 
            t1_t0_dists_matrix = dist.iloc[:, 3:].values
            
            # pass distances and anchor conf to calculate assoc. cost between detections
            dist.iloc[:,3:] = self.cntn_id_cost(t1_conf_matrix, t0_conf_matrix, 
                                                t1_t0_dists_matrix, alpha, beta)
            t0_conf = t1_conf
        return dists_detections
            
    def greedy_linear_assignment(self, cntd_cost_detections):
        pass

    def hungarian_ID_assignment(self, cntd_cost_detections, alpha, beta):
        t0_conf = pd.DataFrame([])
        axon_id_generator = generate_axon_id()

        # iterate detections: [(conf+anchor coo) + distances to prv frame dets]
        assigned_ids = []
        for t, cntn_c_dets in enumerate(cntd_cost_detections):
            # unpack confidences of current t 
            t1_conf = cntn_c_dets.iloc[:,0]
            # each detections will have a new ID assigned to it, create empty here
            ass_ids = pd.Series(-1, index=cntn_c_dets.index, name=t)
            
            if t == 0:
                # at t=0 every detections is a new one, special case
                ass_ids[:] = [next(axon_id_generator) for _ in range(len(ass_ids))]
            else:
                # t1 and t0 have different lengths, save shorter n of the two
                nt1 = cntn_c_dets.shape[0]
                ndet_min = min((nt1, cntn_c_dets.shape[1]-3))

                # get the costs for creating t1 ID and killing t0 ID (dist indep.)
                create_t1_c = self.create_id_cost(t1_conf, alpha, beta).values
                kill_t0_c = self.create_id_cost(t0_conf, alpha, beta).values
                
                # get the costs for continuing IDs
                cntn_c_matrix = cntn_c_dets.iloc[:,3:].values
                # hungarian algorithm
                t1_idx_cntn, t0_idx_cntn = linear_sum_assignment(cntn_c_matrix)
                # index costs to assignments, associating t1 with t0, dim == ndet_min
                cntn_t1_c = cntn_c_matrix[t1_idx_cntn, t0_idx_cntn]

                # an ID is only NOT continued if the two associated detections both 
                # have lower costs on their own (kill/create)
                cntn_t1_mask = (cntn_t1_c<create_t1_c[:ndet_min]) | (cntn_t1_c<kill_t0_c[:ndet_min])
                # pad t1 mask to length of t1 (in case t1 was shorter than t0)
                cntn_t1_mask = np.pad(cntn_t1_mask, (0, nt1-ndet_min), constant_values=False)
                created_t1_mask = ~cntn_t1_mask

                # created IDs simply get a new ID from the generator
                ass_ids[created_t1_mask] = [next(axon_id_generator) for _ in range(created_t1_mask.sum())]
                # for cntn ID, mask the t0_idx (from hung. alg.) and use it as 
                # indexer of previous assignments timpoint
                assoc_id = t0_idx_cntn[cntn_t1_mask[:ndet_min]]
                ass_ids[cntn_t1_mask] = assigned_ids[-1].iloc[assoc_id].values

            t0_conf = t1_conf
            assigned_ids.append(ass_ids)
        assigned_ids = pd.concat(assigned_ids, axis=1).stack().swaplevel().sort_index()
        assigned_ids = assigned_ids.apply(lambda ass_id: f'Axon_{int(ass_id):0>3}')
        return assigned_ids

    def assign_detections_ids(self, data, params, run_dir):
        alpha, beta, gamma = 0.1, 6, 0.2
        max_dist = 400
        device, conf_thr, num_workers = params['DEVICE'], params['BBOX_THRESHOLD'], params['NUM_WORKERS']
        # conf_thr = np.log((1/gamma) -1)/beta +0.5
        max_cost = self.cntn_id_cost(conf_thr, conf_thr, np.array([max_dist]), alpha, beta)
        print('Assigning axon IDs ...', end='')

        # aggregate all confidences+anchor coo and distances over timepoints
        dists_detections = []
        for t in range(data.sizet):
            if t == 0:
                dets_t0 = self.get_frame_detections(data, t, device, conf_thr)[3]
                dets_t0 = dets_t0.sort_values('conf', ascending=False)
                dists_detections.append(dets_t0)
                continue
            lbl = f'{data.name}_t1:{t:0>3}-t0:{t-1:0>3}'
            print(lbl, end='...')

            # get detections and stitch tiles to frame, only get stitched prediction
            dets_t1 = self.get_frame_detections(data, t, device, conf_thr)[3]
            # reorder detections by confidence
            dets_t1 = dets_t1.sort_values('conf', ascending=False)
            
            # compute a* dists_detections between followup and current frame detections 
            dists = compute_astar_distances(dets_t1, dets_t0, data.mask, max_dist,
                                            num_workers, filename=f'{run_dir}/astar_dists/{lbl}')

            dists = pd.DataFrame(dists, index=dets_t1.index, columns=dets_t0.index)
            dists_detections.append(pd.concat([dets_t1, dists], axis=1))
            dets_t0 = dets_t1

        # pass confidences, distances and hyperparameters to compute cost matrix
        cntd_cost_detections = self.dist2cntn_cost(dists_detections, alpha, beta)

        assigned_ids = self.hungarian_ID_assignment(cntd_cost_detections, alpha, beta)
        assigned_ids.to_csv(f'{run_dir}/model_out/{data.name}_assigned_ids.csv')
        print('Done.')