
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torchvision import models

import matplotlib.pyplot as plt
from matplotlib import animation

from model_out_utils import Y2pandas_anchors, non_max_supression_pandas, non_max_supress_pred
from plotting import draw_frame

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation_function, **kwargs):
        super(CNNBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels, out_channels, bias=True, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.activation_function = activation_function
    
    def forward(self, x):
        conv_out = self.conv(x)
        # print(f'CNN in size: {x.shape}, \n\t\tout: {conv_out.shape}\n')
        return self.activation_function(self.batchnorm(conv_out)) 

class CNNFeature_collector(nn.Module):
    def __init__(self, architecture, activation_function, in_c):
        super(CNNFeature_collector, self).__init__()
        
        self.feature_blocks = []
        for layer in range(len(architecture)):
            out_c = architecture[layer][1]
            fblock = CNNBlock(in_channels = in_c, 
                              out_channels = out_c, 
                              activation_function = activation_function,
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
    def __init__(self, initial_in_channels, architecture, activation_function, 
                 tilesize, Sy, Sx, **kwargs):
        super(YOLO_AXTrack, self).__init__()

        self.architecture = architecture
        self.activation_function = activation_function
        self.Sx, self.Sy = Sy, Sx
        self.initial_in_channels = initial_in_channels
        self.tilesize = tilesize

        if self.architecture  in ['mobilenet', 'alexnet', 'resnet']:
            self.PreConvNet = self._from_pretrained(self.initial_in_channels, self.architecture)
            self.PostConvNet = None
            self.get_CNN_outdim()
        else:
            # create preCNN
            # self.architecture.append(None)
            self.PreConvNet, out_channels = self._create_PreConvNet(self.architecture[0])
            # create postCNN
            self.PostConvNet = CNNFeature_collector(self.architecture[1], 
                                                    self.activation_function,
                                                    out_channels)
        # create FC
        self.fcs = self._create_fcs(self.architecture[2])
    
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
    
    def _create_PreConvNet(self, architecture):
        PreConvNet = nn.Sequential()
        in_c = self.initial_in_channels
        for layer in range(len(architecture)):
            if architecture[layer] != 'M':
                groups = architecture[layer][3]
                out_c = architecture[layer][1]
                block = CNNBlock(in_channels = in_c, 
                                 out_channels = out_c, 
                                 activation_function = self.activation_function,
                                 padding = (1,1),
                                 groups = groups,
                                 kernel_size = architecture[layer][0], 
                                 stride = architecture[layer][2])
                in_c = out_c
            else:
                block = nn.MaxPool2d(2,2)
            PreConvNet.add_module(f'PreConvBlock_{layer}', block)
        return PreConvNet, out_c

    # def _create_fcs(self, architecture):
    #     features_in = self.get_CNN_outdim()
    #     features_out = self.Sy * self.Sx *3
    #     return nn.Sequential(
    #         nn.Flatten(),

    #         nn.Linear(features_in, FC_size),
    #         # nn.LeakyReLU(),
    #         nn.Sigmoid(),
    #         # nn.Dropout(.15),
    #         nn.Linear(FC_size, FC_size),
    #         # nn.LeakyReLU(),
    #         nn.Sigmoid(),
    #         nn.Linear(FC_size, features_out),
    #     )

    def _create_fcs(self, architecture):
        in_c = self.get_CNN_outdim()
        seq = [nn.Flatten()]
        for element_type, param  in architecture:
            if element_type == 'FC':
                seq.append(nn.Linear(in_c, param))
                in_c = param
            elif element_type == 'dropout':
                seq.append(nn.Dropout(param))
            elif element_type == 'activation':
                seq.append(param)
        seq.append(nn.Linear(in_c, self.Sy * self.Sx *3))
        return nn.Sequential(*seq)
            
    def detect_axons(self, X):
        self.eval()
        n_tiles = X.shape[0]
        with torch.no_grad():
            pred = self(X).reshape(n_tiles, self.Sx, self.Sy, -1)
        self.train()
        return pred
    
    def get_frame_detections(self, data, t, device, anchor_conf_thr, assign_ids=None, nms=True):
        # get a stack of tiles that makes up a frame
        # batch dim of X corresbonds to all tiles in frame at time t
        X, target = data.get_stack('image', t)
        X = X.to(device)
        # use the model to detect axons, output here follows target.shape
        pred_target = self.detect_axons(X)
        if nms:
            pred_target = non_max_supress_pred(pred_target, self.Sx, self.Sy, self.tilesize, device)

        # convert the detected anchors in YOLO format to normal ones
        pred_dets = Y2pandas_anchors(pred_target, self.Sx, self.Sy, 
                                     self.tilesize, device, anchor_conf_thr) 
        # non max supression on stichted data supressing detections of neighboring tiles
        pred_dets = [non_max_supression_pandas(pred_det) for pred_det in pred_dets]
        
        if target is not None:
            true_dets = Y2pandas_anchors(target.to(device), self.Sx, self.Sy, 
                                         self.tilesize, device, 1) 

        # stitch the prediction from tile format back to frame format
        X_stch, pred_dets_stch, true_dets_stch = data.stitch_tiles(X, pred_dets, true_dets)
        # pred_dets_stch = non_max_supression_pandas(pred_dets_stch)
        if assign_ids:
            run_dir, data_name = assign_ids
            pred_dets_stch = self.get_detections_ids(pred_dets_stch, t, run_dir, data_name)
        return X,X_stch, pred_dets,pred_dets_stch, true_dets,true_dets_stch

    def get_detections_ids(self, dets, t, run_dir, data_name):
        assigned_ids = pd.read_csv(f'{run_dir}/{data_name}_assigned_ids.csv', index_col=(0,1))
        dets.index = assigned_ids.loc[(t, dets.index),:].values[:,0]
        return dets

    def predict_all(self, data, params, dest_dir=None, show=False, 
                    save_single_tiles=False, animated=False, assign_ids=False, 
                    **kwargs):
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
            lbl = f'{data.name}_t{t:0>2}|{data.sizet:0>2}_({data.timepoints[t]})_{params["NOTES"]}'
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
                                       draw_grid=self.tilesize, fname=lbl, 
                                       show=show, **kwargs)
            if animated:
                anim_frames.append(frame_artists)

            # also save the single tile images
            if save_single_tiles:
                # for the current timepoints, iter non-empty tiles
                for tile in range(n_tiles):
                    draw_frame(X[tile, data.get_tcenter_idx()], true_dets[tile],
                               pred_dets[tile], draw_grid=self.tilesize/self.Sx,
                               dest_dir=dest_dir, show=True, 
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