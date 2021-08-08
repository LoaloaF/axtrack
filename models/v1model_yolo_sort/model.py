import torch
import torch.nn as nn

import pandas as pd
import numpy as np
from scipy.optimize import linear_sum_assignment
from utils import get_astar_path

import scipy.sparse as sparse
from scipy.signal import convolve

import matplotlib.pyplot as plt
from utils import Y2pandas_anchors
from plotting import draw_tile

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
            self.PreConvNet = self._create_PreConvNet(self.architecture[0],
                                                    self.initial_in_channels)
            # create postCNN
            self.PostConvNet = CNNFeature_collector(self.architecture[1], 
                                                    self.PreConvNet[-1].out_channels)
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
            groups = architecture[layer][3]
            if architecture[layer] != 'M':
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
        return PreConvNet

    def _create_fcs(self, FC_size=512):
    # def _create_fcs(self, FC_size=1024):
        features_in = self.get_CNN_outdim()
        features_out = self.Sy * self.Sx *self.B*3
        return nn.Sequential(
            nn.Flatten(),
            # nn.Dropout(.2),
            nn.Linear(features_in, FC_size),
            nn.Sigmoid(),
            nn.Linear(FC_size, FC_size),
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

    def compute_astar_distances(self, t0_target, t1_target, mask, max_dist, 
                                filename, from_cache=False):
        if not from_cache:
            weights = np.where(mask==1, 1, 2**16).astype(np.float32)

            # with concurrent.futures.ProcessPoolExecutor(max_workers=how['threads']) as executer:
            #             [executer.submit(MUA_analyzeMouseParadigm, folder) for folder in dirs]

            dists = []
            for i, t0_det in enumerate(t0_target):
                print(f'{i}/{(t0_target[:,0]!=-1).sum()}', end='...', flush=True)

                t0_dists = []
                for t1_det in t1_target:
                    if (t0_det == -1).all() or (t1_det == -1).all():
                        dist = -1
                    else:
                        dist = np.linalg.norm(t0_det-t1_det)
                        if dist < max_dist:
                            dist = get_astar_path(np.flip(t0_det), np.flip(t1_det), weights)
                        dist = min((dist, max_dist))
                    t0_dists.append(dist)
                dists.append(t0_dists)

            dists = np.array(dists)
            np.save(filename, dists)
        else:
            dists = np.load(filename)
        return dists
    
    def compute_association_cost(self, D, Conf, alpha, beta, gamma, max_cost):
        def compute_cost(which, dist=None, t0_conf=None, t1_conf=None):
            if which in ('ID_continues', 'ID_created', 'ID_killed'):
                if which == 'ID_continues': 
                    c = (alpha*dist) * (1/(1 + np.e**(beta*(t0_conf+t1_conf-1))))
                    c[no_IDt0_mask | no_IDt1_mask] = max_cost
                    # c[no_IDt0_mask & no_IDt1_mask & diag_mask] = 0

                elif which == 'ID_killed':
                    c = 1/(1 + np.e**(beta*(t0_conf-.5)))
                    c[no_IDt0_mask | ~diag_mask] = max_cost
                    # c[no_IDt0_mask & diag_mask] = 0

                elif which == 'ID_created':
                    c = 1/(1 + np.e**(beta*(t1_conf-.5)))
                    c[no_IDt1_mask | ~diag_mask] = max_cost
                    # c[no_IDt1_mask & diag_mask] = 0
                
            elif which in ('ID_t0_FP', 'ID_t1_FP'):
                c = gamma*np.eye(dist.shape[0])
                
                if which == 'ID_t0_FP':
                    c[no_IDt0_mask | ~diag_mask] = max_cost
                    c[no_IDt0_mask & diag_mask] = 0
                
                if which == 'ID_t1_FP':
                    c[no_IDt1_mask | ~diag_mask] = max_cost
                    c[no_IDt1_mask & diag_mask] = 0
            return c.astype(float)

        capacity = D.shape[1]
        filler_block = np.full((capacity,capacity), max_cost)
        C = []
        for t in range(D.shape[0]):
            # t0 detection confidence vector to row-matrix
            det_conf_t0 = np.tile(Conf[t], (400,1)).T
            # t1 detection confidence vector to column-matrix
            det_conf_t1 = np.tile(Conf[t+1], (400,1))
            det_dists = D[t]

            diag_mask = np.eye(det_dists.shape[0], dtype=bool)
            # no_ID_mask = (det_conf_t0==-1) | (det_conf_t1==-1)
            no_IDt0_mask = det_conf_t0 == -1
            no_IDt1_mask = det_conf_t1 == -1

            ID_cont = compute_cost('ID_continues', det_dists, det_conf_t0, det_conf_t1)
            ID_killed = compute_cost('ID_killed', det_dists, det_conf_t0, det_conf_t1)
            ID_created = compute_cost('ID_created', det_dists, det_conf_t0, det_conf_t1)
            ID_t0_FP = compute_cost('ID_t0_FP', det_dists, det_conf_t0, det_conf_t1)
            ID_t1_FP = compute_cost('ID_t1_FP', det_dists, det_conf_t0, det_conf_t1)

            cost_t0t1 = np.block([[ID_cont,    ID_killed,    ID_t0_FP],
                                  [ID_created, filler_block, filler_block],
                                  [ID_t1_FP,   filler_block, filler_block]])
            print(cost_t0t1.min())
            print(cost_t0t1.max())
            plt.imshow(cost_t0t1, cmap='nipy_spectral')
            plt.show()
            C.append(cost_t0t1)
        return np.stack(C)
            

    def hungarian_algorithm(self, C):
        return linear_sum_assignment(C)

    def assign_ids(self, data, device, run_dir):
        thr = .5
        capacity = 400
        alpha, beta, gamma = 0.1, 6, 0.2
        max_dist = 400
        max_cost = (alpha*max_dist) * (1/(1 + np.e**(beta*(thr+thr-1))))

        # aggregate all distances and confidences over timepoints
        prv_t_pred_target = np.full((capacity, 3), -1) 
        D, Conf, Coos = [], [], []
        for t in range(data.sizet):
            # get a stack of tiles that makes up a frame
            img, target = data.get_stack('image', t)
            img, target = img.to(device), target.to(device)
            # use the model to detect axons, output here follows target.shape
            pred_target = self.detect_axons_in_frame(img)

            # convert the detected anchors in YOLO format to normal ones
            target = Y2pandas_anchors(target, self.Sx, self.Sy, self.tilesize, 
                                      device, 1) 
            pred_target = Y2pandas_anchors(pred_target, self.Sx, self.Sy, 
                                           self.tilesize, device, thr) 

            # stitch the data from tile format back to frame format
            img, target, pred_target = data.stitch_tiles(img, target, pred_target)
            
            # reorder detections by confidence
            conf_order = pred_target.conf.sort_values(ascending=False).index
            pred_target = pred_target.reindex(conf_order)
            print(pred_target.shape)
            
            # for the association, a consistent length is needed, pad with -1 yx and 0 conf
            pad_to_cap = np.full((capacity-len(pred_target), 3), -1) 
            pred_target = np.concatenate([pred_target.values, pad_to_cap])

            # compute the a star distances between current frame- and followup
            # frame detections 
            fname = f'{run_dir}/astar_dists/dists_t{t:0>3}.npy'
            dists = self.compute_astar_distances(prv_t_pred_target[:,1:], 
                                                 pred_target[:,1:], 
                                                 data.mask, max_dist,
                                                 filename=fname, 
                                                 from_cache=False)

            D.append(dists)
            Conf.append(prv_t_pred_target[:, 0])
            Coos.append(prv_t_pred_target[:, 1:])
            prv_t_pred_target = pred_target.copy()
        D.append(D[0])
        Conf.extend([prv_t_pred_target[:, 0], np.full(capacity, -1)])
        Coos.extend([prv_t_pred_target[:, 1:], np.full((capacity,2), -1)])

        Conf = np.stack(Conf)
        D = np.stack(D)
        Coos = np.stack(Coos)
        print(Conf.shape, D.shape, Coos.shape)
        print()

        C = self.compute_association_cost(D, Conf, alpha, beta, gamma, max_cost)
        # for t in range(C.shape[0]):
        for t in range(len(C)):

            fig,ax = plt.subplots(2,2, figsize=(13,18), tight_layout=True, sharex=True, sharey=True)
            
            for i in range(2):
                if not i:
                    cost = C[t, :, :400]
                else:
                    cost = C[t, :400]
                print(cost.shape)
                print()
                y_idx_assignm, x_idx_assignm = self.hungarian_algorithm(cost)
                print(y_idx_assignm.shape)
                print(x_idx_assignm.shape)
                print()

                fig.suptitle(cost[y_idx_assignm, x_idx_assignm].sum())

                ass_plot = np.zeros_like(cost)
                ass_plot[y_idx_assignm, x_idx_assignm] = 1
                ax[0,i].set_title('assignment matrix')
                ax[0,i].imshow(ass_plot, cmap='nipy_spectral')
                
                ax[0,i].set_title('cost matrix')
                ax[1,i].imshow(cost, cmap='nipy_spectral')

            plt.show()
        exit()



    def predict_all(self, dataset, params, dest_dir=None, show=False, 
                    save_sinlge_tiles=True):
        print('Computing [eval] model output...', end='')
        n_tiles = dataset.X_tiled.shape[1]
        # iterate the timepoints
        for t in range(dataset.sizet):
            # make a figure that plots the tiles as axis
            fig, axes = plt.subplots(dataset.ytiles, dataset.xtiles, facecolor='#242424',
                                    figsize=(dataset.sizex/400, dataset.sizey/400))
            fig.subplots_adjust(wspace=0, hspace=0, top=.99, bottom=.01, 
                                left=.01, right=.99)
            for ax in axes.flatten():
                ax.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
                ax.set_facecolor('#242424')
                ax.spines['right'].set_color('#787878')
                ax.spines['left'].set_color('#787878')
                ax.spines['top'].set_color('#787878')
                ax.spines['bottom'].set_color('#787878')

            # get the data: both image and targets
            X, target = dataset.get_stack('image', t)
            X = X.to(params['DEVICE'])
            target = target.to(params['DEVICE'])

            # predict anchor coos
            self.eval()
            with torch.no_grad():
                pred = self(X).reshape(n_tiles, self.Sx, self.Sy, -1)
            self.train()
            pred_anchors = Y2pandas_anchors(pred, self.Sx, self.Sy, self.tilesize, 
                                            params['DEVICE'], params['BBOX_THRESHOLD'])
            
            # get the labels and convert to anchors like above
            target_anchors = Y2pandas_anchors(target, dataset.Sx, 
                                              dataset.Sy, dataset.tilesize, 
                                              params['DEVICE'], 1)
            
            # for the current timepoints, iter non-empty tiles
            print(t+1, end='...t:', flush=True)
            for tile in range(n_tiles):
                # get the original coordinate of the tile
                ytile_coo, xtile_coo = dataset.flat_tile_idx2yx_tile_idx(tile)
                ax = axes[ytile_coo, xtile_coo]
                
                # slice to tile and correct colorcolorchanels
                rgb_tile = torch.moveaxis(X[tile, dataset.t_center], 0, -1).detach().cpu()
                target_anchors_tile = target_anchors[tile]
                pred_anchors_tile = pred_anchors[tile]

                # draw tile, also save the single tile images
                draw_tile(rgb_tile, target_anchors_tile, pred_anchors_tile, 
                          tile_axis=ax, draw_YOLO_grid=(self.Sx, self.Sy),
                          dest_dir=dest_dir if save_sinlge_tiles else None,
                          fname=f'{dataset.name}_tile{tile:0>2}_t{t:0>2}')

                # add label
                ax.text(.1, .93, f't: {t}', transform=fig.transFigure, 
                        fontsize=16, color='w')
                ax.text(.2, .93, ('solid=label, dashed=predicted (larger means '
                        'more confident)'), transform=fig.transFigure, 
                        fontsize=10, color='w')
            if show:
                plt.show()
            if dest_dir:
                fig.savefig(f'{dest_dir}/{dataset.name}_t{t:0>2}.png')
            plt.close()
        print(' - Done.')


if __name__ == '__main__':
    from core_functionality import get_default_parameters, setup_model
    parameters = get_default_parameters()

    ARCHITECTURE = [
        #kernelsize, out_channels, stride, concat_to_feature_vector
        [(3, 20,  2,  5),     # y-x out: 256
         (3, 40,  2,  5),     # y-x out: 128
         (3, 80,  2,  5),     # y-x out: 64
         (3, 80,  1,  1)],     # y-x out: 64
        [(3, 80,  2,  1),     # y-x out: 32
         (3, 160, 2,  1)],     # y-x out: 16
    ]
    
    parameters['ARCHITECTURE'] = ARCHITECTURE
    parameters['GROUPING'] = False
    parameters['USE_MOTION_DATA'] = 'include'
    parameters['USE_MOTION_DATA'] = 'temp_context'

    bs = 2
    inchannels = 5


    model, loss_fn, optimizer = setup_model(parameters)

    X = torch.rand((bs, inchannels, parameters['TILESIZE'], parameters['TILESIZE']))

    OUT = model(X)

    # print(OUT)
    print(OUT.shape)
    

