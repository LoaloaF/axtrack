import torch
import torch.nn as nn

import scipy.sparse as sparse
from scipy.signal import convolve

import matplotlib.pyplot as plt
from utils import Y2pandas_anchors
from plotting import draw_tile
# init: 2, 2920, 6364, 2 color channels
# kernel, channels_out, stride, padding (y,x)



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
    def __init__(self, initial_in_channels, architecture_config, img_dim, Sy ,Sx, **kwargs):
        super(YOLO_AXTrack, self).__init__()
        self.architecture = architecture_config
        self.Sy = Sy
        self.Sx = Sx
        self.B = 1
        self.initial_in_channels = initial_in_channels
        self.tilesize = img_dim[0]
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
        [fb.to(device) for fb in self.PostConvNet.feature_blocks]

    def get_CNN_outdim(self):
        x = torch.zeros((1, self.initial_in_channels, self.tilesize, self.tilesize))
        x = self.PreConvNet(x)
        cnn_features = self.PostConvNet(x)
        return cnn_features.shape[1]

    def forward(self, x):
        x = self.PreConvNet(x)
        cnn_features = self.PostConvNet(x)
        return self.fcs(cnn_features)

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
        features_in = self.get_CNN_outdim()
        features_out = self.Sy * self.Sx *self.B*3
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(features_in, FC_size),
            nn.Sigmoid(),
            nn.Linear(FC_size, FC_size),
            nn.Sigmoid(),
            nn.Linear(FC_size, features_out),
        )

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
            X, target = [], []
            for tile in range(n_tiles):
                idx = dataset.fold_idx((t, tile))
                x, tar = dataset[idx]
                X.append(x)
                target.append(tar)
            X = torch.stack(X, 0).to(params['DEVICE'])
            target = torch.stack(target, 0).to(params['DEVICE'])

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
                if X.shape[1] == 5:
                    chanl_idx = slice(2,3)
                elif X.shape[1] == 10:
                    chanl_idx = slice(4,6)
                elif X.shape[1] == 15:
                    chanl_idx = slice(6,9)
                rgb_tile = torch.moveaxis(X[tile, chanl_idx], 0, -1).detach().cpu()
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
    

