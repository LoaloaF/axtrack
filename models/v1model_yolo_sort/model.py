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
    def __init__(self, in_channels, out_channels, use_bachnorm=True, **kwargs):
        super(CNNBlock, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, bias=True, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.LeakyReLU = nn.LeakyReLU(.1)
        self.use_bachnorm = use_bachnorm
    
    def forward(self, x):
        if self.use_bachnorm:
            conv_out = self.conv(x)
            # print(f'CNN in size: {x.shape}, \n\t\tout: {conv_out.shape}\n')
            return self.LeakyReLU(self.batchnorm(conv_out)) 
        else:    
            return self.LeakyReLU(self.conv(x))


class YOLO_AXTrack(nn.Module):
    def __init__(self, initial_in_channels, architecture_config, img_dim, grouping, Sy ,Sx, **kwargs):
        super(YOLO_AXTrack, self).__init__()
        self.architecture = architecture_config
        self.Sy = Sy
        self.Sx = Sx
        self.B = 1
        self.initial_in_channels = initial_in_channels
        self.tilesize = img_dim[0]
        
        # create CNN
        self.ConvNet = self._create_ConvNet(self.architecture, grouping)
        cnn_input = torch.zeros((1, self.initial_in_channels, *img_dim))
        self.conv_outdim = torch.tensor(list(self.ConvNet(cnn_input).shape[1:]))

        # create FC
        self.fcs = self._create_fcs(**kwargs)

    def forward(self, x):
        x = self.ConvNet(x)
        features = torch.flatten(x, start_dim=1)
        return self.fcs(features)

    def _create_ConvNet(self, architecture, grouping):
        ConvNet = nn.Sequential()
        in_c = self.initial_in_channels
        for i in range(len(architecture)):
            groups = in_c if grouping and i < 2 else 1
            if architecture[i] != 'M':
                out_c = architecture[i][1]
                block = CNNBlock(in_channels = in_c, 
                                 out_channels = out_c, 
                                 padding = (1,1),
                                 groups = groups,
                                 kernel_size = architecture[i][0], 
                                 stride = architecture[i][2])
                in_c = out_c
            else:
                block = nn.MaxPool2d(2,2)
            ConvNet.add_module(f'block_{i}', block)
        return ConvNet


    def _create_fcs(self, FC_size=512):
        features_in = torch.prod(self.conv_outdim)
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

            # get the image data
            X = dataset.X_tiled[t].to(params['DEVICE'])
            target = dataset.target_tiled[t].to(params['DEVICE'])
            n_tiles = X.shape[0]

            # forward pass through the model - get predictions, convert to anchors
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
                
                # slice to tile
                rgb_tile = torch.moveaxis(X[tile], 0, -1).detach().cpu()
                target_anchors_tile = target_anchors[tile]
                pred_anchors_tile = pred_anchors[tile]

                # draw tile, also save the single tile images
                draw_tile(rgb_tile, target_anchors_tile, pred_anchors_tile, 
                          tile_axis=ax, draw_YOLO_grid=(self.Sx, self.Sy),
                          dest_dir=dest_dir if save_sinlge_tiles else None,
                          fname=f'tile{tile:0>2}_t{t:0>2}')

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
        (3, 12, 2),     # y-x out: 256
        (3, 24, 2),     # y-x out: 128
        (3, 30, 2),     # y-x out: 64
        (3, 30, 1, True),     # y-x out: 64
        (3, 60, 2, True),     # y-x out: 32
        (3, 120, 2, True),     # y-x out: 16
    ]
    parameters['ARCHITECTURE'] = ARCHITECTURE
    parameters['GROUPING'] = False


    model, loss_fn, optimizer = setup_model(parameters)

    bs = 2
    inchannels = 3
    X = torch.rand((bs, inchannels, parameters['TILESIZE'], parameters['TILESIZE']))

    OUT = model(X)

    # print(OUT)
    print(OUT.shape)
    

