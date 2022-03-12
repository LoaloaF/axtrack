import torch
import torch.nn as nn
from torchvision import models

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

class YOLO_AXTrack(nn.Module):
    def __init__(self, initial_in_channels, architecture, activation_function, 
                 tilesize, Sy, Sx):
        super(YOLO_AXTrack, self).__init__()

        if len(architecture) == 3:
            lastlayer = architecture.pop(1)[0]
            architecture[0].append(lastlayer)

        self.architecture = architecture
        self.activation_function = activation_function
        self.Sx, self.Sy = Sy, Sx
        self.initial_in_channels = initial_in_channels
        self.tilesize = tilesize

        if self.architecture  in ['mobilenet', 'alexnet', 'resnet']:
            self.ConvNet = self._from_pretrained(self.initial_in_channels, self.architecture)
            self._get_CNN_outdim()
        else:
            # create CNN
            self.ConvNet, out_channels = self._create_ConvNet(self.architecture[0])
        # create FC
        self.fcs = self._create_fcs(self.architecture[1])
    
    def _get_CNN_outdim(self):
        x = torch.zeros((1, self.initial_in_channels, self.tilesize, self.tilesize))
        x = self.ConvNet(x)
        cnn_features = x.flatten(start_dim=1)
        return cnn_features.shape[1]

    def forward(self, x):
        x = self.ConvNet(x)
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
    
    def _create_ConvNet(self, architecture):
        ConvNet = nn.Sequential()
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
            ConvNet.add_module(f'ConvBlock_{layer}', block)
        return ConvNet, out_c

    def _create_fcs(self, architecture):
        in_c = self._get_CNN_outdim()
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
















import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torchvision import models

import matplotlib.pyplot as plt
from matplotlib import animation

# from plotting import draw_frame

from AxonDetections import AxonDetections

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