import sys
import os
import pickle
import time

from config import RAW_DATA_DIR, OUTPUT_DIR, CODE_DIR, PYTORCH_DIR, DEFAULT_DEVICE, DEFAULT_NUM_WORKERS
import numpy as np
import pandas as pd

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler
from torchsummary import summary

from model import YOLO_AXTrack
from loss import YOLO_AXTrack_loss
from dataset_class import Timelapse
from utils import load_checkpoint
from collections import OrderedDict

def get_default_parameters():
    # DATA
    TIMELAPSE_FILE = RAW_DATA_DIR + 'G001_red_compr.deflate.tif'
    LABELS_FILE = OUTPUT_DIR + 'labelled_axons_astardists.csv'
    MASK_FILE = OUTPUT_DIR + 'mask_wells_excl.npy'
    TRAIN_TIMEPOINTS = range(4,34)
    TEST_TIMEPOINTS = list(range(1,4)) + list(range(34,37))
    LOG_CORRECT = True
    PLOT_PREPROC = True
    STANDARDIZE = True
    RETAIN_TEMPORAL_VAR = False
    USE_MOTION_DATA = 'include' #, 'exclude'  'only'
    USE_SPARSE = False
    USE_TRANSFORMS = ['vflip', 'hflip', 'rot', 'translateY', 'translateX']
    CLIP_LOWERLIM = 43 /2**16
    PAD = [0,300,0,300]
    CACHE = None
    FROM_CACHE = None
    SHUFFLE = True
    DROP_LAST = True

    # MODEL
    ARCHITECTURE = [
        #kernelsize, out_channels, stride, concat_to_feature_vector
        (3, 12, 2),     # y-x out: 256
        (3, 24, 2),     # y-x out: 128
        (3, 30, 2),     # y-x out: 64
        (3, 30, 1, True),     # y-x out: 64
        (3, 60, 2, True),     # y-x out: 32
        (3, 120, 2, True),     # y-x out: 16
    ]
    IMG_DIM = 2920, 6364
    SY, SX = 12, 12
    TILESIZE = 512
    GROUPING = True
    LR = 1e-4
    WEIGHT_DECAY = 5e-4
    BATCH_SIZE = 32
    EPOCHS = 301
    LOAD_MODEL = None
    # LOAD_MODEL = ['Exp1_yolo_only_firstfirst', 'run02', 'E0']   # [ExpName, #run, #epoch]
    BBOX_THRESHOLD = .7

    # LOSS
    L_OBJECT = 49.5
    L_NOBJECT = 1
    L_COORD_ANCHOR = 49.5

    # RUN SETTINGS
    SEED = 42
    DEVICE = DEFAULT_DEVICE
    NUM_WORKERS = DEFAULT_NUM_WORKERS
    PIN_MEMORY = True
    NOTES = 'no notes - shame on you!'

    param_dict = OrderedDict({key: val for key, val in locals().items()})
    return param_dict


# this iterates over the entire dataset once
def run_epoch(dataset, model, loss_fn, params, epoch, optimizer=None):
    which_dataset = 'train' if optimizer is not None else 'test'
    tstart = time.time()
    
    dataset.construct_tiles(params['DEVICE'])
    data_loader = setup_data_loaders(params, dataset)

    ntiles = (dataset.tile_info[..., 0] >0).sum().item()
    npos_labels = dataset.tile_info[..., 1].sum().item()
    print(f' - {which_dataset} data - n tiles:{ntiles}, n positive labels:'
          f'{npos_labels}  / -> {npos_labels/ntiles:.3f} loss: ', end='')
    epoch_loss = []
    for batch, (x,y) in enumerate(data_loader):
        x, y = x.to(params['DEVICE']), y.to(params['DEVICE'])
            
        out = model(x)
        loss, loss_components = loss_fn(out, y)
        print(f'{loss.item():.3f}', end='...', flush=True)
        
        if which_dataset == 'train':
            # reset gradients, backprop, step down the loss gradient
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss_components.rename((epoch, which_dataset, batch), inplace=True)
        epoch_loss.append(loss_components)
    epoch_loss = pd.concat(epoch_loss, axis=1)
    epoch_loss.columns.names = ('epoch', 'dataset', 'batch')
    print(f'epoch time elapsed: {round(time.time()-tstart)} s ++++ Mean: '
          f'{epoch_loss.loc["total_summed_loss"].mean():.3f} ++++')
    return epoch_loss

def setup_model(P):
    if P['USE_MOTION_DATA'] == 'include':
        initial_in_channels = 3
    if P['USE_MOTION_DATA'] == 'only':
        initial_in_channels = 2
    if P['USE_MOTION_DATA'] == 'exclude':
        initial_in_channels = 1
    model = YOLO_AXTrack(initial_in_channels, 
                         P['ARCHITECTURE'], 
                         (P['TILESIZE'],P['TILESIZE']), 
                         P['GROUPING'], 
                         P['SY'], 
                         P['SX'])
    model.to(P['DEVICE'])

    print(model)
    summary(model, input_size=(initial_in_channels, P['TILESIZE'], P['TILESIZE']))

    optimizer = optim.Adam(model.parameters(), lr=P['LR'], weight_decay=P['WEIGHT_DECAY'])
    
    loss_fn = YOLO_AXTrack_loss(Sy = P['SX'], 
                                Sx = P['SX'],
                                lambda_obj = P['L_OBJECT'],
                                lambda_noobj = P['L_NOBJECT'],
                                lambda_coord_anchor = P['L_COORD_ANCHOR'],)
    
    if P['LOAD_MODEL']:
        load_checkpoint(OUTPUT_DIR, P['LOAD_MODEL'], model, optimizer, P['DEVICE'])
    return model, loss_fn, optimizer

def setup_data(P):
    # training data
    train_data = Timelapse(P['TIMELAPSE_FILE'], P['LABELS_FILE'], P['MASK_FILE'], 
                           timepoints = P['TRAIN_TIMEPOINTS'],
                           log_correct = P['LOG_CORRECT'],
                           retain_temporal_var = P['RETAIN_TEMPORAL_VAR'],
                           standardize = P['STANDARDIZE'],
                           use_motion_filtered = P['USE_MOTION_DATA'],
                           use_sparse = P['USE_SPARSE'],
                           use_transforms = P['USE_TRANSFORMS'],
                           contrast_llim = P['CLIP_LOWERLIM'],
                           pad = P['PAD'], 
                           plot = P['PLOT_PREPROC'], 
                           cache = P['CACHE'], 
                           from_cache = P['FROM_CACHE'],
                           name = 'train',
                           tilesize = P['TILESIZE'],
                           Sy = P['SY'], 
                           Sx = P['SX'],)
                           
    # test data
    test_data = Timelapse(P['TIMELAPSE_FILE'], P['LABELS_FILE'], P['MASK_FILE'], 
                           timepoints = P['TEST_TIMEPOINTS'],
                           log_correct = P['LOG_CORRECT'],
                           retain_temporal_var = P['RETAIN_TEMPORAL_VAR'],
                           standardize = P['STANDARDIZE'],
                           use_motion_filtered = P['USE_MOTION_DATA'],
                           use_sparse = P['USE_SPARSE'],
                           use_transforms = P['USE_TRANSFORMS'],
                           contrast_llim = P['CLIP_LOWERLIM'],
                           pad = P['PAD'], 
                           plot = P['PLOT_PREPROC'], 
                           cache = P['CACHE'], 
                           from_cache = P['FROM_CACHE'],
                           name = 'test',
                           tilesize = P['TILESIZE'],
                           Sy = P['SY'], 
                           Sx = P['SX'],)
    return train_data, test_data

# def setup_data_loaders(P, train_data, test_data, train_indices_to_sample=None, 
#                        test_indices_to_sample=None):
#     if train_indices_to_sample is None:
#         train_indices_to_sample = train_data.non_empty_tiles
#     train_loader = DataLoader(
#     dataset = train_data,
#     sampler = SubsetRandomSampler(train_indices_to_sample),
#     batch_size = P['BATCH_SIZE'],
#     num_workers = P['NUM_WORKERS'],
#     pin_memory = P['PIN_MEMORY'],
#     drop_last = P['DROP_LAST'],)

#     if test_indices_to_sample is None:
#         test_indices_to_sample = test_data.non_empty_tiles
#     test_loader = DataLoader(
#         dataset = test_data,
#         sampler = SubsetRandomSampler(test_indices_to_sample),
#         batch_size = P['BATCH_SIZE'],
#         num_workers = P['NUM_WORKERS'],
#         pin_memory = P['PIN_MEMORY'],
#         drop_last = P['DROP_LAST'],)
    
def setup_data_loaders(P, dataset):
    data_loader = DataLoader(
        dataset = dataset,
        shuffle = P['SHUFFLE'],
        batch_size = P['BATCH_SIZE'],
        num_workers = P['NUM_WORKERS'],
        pin_memory = P['PIN_MEMORY'],
        drop_last = P['DROP_LAST'],)

    return data_loader