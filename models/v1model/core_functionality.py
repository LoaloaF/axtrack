import numpy as np
import pandas as pd

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchsummary import summary

from model import YOLO_AXTrack
from loss import YOLO_AXTrack_loss
from dataset_class import Timelapse
from utils import load_checkpoint

from AxonDetections import AxonDetections

def setup_data(P):
    # training data
    train_data = Timelapse(P['TIMELAPSE_FILE'], P['LABELS_FILE'], P['MASK_FILE'], 
                           timepoints = P['TRAIN_TIMEPOINTS'],
                           log_correct = P['LOG_CORRECT'],
                           standardize_framewise = P['STANDARDIZE_FRAMEWISE'],
                           standardize = P['STANDARDIZE'],
                           use_motion_filtered = P['USE_MOTION_DATA'],
                           use_sparse = P['USE_SPARSE'],
                           use_transforms = P['USE_TRANSFORMS'],
                           temporal_context= P['TEMPORAL_CONTEXT'],
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
                           standardize_framewise = P['STANDARDIZE_FRAMEWISE'],
                           use_motion_filtered = P['USE_MOTION_DATA'],
                           use_sparse = P['USE_SPARSE'],
                           use_transforms = P['USE_TRANSFORMS'],
                           temporal_context= P['TEMPORAL_CONTEXT'],
                           contrast_llim = P['CLIP_LOWERLIM'],
                           pad = P['PAD'],  
                           plot = P['PLOT_PREPROC'], 
                           cache = P['CACHE'], 
                           from_cache = P['FROM_CACHE'],
                           name = 'test',
                           tilesize = P['TILESIZE'],
                           Sy = P['SY'], 
                           Sx = P['SX'],
                        #    standardize = P['STANDARDIZE'],)
                           standardize = train_data.stnd_scaler,)
    return train_data, test_data

def setup_model(P):
    if P['USE_MOTION_DATA'] == 'include':
        initial_in_channels = 3 *(P['TEMPORAL_CONTEXT']*2+1)
    if P['USE_MOTION_DATA'] == 'only':
        initial_in_channels = 2 *(P['TEMPORAL_CONTEXT']*2+1)
    if P['USE_MOTION_DATA'] == 'exclude':
        initial_in_channels = 1 *(P['TEMPORAL_CONTEXT']*2+1)

    model = YOLO_AXTrack(initial_in_channels = initial_in_channels, 
                         architecture = P['ARCHITECTURE'], 
                         activation_function = P['ACTIVATION_FUNCTION'],
                         tilesize = P['TILESIZE'], 
                         Sy = P['SY'],
                         Sx =  P['SX'],)
    model.to_device(P['DEVICE'])
    summary(model, input_size=(initial_in_channels, P['TILESIZE'], P['TILESIZE']), 
                               device=P['DEVICE'])

    optimizer = optim.Adam(model.parameters(), lr=P['LR'], weight_decay=P['WEIGHT_DECAY'])

    if P['LR_DECAYRATE']:
        decay = lambda E: np.e**((-1/P['LR_DECAYRATE']) * np.sqrt(E)) 
    else:
        decay = lambda E: 1
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, decay)
    
    loss_fn = YOLO_AXTrack_loss(Sy = P['SX'], 
                                Sx = P['SX'],
                                conf_thr = P['BBOX_THRESHOLD'],
                                lambda_obj = P['L_OBJECT'],
                                lambda_noobj = P['L_NOBJECT'],
                                lambda_coord_anchor = P['L_COORD_ANCHOR'],)
    
    if P['LOAD_MODEL']:
        load_checkpoint(P['LOAD_MODEL'], model, optimizer, lr_scheduler,  P['DEVICE'])
    return model, loss_fn, optimizer, lr_scheduler

def setup_data_loaders(P, dataset):
    data_loader = DataLoader(
        dataset = dataset,
        shuffle = P['SHUFFLE'],
        batch_size = P['BATCH_SIZE'],
        num_workers = P['NUM_WORKERS'],
        pin_memory = P['PIN_MEMORY'],
        drop_last = P['DROP_LAST'],)
    return data_loader

def run_epoch(data_loader, model, loss_fn, device, epoch, which_dataset, optimizer):
    print(f'LOSS: ', end='')
    
    epoch_loss = []
    for batch, (X, target) in enumerate(data_loader):
        X, target = X.to(device), target.to(device)
            
        pred = model(X)
        loss, loss_components = loss_fn(pred, target)
        loss_components.rename((epoch, which_dataset, batch), inplace=True)
        epoch_loss.append(loss_components)
        print(f'{loss.item():.3f}', end='...', flush=True)
        
        if which_dataset == 'train':
            # reset gradients, backprop, step down the loss gradient
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    epoch_loss = pd.concat(epoch_loss, axis=1)
    epoch_loss.columns.names = ('epoch', 'dataset', 'batch')
    return epoch_loss

def prepare_data(device, dataset):
    dataset.construct_tiles(device)
    ntiles = (dataset.tile_info[..., 0] >0).sum().item()
    npos_labels = dataset.tile_info[..., 1].sum().item()
    avg_pos_rate = npos_labels/(ntiles+1)
    print(f' - {dataset.name} data - n_positive_labels:{npos_labels} / ntiles'
        f':{ntiles} = {avg_pos_rate:.3f} per tile - ', end='')
    return avg_pos_rate

# this iterates over the entire dataset once
def one_epoch(dataset, model, loss_fn, params, epoch, optimizer=None, lr_scheduler=None):
    which_dataset = 'train' if optimizer is not None else 'test'
    while prepare_data(params['DEVICE'], dataset) < 1:
        print('Bad data augmentation -- Doing it again --')
    
    # get the dataloader and run model on entire dataset
    data_loader = setup_data_loaders(params, dataset)
    epoch_loss = run_epoch(data_loader, model, loss_fn, params['DEVICE'], epoch, 
                           which_dataset, optimizer)
    
    # every 5th epoch calculate precision, recall, and F1 for entire dataset
    if not (epoch % 5):
        ax_dets = AxonDetections(model, dataset, params)
        cnfs_mtrx = sum([ax_dets.compute_TP_FP_FN(t) for t in range(len(ax_dets))])
        epoch_metrics = ax_dets.compute_prc_rcl_F1(cnfs_mtrx, epoch, which_dataset)
    else:
        epoch_metrics = pd.DataFrame([], index=['precision', 'recall', 'F1'])

    print((f'\n++++ Mean: {epoch_loss.loc["total_summed_loss"].mean():.3f}, F1'
           f' {epoch_metrics.loc["F1"].max():.3f} ++++'))

    if which_dataset == 'train' and lr_scheduler:
        lr_scheduler.step()
    torch.cuda.empty_cache()
    return epoch_loss, epoch_metrics