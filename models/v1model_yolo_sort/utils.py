import os
import glob
import shutil
import pickle
from datetime import datetime
from collections import Counter
from config import RAW_DATA_DIR, OUTPUT_DIR, CODE_DIR, PYTORCH_DIR, SPACER

import numpy as np
import torch
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.font_manager

# from plotting import plot_image

def SSBox2ImgBox_coords(target_tensor, Sx, Sy, tilesize, device, with_batch_dim=True):
    # nobatch dim
    # this creates a bool mask tensor that is true for all pos label entries
    box_mask = target_tensor > 0

    # these rescale vectors scale depending on the gridbox the label is in
    rescale_x = torch.arange(Sx).reshape(Sx, 1, 1).to(device)
    rescale_y = torch.arange(Sy).reshape(1, Sy, 1).to(device)
    # add batch dim at position 0
    if with_batch_dim:
        rescale_x = rescale_x.unsqueeze(0)
        rescale_y = rescale_y.unsqueeze(0)
        
    # this is 200 IQ: we add the gridbox (0-S) to the within-gridbox
    # coordinate (0-1). The coordiante is now represented in terms of
    # gridboxsize (here 64)
    target_tensor[..., 1::3] = (target_tensor[..., 1::3] + rescale_x)
    target_tensor[..., 2::3] = (target_tensor[..., 2::3] + rescale_y)
    # now we just muplitply by gridboxsize to scale to image coo
    target_tensor[..., 1::3] = (target_tensor[..., 1::3] * tilesize/Sx).round()
    target_tensor[..., 2::3] = (target_tensor[..., 2::3] * tilesize/Sy).round()
    # the rescale vector also adds to the 0-elements (unwanted). Mask them here
    target_tensor = torch.where(box_mask, target_tensor, torch.tensor(0.0).to(device))
    return target_tensor

def filter_anchors(target_tensor, conf_threshold):
    # convert the target tensor to shape (batchsize, Sx*Sy, 3)
    batch_size, labelsize = target_tensor.shape[0], target_tensor.shape[-1]
    target_tensor = target_tensor.reshape((batch_size, -1, labelsize))
    
    # split batches into list because of variable sizes 
    frame_targets = list(torch.tensor_split(target_tensor, batch_size))
    # filter each example
    frame_targets_fltd = [tar[tar[:,:,0] >= conf_threshold] for tar in frame_targets]
    return frame_targets_fltd

def fltd_tensor2pandas(target_tensor, t=0, sort=True):
    axon_lbls = [f'Axon_{i:0>2}' for i in range(target_tensor.shape[0])]
    axon_lbls = pd.MultiIndex.from_product([[t], axon_lbls])
    cols = 'conf', 'anchor_x', 'anchor_y'
    
    target_df = pd.DataFrame(target_tensor.to('cpu').numpy(), index=axon_lbls, 
                             columns=cols)
    if sort:
        target_df.sort_values('conf', inplace=True)
    return target_df

def Y2pandas_anchors(Y, Sx, Sy, tilesize, device, conf_thr):
    # convert the YOLO coordinates (0-1) to image coordinates (tilesize x tilesize)
    Y_imgcoo = SSBox2ImgBox_coords(Y.clone(), Sx, Sy, tilesize, device)

    # filter the (batch, Sx, Sy, 3) output into a list of length batchsize,
    # where the list elements are tensors of shape (#_boxes_above_thr x 3)
    # each element in the list represents a frame/tile
    Ys_list_fltd = filter_anchors(Y_imgcoo, conf_threshold=conf_thr)
    # convert each frame from tensor to pandas
    for i in range(len(Ys_list_fltd)):
        Ys_list_fltd[i] = fltd_tensor2pandas(Ys_list_fltd[i], t=i)
    return Ys_list_fltd

# def get_model_output_anchors(model, x, target, params):
#     print('Computing [eval] model output...', end='')
#     batch_size, _, height, width = x.shape

#     model.eval()
#     with torch.no_grad():
#         pred = model(x).reshape(batch_size, model.Sx, model.Sy, -1)

#     pred_anchors = Y2pandas_anchors(pred, model.Sx, model.Sy, model.tilesize, 
#                                     params['DEVICE'], params['BBOX_THRESHOLD'])
#     true_anchors = Y2pandas_anchors(target.clone(), model.Sx, model.Sy, 
#                                     model.tilesize, params['DEVICE'], 1)
#     model.train()
#     print('Done.')
#     return true_anchors, pred_anchors

# def check_training_progress(model, x, y, params, which_dataset, epoch, nframes=6):    
#     targets_list_fltd, preds_list_fltd = get_model_output_anchors(model, x, y, params)

#     if nframes is None:
#         nframes = x.shape[0] # = batch size (all)
#     frames = x[:nframes].permute(0,2,3,1).detach().to('cpu') # move RGB dim to end 
#     frames = [t[0] for t in torch.tensor_split(frames, nframes)]
#     targets = targets_list_fltd[:nframes]
#     preds = preds_list_fltd[:nframes]
#     return {f'E{epoch} - {which_dataset} data - ': (frames, targets, preds)}

def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {"state_dict": model.state_dict(),
                   "optimizer": optimizer.state_dict(),}
    torch.save(checkpoint, filename)

def load_checkpoint(OUTPUT_DIR, load_model, model, optimizer, device):
    print("=> Loading checkpoint...", end='')
    exp_dir = f'{OUTPUT_DIR}/runs/{load_model[0]}/'
    run_dir = [rd for rd in os.listdir(exp_dir) if load_model[1] in rd][0]
    file = f'{exp_dir}/{run_dir}/models/E{load_model[2]}.pth'
    checkpoint = torch.load(file, map_location=torch.device(device))
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    print('Done.')

def create_logging_dirs(exp_name):
    EXP_DIR = f'{OUTPUT_DIR}/runs/{exp_name}'
    os.makedirs(EXP_DIR, exist_ok=True)
    
    runs = [int(r[3:5]) for r in os.listdir(EXP_DIR)]
    if 99 in runs:
        print('Run dir full (run99 in there), tidy up!')
        exit(1)
    run = 0 if not runs else max(runs)+1
    run_label = f'run{run:0>2}_' + datetime.now().strftime("%d.%m.%Y_%H.%M.%S")

    RUN_DIR = f'{EXP_DIR}/{run_label}'
    MODELS_DIR = f'{RUN_DIR}/models'
    METRICS_DIR = f'{RUN_DIR}/metrics'
    PREPROC_DATA_DIR = f'{RUN_DIR}/preproc_data/'
    
    [os.makedirs(d) for d in [RUN_DIR, MODELS_DIR, METRICS_DIR, PREPROC_DATA_DIR]]
    return (RUN_DIR, MODELS_DIR, METRICS_DIR, PREPROC_DATA_DIR), run_label

def clean_rundirs(exp_name, delete_runs=None, keep_runs=None):
    EXP_DIR = f'{OUTPUT_DIR}/runs/{exp_name}'
    for d in sorted(os.listdir(EXP_DIR)):
        n_metrics = len(glob.glob(f'{EXP_DIR}/{d}/metrics/E*.csv'))
        n_models = len(glob.glob(f'{EXP_DIR}/{d}/models/*.pth'))
        print(f'{d} - Epochs: {n_metrics}, models saved: {n_models}')

        if delete_runs and n_metrics<delete_runs:
            shutil.rmtree(f'{EXP_DIR}/{d}')
            print('--deleted--\n')
        elif keep_runs is not None and int(d[3:5]) not in keep_runs:
            shutil.rmtree(f'{EXP_DIR}/{d}')
            print('--deleted--\n')

def write_parameters(file, params):
    with open(file.replace('pkl', 'txt'), 'w') as txt_file:  
        txt_file.writelines([(f'{key:20} {val}\n') for key, val in params.items()])
    pickle.dump(params, open(file, 'wb'))

def load_parameters(exp_name, run):
    EXP_DIR = f'{OUTPUT_DIR}/runs/{exp_name}/'
    RUN_DIR = f'{EXP_DIR}/{[rd for rd in os.listdir(EXP_DIR) if run in rd][0]}'
    file = f'{RUN_DIR}/params.pkl'
    return pickle.load(open(file, 'rb'))

def create_lossovertime_pkl(metrics_dir):
    print('Recreating loss-over-time pickle from CSVs...', end='')
    data = [pd.read_csv(E, header=[0,1,2], index_col=0) for E in glob.glob(metrics_dir+'/E*.csv')]
    data = pd.concat(data, axis=1)

    # str columns to int....
    order = data.index
    data_stacked = data.stack((1,2))
    data_stacked.columns = data_stacked.columns.astype(int)
    data_stacked = data_stacked.unstack(2).stack(0)
    data_stacked.columns = data_stacked.columns.astype(int)
    data = data_stacked.unstack((1,2)).reindex(order).swaplevel(0,2,axis=1)

    fname = f'{metrics_dir}/all_epochs.pkl'
    data.to_pickle(fname)
    print('Done.')
    return fname

def save_preproc_metrics(run_dir, train_data, test_data):
    samples = []
    for name, arr in train_data.plot_data.items():
        t0_sample = np.random.choice(arr[0].flatten(), int(1e5))
        tn1_sample = np.random.choice(arr[1].flatten(), int(1e5))
        samples.append(pd.Series(t0_sample, name=('train', name, 't_0')))
        samples.append(pd.Series(tn1_sample, name=('train', name, 't_-1')))

    for name, arr in test_data.plot_data.items():
        t0_sample = np.random.choice(arr[0].flatten(), int(1e5))
        tn1_sample = np.random.choice(arr[1].flatten(), int(1e5))
        samples.append(pd.Series(t0_sample, name=('test', name, 't_0')))
        samples.append(pd.Series(tn1_sample, name=('test', name, 't_-1')))

    pd.concat(samples, axis=1).to_csv(f'{run_dir}/preproc_data/preprocessed_data.csv')

def params2text(params):
    text = SPACER +'\n'
    for key, val in params.items():
        if key == 'TIMELAPSE_FILE':
            text += '\n\t>> data parameters <<\n'
        elif key == 'ARCHITECTURE':
            text += '\n\t>> model & training <<\n'
            text += f'\t\t{key:20} {val[1]}\n'
            for layer in val[1:]:
                text += f'\t\t\t\t\t\t\t {layer}\n'
            continue
        elif key == 'LOSS':
            text += '\n\t>> loss <<\n'
        elif key == 'L_OBJECT':
            text += '\n\t>> loss <<\n'
        elif key == 'SEED':
            text += '\n\t>> run settings <<\n'
        text += f'\t\t{key:20} {val}\n'
    text += SPACER+'\n'
    return text

def params2img(fname, params, show=False):
    fontfiles = matplotlib.font_manager.findSystemFonts(fontpaths='/home/loaloa/.fonts/c/', fontext='ttf')
    [matplotlib.font_manager.fontManager.addfont(font_file) for font_file in fontfiles]

    fig, ax = plt.subplots(figsize=(9,9))
    ax.axis('off')
    txt = params2text(params).replace("\t", "    ")
    hfont = {'fontname':'CMU Typewriter Text'}
    ax.text(0,0, txt, fontsize=9, **hfont)

    if show:
        plt.show()
    fig.savefig(fname, dpi=600)
    plt.close()

def check_parameters(passed_params, default_params):
    inval_keys = [key for key in passed_params if key not in default_params]
    if inval_keys:
        print(f'Invalid parameters passed: {inval_keys}')
        exit(1)