import os
import glob
import shutil
import pickle
from datetime import datetime
from config import RAW_DATA_DIR, OUTPUT_DIR, CODE_DIR, PYTORCH_DIR, SPACER

from concurrent.futures import ThreadPoolExecutor
from itertools import repeat

from torchvision import models

import pyastar
from scipy import sparse

import numpy as np
import torch
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.font_manager

def yolo_coo2tile_coo(target_tensor, Sx, Sy, tilesize, device, with_batch_dim=True):
    # nobatch dim
    # this creates a bool mask tensor that is false for all pos label entries
    box_mask = (target_tensor == 0).all(-1)

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
    target_tensor[box_mask] = 0
    return target_tensor

def filter_anchors(target_tensor, conf_threshold, ceil_conf=True):
    # convert the target tensor to shape (batchsize, Sx*Sy, 3)
    batch_size, labelsize = target_tensor.shape[0], target_tensor.shape[-1]
    target_tensor = target_tensor.reshape((batch_size, -1, labelsize))
    if ceil_conf:
        target_tensor[..., 0][target_tensor[..., 0]>1] = 1
    
    # split batches into list because of variable sizes 
    frame_targets = list(torch.tensor_split(target_tensor, batch_size))
    # filter each example
    frame_targets_fltd = [tar[tar[:,:,0] >= conf_threshold] for tar in frame_targets]
    return frame_targets_fltd

def fltd_tensor2pandas(target_tensor, sort=True):
    axons_ids = target_tensor[:,3].to(int) if target_tensor.shape[1]>3 else range(target_tensor.shape[0])
    axon_lbls = [f'Axon_{i:0>3}' for i in axons_ids]
    # axon_lbls = pd.MultiIndex.from_product([[t], axon_lbls])
    cols = 'conf', 'anchor_x', 'anchor_y'
    
    target_df = pd.DataFrame(target_tensor[:,:3].to('cpu').numpy(), index=axon_lbls, 
                             columns=cols).convert_dtypes()
    if sort:
        target_df.sort_values('conf', inplace=True)
    return target_df

def Y2pandas_anchors(Y, Sx, Sy, tilesize, device, conf_thr):
    # convert the YOLO coordinates (0-1) to image coordinates (tilesize x tilesize)
    Y_imgcoo = yolo_coo2tile_coo(Y.clone(), Sx, Sy, tilesize, device)

    # filter the (batch, Sx, Sy, 3) output into a list of length batchsize,
    # where the list elements are tensors of shape (#_boxes_above_thr x 3)
    # each element in the list represents a frame/tile
    Ys_list_fltd = filter_anchors(Y_imgcoo, conf_threshold=conf_thr)
    # convert each frame from tensor to pandas
    for i in range(len(Ys_list_fltd)):
        Ys_list_fltd[i] = fltd_tensor2pandas(Ys_list_fltd[i])
    return Ys_list_fltd




















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

def clean_rundirs(exp_name, delete_runs=None, keep_runs=None, keep_only_latest_model=False):
    EXP_DIR = f'{OUTPUT_DIR}/runs/{exp_name}'
    for d in sorted(os.listdir(EXP_DIR)):
        n_metrics = len(glob.glob(f'{EXP_DIR}/{d}/metrics/E*.csv'))
        n_models = len(glob.glob(f'{EXP_DIR}/{d}/models/*.pth'))
        notes = pickle.load(open(f'{EXP_DIR}/{d}/params.pkl', 'rb'))['NOTES']
        print(f'{d} - Epochs: {n_metrics}, models saved: {n_models}, {notes}')

        if keep_only_latest_model and n_models>1:
            [os.remove(model) for model in glob.glob(f'{EXP_DIR}/{d}/models/*.pth')[:-1]]
            print(f'{n_models-1} models --deleted--')
        if delete_runs and n_metrics<delete_runs:
            shutil.rmtree(f'{EXP_DIR}/{d}')
            print('--deleted--\n')
        elif keep_runs is not None and int(d[3:5]) not in keep_runs:
            shutil.rmtree(f'{EXP_DIR}/{d}')
            print('--deleted--\n')

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
    rng = np.random.default_rng()
    smple_indices = rng.choice(train_data.plot_data['Original'][0].size, int(1e6))

    for name, arr in train_data.plot_data.items():
        t0_sample = arr[0].flatten()[smple_indices]
        tn1_sample =arr[1].flatten()[smple_indices]
        samples.append(pd.Series(t0_sample, name=('train', name, 't_0')))
        samples.append(pd.Series(tn1_sample, name=('train', name, 't_-1')))

    for name, arr in test_data.plot_data.items():
        t0_sample = arr[0].flatten()[smple_indices]
        tn1_sample = arr[1].flatten()[smple_indices]
        samples.append(pd.Series(t0_sample, name=('test', name, 't_0')))
        samples.append(pd.Series(tn1_sample, name=('test', name, 't_-1')))

    pd.concat(samples, axis=1).to_csv(f'{run_dir}/preproc_data/preprocessed_data.csv')

def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {"state_dict": model.state_dict(),
                   "optimizer": optimizer.state_dict(),}
    torch.save(checkpoint, filename)

def load_checkpoint(OUTPUT_DIR, load_model, model, optimizer, device):
    print("=> Loading checkpoint...", end='')
    exp_dir = f'{OUTPUT_DIR}/runs/{load_model[0]}/'
    run_dir = [rd for rd in os.listdir(exp_dir) if load_model[1] in rd][0]
    if load_model[2] == 'latest':
        file = [pthfile for pthfile in glob.glob(f'{exp_dir}/{run_dir}/models/*.pth')][-1]
    elif load_model[2]:
        file = f'{exp_dir}/{run_dir}/models/E{load_model[2]}.pth'

    checkpoint = torch.load(file, map_location=torch.device(device))
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    print(f' - {file} - Done.')























def write_parameters(file, params):
    with open(file.replace('pkl', 'txt'), 'w') as txt_file:  
        txt_file.writelines([(f'{key:20} {val}\n') for key, val in params.items()])
    pickle.dump(params, open(file, 'wb'))

def load_parameters(exp_name, run):
    EXP_DIR = f'{OUTPUT_DIR}/runs/{exp_name}/'
    RUN_DIR = f'{EXP_DIR}/{[rd for rd in os.listdir(EXP_DIR) if run in rd][0]}'
    file = f'{RUN_DIR}/params.pkl'
    return pickle.load(open(file, 'rb'))

def params2text(params):
    text = SPACER +'\n'
    for key, val in params.items():
        if key == 'TIMELAPSE_FILE':
            text += '\n\t>> data parameters <<\n'
        elif key == 'ARCHITECTURE':
            text += '\n\t>> model & training <<\n'
            text += f'\t\t{key:20} {val[0][0]}\n'
            for layer in val[0][1:]:
                text += f'\t\t\t\t\t\t\t {layer}\n'
            text += '\t\t\t\t\t\t\t== cat CNN features ==\n'
            for layer in val[1]:
                text += f'\t\t\t\t\t\t\t {layer}\n'
            text += '\t\t\t\t\t\t\t== FullyConnected Head ==\n'
            for layer in val[2]:
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

def to_device_specifc_params(model_parameters, local_default_params):
    to_update = ('TIMELAPSE_FILE', 'LABELS_FILE', 'MASK_FILE', 'DEVICE', 'FROM_CACHE')
    [model_parameters.update({key: local_default_params[key]}) for key in to_update]
    return model_parameters

















def generate_axon_id():
    axon_id = 0
    while True:
        yield axon_id
        axon_id += 1

def compute_astar_path(source, target, weights, return_path=True):
    path_coo = pyastar.astar_path(weights, source, target)
    if return_path:
        ones, row, cols = np.ones(path_coo.shape[0]), path_coo[:,0], path_coo[:,1]
        path = sparse.coo_matrix((ones, (row, cols)), shape=weights.shape, dtype=bool)
        return path_coo.shape[0], path
    return path_coo.shape[0]

def get_detections_distance(t1_det, t0_det, weights, max_dist, fname):
    # print(t1_det, t0_det, weights, max_dist)
    t1id, t1y, t1x = t1_det 
    t0id, t0y, t0x = t0_det
    dist = np.sqrt((t1y-t0y)**2 + (t1x-t0x)**2)
    if dist < max_dist:
        dist, path = compute_astar_path((t1y,t1x), (t0y,t0x), weights)
        sparse.save_npz(f'{fname}_id{t0id}-id{t1id}.npz', path)
    dist = min((dist, max_dist))
    return dist

def compute_astar_distances(t1_dets, t0_dets, mask, max_dist, num_workers, filename):
    # check if a cached file exists, must hace the correct shape (lazy check)
    if os.path.exists(filename+'.npy'):
        distances = np.load(filename+'.npy')
        if distances.shape == (t1_dets.shape[0], t0_dets.shape[0]):
            return distances
    
    # get the cost matrix for finding the astar path
    weights = np.where(mask==1, 1, 2**16).astype(np.float32)
    # get the current ids (placeholders)
    t1_ids = [int(idx[-3:]) for idx in t1_dets.index]
    t0_ids = [int(idx[-3:]) for idx in t0_dets.index]
    
    # iterate t1_detections, compute its distance to all detections in t0_dets
    distances = []
    t0_dets = np.stack([t0_ids, t0_dets.anchor_y, t0_dets.anchor_x], -1)
    for i, t1_det in enumerate(np.stack([t1_ids, t1_dets.anchor_y, t1_dets.anchor_x], -1)):
        print(f'{i}/{len(t1_dets)}', end='...')

        with ThreadPoolExecutor(num_workers) as executer:
            args = repeat(t1_det), t0_dets, repeat(weights), repeat(max_dist), repeat(filename)
            dists = executer.map(get_detections_distance, *args)
        distances.append([d for d in dists])
    distances = np.array(distances)
    np.save(filename+'.npy', distances)
    return distances







def print_models():
    from torchsummary import summary
    models_to_print = [(models.alexnet(), 'AlexNet'),
                       (models.vgg11_bn(), 'vgg11'),
                       (models.vgg19_bn(), 'vgg19'),
                       (models.resnet18(), 'resnet18'),
                       (models.resnext50_32x4d(), 'resnext50'),
                       (models.shufflenet_v2_x0_5(), 'shufflenet')
                    #    (models.densenet121(), 'densenet121'),
                    #    (models.densenet201(), 'densenet201'),
                    #    (models.GoogLeNet(), 'GoogLeNet'),

                    #    (models.inception_v3(), 'Incpetion v3'),
                    #    (models.mnasnet1_0(), 'MNAS Net'),
                    #    (models.mobilenet_v3_large(), 'mobileNet large'),
                    #    (models.mobilenet_v3_small(), 'mobileNet small'),
    ]
    for model, name in models_to_print:
        print()
        print(name)
        model.to('cuda:0')
        summary(model, input_size=(3, 512, 512))
        print()
