import os
import glob
import shutil
import pickle
from datetime import datetime

import numpy as np
import pandas as pd

import torch
from torchvision import models
from torchsummary import summary

from config import OUTPUT_DIR

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

def load_checkpoint(load_model, model, optimizer, device):
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

def print_torchvision_models(initial_in_channels, tilesize):
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
        print(f'\n\n\n{name}\n')
        model.to('cuda:0')
        summary(model, input_size=(initial_in_channels, tilesize, tilesize))