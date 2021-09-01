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

def clean_rundirs(exp_name, delete_runs=None, keep_runs=None, keep_only_latest_model=False, filetype='pkl'):
    EXP_DIR = f'{OUTPUT_DIR}/runs/{exp_name}'
    for d in sorted(os.listdir(EXP_DIR)):
        all_epoch_files = glob.glob(f'{EXP_DIR}/{d}/metrics/E*.{filetype}')
        n_metrics = len([f for f in all_epoch_files if not f.endswith(f'_metrics.{filetype}')])
        n_models = len(glob.glob(f'{EXP_DIR}/{d}/models/*.pth'))
        notes = pickle.load(open(f'{EXP_DIR}/{d}/params.pkl', 'rb'))['NOTES']
        print(f'{d} - Epochs: {n_metrics}, models saved: {n_models}, {notes}', flush=True)

        if keep_only_latest_model and n_models>1:
            [os.remove(model) for model in glob.glob(f'{EXP_DIR}/{d}/models/*.pth')[:-1]]
            print(f'{n_models-1} models --deleted--')
        if delete_runs and n_metrics<delete_runs:
            shutil.rmtree(f'{EXP_DIR}/{d}')
            print('--deleted--\n')
        elif keep_runs is not None and int(d[3:d.find('_')]) not in keep_runs:
            shutil.rmtree(f'{EXP_DIR}/{d}')
            print('--deleted--\n')

def get_run_dir(exp_dir, run):
    run_dir = [rd for rd in os.listdir(exp_dir) if run in rd]
    if not run_dir:
        print(f'Run not found: exp_dir: {exp_dir}\nrun: {run}')
        exit()
    return f'{exp_dir}/{run_dir[0]}'

def create_metricsovertime_pkl(metrics_dir):
    print('Recreating metric-over-time pickle from pkl\'s...', end='')
    metrics_files = [f for f in glob.glob(metrics_dir+'/E*.pkl') if f.endswith('_metrics.pkl')]
    metrics = [pd.read_pickle(E) for E in metrics_files]
    metrics = pd.concat(metrics, axis=1)
    
    # select best threshold based on best F1
    maxF1_idx = metrics.loc['F1'].groupby(level=(0,1)).apply(lambda s: s.index[s.argmax()])
    metrics = metrics.loc[:, maxF1_idx]#.droplevel((0,2),1).unstack()

    fname = f'{metrics_dir}/metrics_all_epochs.pkl'
    metrics.to_pickle(fname)
    print('Done.')
    return fname

def create_lossovertime_pkl(metrics_dir):
    print('Recreating loss-over-time pickle from pkl\'s...', end='')
    loss_files = [f for f in glob.glob(metrics_dir+'/E*.pkl') if not f.endswith('_metrics.pkl')]
    loss = [pd.read_pickle(E) for E in loss_files]
    loss = pd.concat(loss, axis=1)

    fname = f'{metrics_dir}/loss_all_epochs.pkl'
    loss.to_pickle(fname)
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

def save_checkpoint(model, optimizer, filename):
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
        file = f'{exp_dir}/{run_dir}/models/E{load_model[2]:0>4}.pth'

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

def merge_axes(axes, fig, merge):
    axes = list(axes.flatten())
    # get size  of 2 bottom right plots
    gs = axes[min(merge)].get_gridspec()
    # remove them
    [axes.pop(ax_i).remove() for ax_i in range(merge)]
    # add one big one
    axes.append(fig.add_subplot(gs[min(merge):]))
    return axes

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

def architecture_to_text(arch):
    text = ''
    empty = ''
    for i, arch_group in enumerate(arch):
        if i == 1:
            text += f'{empty:28}== cat CNN features ==\n'
        elif i == 2:
            text += f'{empty:28}== FullyConnected Head ==\n'
        for j, layer in enumerate(arch_group):
            if i == j == 0:
                text += f'{empty:8} {layer}\n'
            text += f'{empty:28} {layer}\n'
    return text

def prepend_prev_run(exp_name, older_run, newer_run):
    EXP_DIR = f'{OUTPUT_DIR}/runs/{exp_name}/'

    data = []
    for i, run in enumerate([older_run, newer_run]):
        RUN_DIR = get_run_dir(EXP_DIR, run)
        if i == 0:
            last_epoch = dat.columns.unique(0)[-1]
            training_file = f'{RUN_DIR}/metrics/all_epochs.pkl'
            dat = pd.read_pickle(training_file)

        else:
            dat = dat.stack(level=(-1,-2))
            dat.columns = dat.columns.values + last_epoch+1
            dat = dat.unstack(level=(-1,-2))
        data.append(dat)
    data = pd.concat(data, axis=1)
    data.to_pickle(training_file.replace('all_epochs', 'all_epochs_prepend'))