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

import matplotlib.pyplot as plt

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
    return (RUN_DIR, MODELS_DIR, METRICS_DIR), run_label

def clean_rundirs(exp_name, delete_runs_min_epochs=None, delete_all_except=None, 
                  keep_only_latest_model=False, filetype='pkl'):
    # check for valid input
    if delete_all_except is not None:
        if not all([isinstance(run, int) for run in delete_all_except]):
            print('`delete_all_except` should only contain integers.')
            exit(1)

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
        if delete_runs_min_epochs and n_metrics<delete_runs_min_epochs:
            shutil.rmtree(f'{EXP_DIR}/{d}')
            print('--deleted--\n')
        elif delete_all_except is not None and int(d[3:d.find('_')]) not in delete_all_except:
            shutil.rmtree(f'{EXP_DIR}/{d}')
            print('--deleted--\n')

def get_run_dir(exp_dir, run):
    run_dir = [rd for rd in os.listdir(exp_dir) if run in rd]
    if not run_dir:
        print(f'Run not found: exp_dir: {exp_dir}\nrun: {run}')
        exit()
    return f'{exp_dir}/{run_dir[0]}'

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
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

def save_preproc_metrics(dest_dir, dataset1, dataset2=None):
    samples = []
    rng = np.random.default_rng()
    smple_indices = rng.choice(dataset1.plot_data['Original'][0].size, int(1e6))

    for which_step, arr in dataset1.plot_data.items():
        t0_sample = arr[0].flatten()[smple_indices]
        tn1_sample =arr[1].flatten()[smple_indices]
        samples.append(pd.Series(t0_sample, name=(dataset1.name, which_step, 't_0')))
        samples.append(pd.Series(tn1_sample, name=(dataset1.name, which_step, 't_-1')))

    if dataset2 is not None:
        for which_step, arr in dataset2.plot_data.items():
            t0_sample = arr[0].flatten()[smple_indices]
            tn1_sample = arr[1].flatten()[smple_indices]
            samples.append(pd.Series(t0_sample, name=(dataset2.name, which_step, 't_0')))
            samples.append(pd.Series(tn1_sample, name=(dataset2.name, which_step, 't_-1')))

    fname = f'{dest_dir}/preprocessed_data.csv'
    pd.concat(samples, axis=1).to_csv(fname)
    return fname

def create_all_epochs_info(metrics_dir):
    # glue all epochs together to on df
    info_files = sorted([f for f in glob.glob(metrics_dir+'/E*.pkl')])
    all_epochs_info = pd.concat([pd.read_pickle(E) for E in info_files], axis=0)

    # slice to metrics (prc, rcl F1) including all the different thresholds
    all_epochs_metrics = all_epochs_info.iloc[:,10:]
    multi_idx = [(*col[0],col[1]) for col in all_epochs_metrics.columns]
    all_epochs_metrics.columns = pd.MultiIndex.from_tuples(multi_idx)
    
    # select best thr based on max mean F1 on test data
    bestF1_thr = all_epochs_metrics.loc[:,('F1', slice(None), 'test')].mean(0).idxmax()[1]
    best_thr_metric = all_epochs_metrics.loc[:,(slice(None), bestF1_thr)].droplevel(1,1)
    # glue together with loss terms
    all_epochs_info = pd.concat((all_epochs_info.iloc[:,:10], best_thr_metric), axis=1)
    
    fname = f'{metrics_dir}/loss_all_epochs.pkl'
    all_epochs_info.to_pickle(fname)
    fname = f'{metrics_dir}/metrics_all_epochs.pkl'
    all_epochs_metrics.swaplevel(axis=1).to_pickle(fname)

def get_all_epoch_data(exp_name, run, recreate=False, use_prepend_ifavail=True):
    EXP_DIR = f'{OUTPUT_DIR}/runs/{exp_name}/'
    RUN_DIR = get_run_dir(EXP_DIR, run)

    all_epochs_info_fname = f'{RUN_DIR}/metrics/loss_all_epochs.pkl'
    all_epochs_metrics_fname = f'{RUN_DIR}/metrics/metrics_all_epochs.pkl'

    if use_prepend_ifavail:
        all_epochs_info_prd_fname = all_epochs_info_fname.replace('.pkl', '_prepend.pkl')
        all_epochs_metrics_prd_fname = all_epochs_metrics_fname.replace('.pkl', '_prepend.pkl')
        if os.path.exists(all_epochs_info_prd_fname):
            all_epochs_info_fname = all_epochs_info_prd_fname
            all_epochs_metrics_fname = all_epochs_metrics_prd_fname

    if not os.path.exists(all_epochs_info_fname) or recreate:
        create_all_epochs_info(f'{RUN_DIR}/metrics/')
    
    all_epochs_info = pd.read_pickle(all_epochs_info_fname)
    all_epochs_metrics = pd.read_pickle(all_epochs_metrics_fname)
    return all_epochs_info, all_epochs_metrics

def prepend_prev_run(exp_name, older_run, newer_run, older_run_until_e=None, 
                     newer_run_until_e=None):
    print(f'Prepending {older_run} to {newer_run} (<= output).')
    EXP_DIR = f'{OUTPUT_DIR}/runs/{exp_name}/'
    RUN_DIR = get_run_dir(EXP_DIR, newer_run)
    
    data = []
    metrics_data = []
    args = [older_run, newer_run], [older_run_until_e, newer_run_until_e]
    for i, (run, until_e) in enumerate(zip(*args)):

        all_epochs_info, all_epochs_metrics = get_all_epoch_data(exp_name, run, use_prepend_ifavail=False)
        if until_e:
            all_epochs_info = all_epochs_info.loc[:until_e]
            all_epochs_metrics = all_epochs_metrics.loc[:until_e]

        if i == 0:
            last_epoch = all_epochs_info.index[-1]
            old_all_epochs_info, old_all_epochs_metrics = all_epochs_info, all_epochs_metrics
        else:
            all_epochs_info.index += last_epoch+1
            all_epochs_metrics.index += last_epoch+1

            all_epochs_info = pd.concat((old_all_epochs_info, all_epochs_info), sort=False)
            all_epochs_metrics = pd.concat((old_all_epochs_metrics, all_epochs_metrics), sort=False)

            all_epochs_info.to_pickle(f'{RUN_DIR}/metrics/loss_all_epochs_prepend.pkl')
            all_epochs_metrics.to_pickle(f'{RUN_DIR}/metrics/metrics_all_epochs_prepend.pkl')

# def create_metricsovertime_pkl(metrics_dir):
#     print('Recreating metric-over-time pickle from pkl\'s...', end='')
#     metrics_files = [f for f in glob.glob(metrics_dir+'/E*.pkl') if f.endswith('_metrics.pkl')]
#     metrics = [pd.read_pickle(E) for E in metrics_files]
#     metrics = pd.concat(metrics, axis=1)
#     if metrics.columns.nlevels != 3:
#         metrics.columns = pd.MultiIndex.from_tuples(metrics.columns)
    
#     # select best threshold based on best F1
#     maxF1_idx = metrics.loc['F1'].groupby(level=(0,1)).apply(lambda s: s.index[s.argmax()])
#     metrics = metrics.loc[:, maxF1_idx]#.droplevel((0,2),1).unstack()

#     fname = f'{metrics_dir}/metrics_all_epochs.pkl'
#     metrics.to_pickle(fname)
#     print('Done.')
#     return fname

# def create_lossovertime_pkl(metrics_dir):
#     print('Recreating loss-over-time pickle from pkl\'s...', end='')
#     loss_files = [f for f in glob.glob(metrics_dir+'/E*.pkl') if not f.endswith('_metrics.pkl')]
#     loss = [pd.read_pickle(E) for E in loss_files]
#     loss = pd.concat(loss, axis=1).sort_index(1)

#     fname = f'{metrics_dir}/loss_all_epochs.pkl'
#     loss.to_pickle(fname)
#     print('Done.')
#     return fname

# def prepend_prev_run(exp_name, older_run, newer_run, older_run_until_e=None, newer_run_until_e=None):
#     pass
#     EXP_DIR = f'{OUTPUT_DIR}/runs/{exp_name}/'

#     data = []
#     metrics_data = []
#     for i, run in enumerate([older_run, newer_run]):
#         RUN_DIR = get_run_dir(EXP_DIR, run)
        
#         if i == 0:
#             training_file = f'{RUN_DIR}/metrics/loss_all_epochs.pkl'
#             training_file_metrics = f'{RUN_DIR}/metrics/metrics_all_epochs.pkl'
#         else:
#             training_file = f'{RUN_DIR}/metrics/loss_all_epochs.pkl'
#             training_file_metrics = f'{RUN_DIR}/metrics/metrics_all_epochs.pkl'

#         dat = pd.read_pickle(training_file)
        
#         dat_metrics = pd.read_pickle(training_file_metrics)
#         dat_metrics.dropna(how='all',axis=1, inplace=True)
#         if i == 0:
#             if older_run_until_e:
#                 dat = dat.T[dat.columns.get_level_values(0) <=older_run_until_e].T
#                 dat_metrics = dat_metrics.T[dat_metrics.columns.get_level_values(0) <=older_run_until_e].T

#             last_epoch = dat.columns.unique(0)[-1]
#             last_epoch_metrics = dat_metrics.columns.unique(0)[-1]
#         else:
#             if newer_run_until_e:
#                 dat = dat.T[dat.columns.get_level_values(0) <=newer_run_until_e].T
#                 dat_metrics = dat_metrics.T[dat_metrics.columns.get_level_values(0) <=newer_run_until_e].T
#             dat = dat.stack(level=(-1,-2))
#             dat.columns = dat.columns.values + last_epoch+1
#             dat = dat.unstack(level=(-1,-2))

#             dat_metrics = dat_metrics.stack(level=(-1,-2))
#             dat_metrics.columns = dat_metrics.columns.values + last_epoch_metrics+1
#             dat_metrics = dat_metrics.unstack(level=(-1,-2))
#             dat_metrics.dropna(how='all',axis=1, inplace=True)
#         data.append(dat)
#         metrics_data.append(dat_metrics)
#     data = pd.concat(data, axis=1)
#     metrics_data = pd.concat(metrics_data, axis=1)
#     data.to_pickle(training_file.replace('all_epochs', 'all_epochs_prepend'))
#     # print(data)
#     metrics_data.to_pickle(training_file_metrics.replace('all_epochs', 'all_epochs_prepend'))

def save_checkpoint(model, optimizer, lr_schedular, filename):
    print("=> Saving model checkpoint")
    checkpoint = {"state_dict": model.state_dict(),
                   "optimizer": optimizer.state_dict(),
                   "lr_schedular": lr_schedular.state_dict(), }
    torch.save(checkpoint, filename)

def load_checkpoint(load_model, model, optimizer, lr_schedular, device):
    print("=> Loading model checkpoint...", end='')
    # at inference/ deployment (skip LR schedular and optimizer )
    if isinstance(load_model, str):
        file = glob.glob(f'{load_model}/*.pth')[0]
        checkpoint = torch.load(file, map_location=torch.device(device))
        model.load_state_dict(checkpoint["state_dict"])
    
    else:
        exp_dir = f'{OUTPUT_DIR}/runs/{load_model[0]}/'
        run_dir = [rd for rd in os.listdir(exp_dir) if load_model[1] in rd][0]
        if load_model[2] == 'latest':
            file = [pthfile for pthfile in glob.glob(f'{exp_dir}/{run_dir}/models/*.pth')][-1]
        elif load_model[2]:
            file = f'{exp_dir}/{run_dir}/models/E{load_model[2]:0>4}.pth'

        checkpoint = torch.load(file, map_location=torch.device(device))
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if 'lr_schedular' in checkpoint:
            lr_schedular.load_state_dict(checkpoint["lr_schedular"])
    print(f' - {file} - Done.\n', flush=True)

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

# def merge_axes(axes, fig, merge):
#     axes = list(axes.flatten())
#     # get size  of 2 bottom right plots
#     gs = axes[min(merge)].get_gridspec()
#     # remove them
#     [axes.pop(ax_i).remove() for ax_i in range(merge)]
#     # add one big one
#     axes.append(fig.add_subplot(gs[min(merge):]))
#     return axes

def turn_tex(on_off):
    if on_off == 'on':
        plt.rc('text', usetex=True)
        plt.rc('text.latex', preamble=r'\usepackage{amsmath} \usepackage{upgreek} \usepackage{underscore}')
    elif on_off == 'off':
        plt.rc('text', usetex=False)

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
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

def get_data_standardization_scaler(fname):
    with open(fname, 'rb') as file:
        stnd_scaler = pickle.load(file)
    return stnd_scaler    
