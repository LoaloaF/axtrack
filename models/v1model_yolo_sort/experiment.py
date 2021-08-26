import os
import time

from copy import copy
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler

from config import RAW_DATA_DIR, OUTPUT_DIR, CODE_DIR, PYTORCH_DIR, SPACER
from model import YOLO_AXTrack
from loss import YOLO_AXTrack_loss
from core_functionality import get_default_parameters, setup_model, setup_data, run_epoch
from dataset_class import Timelapse
from utils import (
    create_logging_dirs,
    write_parameters,
    load_parameters,
    save_preproc_metrics,
    params2text,
    params2img,
    check_parameters,
    save_checkpoint,
    clean_rundirs,
    create_lossovertime_pkl,
    to_device_specifc_params,
    )
from plotting import plot_preprocessed_input_data, plot_training_process

def evaluate_run(exp_name, run, which='training', recreate=True, 
                 use_prepend=False, which_data='test', **kwargs):
    EXP_DIR = f'{OUTPUT_DIR}/runs/{exp_name}/'
    RUN_DIR = f'{EXP_DIR}/{[rd for rd in os.listdir(EXP_DIR) if run in rd][0]}'
    parameters = load_parameters(exp_name, run)
    torch.manual_seed(parameters['SEED'])
    np.random.seed(parameters['SEED'])

    # hack for deprecated runs
    if 'TEMPORAL_CONTEXT' not in parameters:
        parameters['STANDARDIZE'] = ('zscore', None)
        parameters['STANDARDIZE_FRAMEWISE'] = False
        parameters['TEMPORAL_CONTEXT'] = 2

    params2img(f'{RUN_DIR}/params.png', parameters, show=False)
    print(f'Evaluating model: {exp_name}, {run}')
    print(params2text(parameters))

    if which == 'preprocessing':
        PREPROC_DATA_DIR = f'{RUN_DIR}/preproc_data/'
        preproc_file = f'{PREPROC_DATA_DIR}/preprocessed_data.csv'
        if not os.path.exists(preproc_file):
            train_data, test_data = setup_data(parameters)
            save_preproc_metrics(RUN_DIR, train_data, test_data)
        plot_preprocessed_input_data(preproc_file, dest_dir=RUN_DIR, show=True)


    elif which == 'training':
        METRICS_DIR = f'{RUN_DIR}/metrics/'
        training_file = f'{METRICS_DIR}/all_epochs.pkl'
        if use_prepend:
            training_file = f'{METRICS_DIR}/all_epochs_prepend.pkl'
        if os.path.exists(training_file) and not recreate:
            plot_training_process([training_file], [parameters], dest_dir=RUN_DIR, 
                                  show=True)
        else:
            training_file = create_lossovertime_pkl(METRICS_DIR)
            plot_training_process([training_file], [parameters], dest_dir=RUN_DIR, 
                                   show=False)
    
    elif which == 'model_out':
        parameters = to_device_specifc_params(parameters, get_default_parameters())
        parameters['LOAD_MODEL'] = [exp_name, run, 'latest']
        parameters['USE_TRANSFORMS'] = []
        parameters['DEVICE'] = 'cpu'

        model, _, _, _ = setup_model(parameters)
        train_data, test_data = setup_data(parameters)
        data = train_data if which_data == 'train' else test_data
        data.construct_tiles(parameters['DEVICE'])

        if which == 'model_out':
            os.makedirs(f'{RUN_DIR}/model_out', exist_ok=True)
            os.makedirs(f'{RUN_DIR}/astar_dists', exist_ok=True)
            if kwargs.get("assign_ids") and not os.path.exists(f'{RUN_DIR}/model_out/{data.name}_assigned_ids.csv'):
                model.assign_detections_ids(data, parameters, RUN_DIR)
            model.predict_all(data, parameters, dest_dir=f'{RUN_DIR}/model_out', **kwargs)
             


    
def compare_two_runs(exp_name, runs, show=True, use_prepend_if_avail=True):
    EXP_DIR = f'{OUTPUT_DIR}/runs/{exp_name}/'
    training_files = []
    params_list = []
    for run in runs:
        RUN_DIR = f'{EXP_DIR}/{[rd for rd in os.listdir(EXP_DIR) if run in rd][0]}'
        METRICS_DIR = f'{RUN_DIR}/metrics/'
        params_list.append(load_parameters(exp_name, run))
        training_file = f'{METRICS_DIR}/all_epochs.pkl'
        if use_prepend_if_avail:
            training_file_prepend = training_file.replace('all_epochs', 'all_epochs_prepend')
            if os.path.exists(training_file_prepend):
                training_file = training_file_prepend
        training_files.append(training_file)
    
    plot_training_process(training_files, params_list, dest_dir=None, show=show)

def prepend_prev_run(exp_name, older_run, newer_run):
    EXP_DIR = f'{OUTPUT_DIR}/runs/{exp_name}/'

    data = []
    for i, run in enumerate([older_run, newer_run]):
        RUN_DIR = f'{EXP_DIR}/{[rd for rd in os.listdir(EXP_DIR) if run in rd][0]}'
        training_file = f'{RUN_DIR}/metrics/all_epochs.pkl'
        dat = pd.read_pickle(training_file)

        if i == 0:
            last_epoch = dat.columns.unique(0)[-1]
        else:
            dat = dat.stack(level=(-1,-2))
            dat.columns = dat.columns.values + last_epoch+1
            dat = dat.unstack(level=(-1,-2))
        data.append(dat)
    data = pd.concat(data, axis=1)
    data.to_pickle(training_file.replace('all_epochs', 'all_epochs_prepend'))
    

def run_experiment(exp_name, parameters, save_results=True):
    torch.manual_seed(parameters['SEED'])
    np.random.seed(parameters['SEED'])
    torch.cuda.empty_cache()
    # torch.multiprocessing.set_start_method('spawn')

    # Setup saving and parameter checking 
    print(f'Running Experiment: {exp_name}', flush=True)
    check_parameters(parameters, get_default_parameters())
    if save_results:
        dirs, run_label = create_logging_dirs(exp_name)
        RUN_DIR, MODELS_DIR, METRICS_DIR, PREPROC_DATA_DIR = dirs
        write_parameters(f'{RUN_DIR}/params.pkl', parameters)
        params2img(f'{RUN_DIR}/params.png', parameters, show=False)
        print('\tSaving: ', run_label)
    else:
        print('\tRun is not saved!')
    print(params2text(parameters), flush=True)

    # setup model and data
    model, loss_fn, optimizer, lr_scheduler = setup_model(parameters)
    train_data, test_data = setup_data(parameters)
    # train_data.construct_tiles(parameters['DEVICE'])
    # test_data.construct_tiles(parameters['DEVICE'])
    
    # PREPROC_DATA_DIR = f'{RUN_DIR}/preproc_data/'
    # preproc_file = f'{PREPROC_DATA_DIR}/preprocessed_data.csv'
    # save_preproc_metrics(RUN_DIR, train_data, test_data)
    # plot_preprocessed_input_data(preproc_file, dest_dir=RUN_DIR, show=True)




    print_log = []                      
    for epoch in range(parameters['EPOCHS']):
        print(f'\n\n\nEpoch {epoch}/{parameters["EPOCHS"]}', flush=True)
        
        # train and optimize
        epoch_loss = run_epoch(train_data, model, loss_fn, 
                               parameters, epoch, optimizer, lr_scheduler)
        # test data
        model.eval()
        epoch_test_loss = run_epoch(test_data, model, loss_fn, 
                                    parameters, epoch)
        model.train()

        # keep track of the loss development
        epoch_log = pd.concat([epoch_loss, epoch_test_loss], axis=1)
        print_log.append(epoch_log.groupby(axis=1, level=1).mean().unstack().rename(epoch))
        
        if save_results:
            # save preprocessing of input data for plotting 
            if epoch == 0 and parameters['PLOT_PREPROC']:
                save_preproc_metrics(RUN_DIR, train_data, test_data)
            # save the current epoch loss
            epoch_log.to_csv(f'{METRICS_DIR}/E{epoch:0>3}.csv')
            # save the model every 100 Epochs
            if epoch and not (epoch % 500):
                save_checkpoint(model, optimizer, filename=f'{MODELS_DIR}/E{epoch:0>4}.pth')
            
            # check the model predictions every 20 Epochs
            # every 20th epoch, check how the model performs in eval model
            if epoch and epoch%500 == 0:
                epoch_dir = f'{METRICS_DIR}/{epoch:0>3}_results/'
                os.makedirs(epoch_dir)
                model.predict_all(train_data, parameters['DEVICE'], 
                                  parameters['BBOX_THRESHOLD'], dest_dir=epoch_dir, 
                                  show=False)
                model.predict_all(test_data, parameters['DEVICE'], 
                                  parameters['BBOX_THRESHOLD'], dest_dir=epoch_dir, 
                                  show=False)

        # print out current progress
        current_progress = pd.concat(print_log, axis=1).unstack(level=0)
        current_progress.index = [lbl[6:lbl.rfind('_loss')] for lbl in current_progress.index]
        print(SPACER)
        print(parameters['NOTES'], parameters['DEVICE'])
        print(current_progress.stack(level=1).T.round(3).to_string())





if __name__ == '__main__':
    default_parameters = get_default_parameters()
    exp_name = 'v1Model_exp2_boxlossfocus'
    exp_name = 'v1Model_exp3_newcnn'
    exp_name = 'v1Model_exp4_NewCNN'
    # from utils import print_models
    # print_models()
    # exit()

    clean_rundirs(exp_name, delete_runs=50, keep_only_latest_model=True)
    
    # prepend_prev_run(exp_name, 'run09', 'run23')
    # evaluate_run(exp_name, 'run99', which='training', recreate=False)
    # compare_two_runs(exp_name, ['run90', 'run99'])
    # evaluate_run(exp_name, 'run94', which='training', recreate=False)
    # evaluate_run(exp_name, 'run95', which='training', recreate=False)
    # evaluate_run(exp_name, 'run96', which='training', recreate=False)
    # compare_two_runs(exp_name, ['run99', 'run59'])
    # evaluate_run(exp_name, 'run59', which='model_out', assign_ids=True, animated=True, show=False, which_data='train')

    new_ARCHITECTURE = [
        #kernelsize, out_channels, stride, groups
        [(3, 20,  2,  1),      # y-x out: 256
         (3, 40,  2,  1),      # y-x out: 128
         (3, 80,  2,  1),      # y-x out: 64
         (3, 80,  1,  1),      # y-x out: 64
         (3, 80,  2,  1),      # y-x out: 32
         (3, 80,  1,  1),      # y-x out: 32
         (3, 80,  1,  1),      # y-x out: 32
         ],      
        [(3, 160, 2,  1),      # y-x out: 16
         ],
        [('FC', 1024),
         ('activation', torch.nn.Sigmoid()), 
         ('FC', 1024),
         ('activation', torch.nn.Sigmoid()), 
        ]     
    ]

    parameters = copy(default_parameters)
    # parameters['CACHE'] = OUTPUT_DIR
    # parameters['DEVICE'] = 'cpu'
    # parameters['FROM_CACHE'] = None
    parameters['USE_MOTION_DATA'] = 'exclude'
    parameters['NUM_WORKERS'] = 4
    parameters['BATCH_SIZE'] = 32
    parameters['LR'] = 1e-4
    parameters['SHUFFLE'] = True
    parameters['USE_TRANSFORMS'] = []
    parameters['USE_TRANSFORMS'] = ['vflip', 'hflip', 'rot', 'translateY', 'translateX']
    parameters['BBOX_THRESHOLD'] = 4
    # parameters['TEST_TIMEPOINTS'] = [4,5]
    # parameters['TRAIN_TIMEPOINTS'] = [11,12,13]
    parameters['ARCHITECTURE'] = new_ARCHITECTURE
    parameters['TEMPORAL_CONTEXT'] = 2
    parameters['NOTES'] = 'trying FP TP stuff'
    run_experiment(exp_name, parameters, save_results=True)


    new_ARCHITECTURE_deeper = [
        #kernelsize, out_channels, stride, concat_to_feature_vector

        [(3, 20,  2,  1),      # y-x out: 256
         (3, 40,  2,  1),      # y-x out: 128

         (3, 80,  2,  1),      # y-x out: 64
         (3, 80,  1,  1),     # y-x out: 64
         (3, 80,  2,  1),      # y-x out: 32
         # new
         (3, 80,  1,  1),      # y-x out: 32
         (3, 80,  1,  1)],      # y-x out: 32
         
         [(3, 160, 2,  1)],     # y-x out: 16
    ]
    new_ARCHITECTURE_deeper_Maxpool = [
        #kernelsize, out_channels, stride, concat_to_feature_vector

        [(3, 20,  2,  1),      # y-x out: 256
         (3, 40,  2,  1),      # y-x out: 128

         (3, 80,  1,  1),      # y-x out: 64
         'M',
         (3, 80,  1,  1),     # y-x out: 64
         (3, 80,  1,  1),      #   y-x out: 32
         'M',
         
         # new
         (3, 80,  1,  1),      # y-x out: 32
         (3, 80,  1,  1),      # y-x out: 32
         'M'],
         
         [(3, 160, 1,  1),],     # y-x out: 16
    ]

    # ========== GPU experiments ===========
    parameters = copy(default_parameters)
    # parameters['DEVICE'] = 'cuda:1'
    
    parameters = load_parameters(exp_name, 'run59')
    # parameters['DEVICE'] = 'cpu'
    # parameters['FROM_CACHE'] = OUTPUT_DIR
    # parameters['LR_DECAYRATE'] = 15
    # parameters['LR'] = 5e-4
    # parameters['EPOCHS'] = 2001
    # parameters['NUM_WORKERS'] = 6
    # parameters['FROM_CACHE'] = None
    parameters['FROM_CACHE'] = None
    parameters['CACHE'] = OUTPUT_DIR
    parameters['STANDARDIZE_FRAMEWISE'] = True
    parameters['TEMPORAL_CONTEXT'] = 2
    parameters['STANDARDIZE'] = ('zscore', None)
    parameters['NOTES'] = 'trying to reproduce R59'
    # run_experiment(exp_name, parameters, save_results=True)
    



    parameters = load_parameters(exp_name, 'run59')
    parameters['DEVICE'] = 'cuda:1'
    parameters['EPOCHS'] = 2001
    parameters['NUM_WORKERS'] = 3
    parameters['NOTES'] = 'r59params Dropout(FC now) 0.5, higher'
    # run_experiment(exp_name, parameters, save_results=True)
    
    parameters = load_parameters(exp_name, 'run59')
    parameters['DEVICE'] = 'cuda:2'
    parameters['EPOCHS'] = 2001
    parameters['NUM_WORKERS'] = 3
    parameters['NOTES'] = 'TPR thr (all4) r59params Dropout(FC now) 0.15, lower'
    # run_experiment(exp_name, parameters, save_results=True)
    
    parameters = load_parameters(exp_name, 'run59')
    parameters['DEVICE'] = 'cuda:3'
    parameters['EPOCHS'] = 2001
    parameters['NUM_WORKERS'] = 3
    parameters['ARCHITECTURE'] = new_ARCHITECTURE_deeper_Maxpool
    parameters['NOTES'] = 'r59params MaxPoolArch.'
    # run_experiment(exp_name, parameters, save_results=True)