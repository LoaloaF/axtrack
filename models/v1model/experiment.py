import os
from copy import copy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from exp_parameters import (
    get_default_parameters, 
    write_parameters, 
    check_parameters,
    load_parameters, 
    params2text, 
    params2img,
    to_device_specifc_params,
    compare_parameters,
    )
from core_functionality import (
    setup_data, 
    setup_model, 
    run_epoch,
    )
from utils import (
    create_logging_dirs,
    save_preproc_metrics,
    save_checkpoint,
    clean_rundirs,
    create_lossovertime_pkl,
    )
from plotting import (
    plot_preprocessed_input_data, 
    plot_training_process,
    )
from id_association import assign_detections_ids
from config import OUTPUT_DIR, SPACER

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
                assign_detections_ids(data, model, parameters, RUN_DIR)
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
                model.predict_all(train_data, parameters, dest_dir=epoch_dir)
                model.predict_all(test_data, parameters, dest_dir=epoch_dir)

        # print out current progress
        current_progress = pd.concat(print_log, axis=1).unstack(level=0)
        current_progress.index = [lbl[6:lbl.rfind('_loss')] for lbl in current_progress.index]
        print(SPACER)
        print(parameters['NOTES'], parameters['DEVICE'])
        print(current_progress.stack(level=1).T.round(3).to_string())



if __name__ == '__main__':
    default_parameters = get_default_parameters()
    exp2_name = 'v1Model_exp2_boxlossfocus'
    exp3_name = 'v1Model_exp3_newcnn'
    exp4_name = 'v1Model_exp4_NewCNN'
    exp5_name = 'v1Model_exp5_RefactIDFocus'
    
    # from utils import print_models
    # print_models()
    # exit()
    # clean_rundirs(exp5_name, delete_runs=30, keep_only_latest_model=False)

    maxp_arch = [
        #kernelsize, out_channels, stride, groups
        [(3, 20,  2,  1),      # y-x out: 256
         (3, 40,  2,  1),      # y-x out: 128
         (3, 80,  1,  1),      # y-x out: 64
         'M',
         (3, 80,  1,  1),      # y-x out: 64
         (3, 80,  1,  1),      # y-x out: 32
         'M',
         (3, 80,  1,  1),      # y-x out: 32
         (3, 80,  1,  1),      # y-x out: 32
         'M',
         ],      
        [(3, 160, 1,  1),      # y-x out: 16
         ],
        [('FC', 1024),
         ('activation', nn.Sigmoid()), 
         ('FC', 1024),
         ('activation', nn.Sigmoid()), 
        ]     
    ]
    dropout25_arch = [
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
         ('activation', nn.Sigmoid()), 
         ('dropout', 0.25),
         ('FC', 1024),
         ('activation', nn.Sigmoid()), 
        ]     
    ]
    dropout35_arch = [
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
         ('activation', nn.Sigmoid()), 
         ('dropout', 0.35),
         ('FC', 1024),
         ('activation', nn.Sigmoid()), 
        ]     
    ]

    # ========== GPU experiments ===========
    parameters = copy(default_parameters)
    parameters['DEVICE'] = 'cuda:1'
    parameters['NOTES'] = 'T1_baseline'
    # run_experiment(exp5_name, parameters, save_results=True)
    
    parameters = copy(default_parameters)
    parameters['DEVICE'] = 'cuda:2'
    parameters['ARCHITECTURE'] = maxp_arch
    parameters['NOTES'] = 'T1_maxpool'
    # run_experiment(exp5_name, parameters, save_results=True)
    
    parameters = copy(default_parameters)
    parameters['DEVICE'] = 'cuda:3'
    parameters['ARCHITECTURE'] = dropout25_arch
    parameters['NOTES'] = 'T1_dropout0.25'
    # run_experiment(exp5_name, parameters, save_results=True)
    
    parameters = copy(default_parameters)
    parameters['DEVICE'] = 'cuda:0'
    parameters['ARCHITECTURE'] = dropout35_arch
    parameters['NOTES'] = 'T1_dropout0.35'
    # run_experiment(exp5_name, parameters, save_results=True)