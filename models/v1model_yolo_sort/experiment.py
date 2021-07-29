import os
import time

from copy import copy
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler

from config import RAW_DATA_DIR, OUTPUT_DIR, CODE_DIR, PYTORCH_DIR, SPACER
from model import YOLO_AXTrack
from loss import YOLO_AXTrack_loss
from core_functionality import get_default_parameters, setup_model, setup_data, run_epoch # train_fn, test_fn,
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
    )
from plotting import plot_preprocessed_input_data, plot_training_process

def evaluate_run(exp_name, run, recreate=True):
    EXP_DIR = f'{OUTPUT_DIR}/runs/{exp_name}/'
    RUN_DIR = f'{EXP_DIR}/{[rd for rd in os.listdir(EXP_DIR) if run in rd][0]}'
    METRICS_DIR = f'{RUN_DIR}/metrics/'
    PREPROC_DATA_DIR = f'{RUN_DIR}/preproc_data/'

    parameters = load_parameters(exp_name, run)
    params2img(f'{RUN_DIR}/params.png', parameters, show=False)
    print(f'Evaluating model: {exp_name}, {run}')
    print(params2text(parameters))

    preproc_file = f'{PREPROC_DATA_DIR}/preprocessed_data.csv'
    print(preproc_file)
    if os.path.exists(preproc_file):
        plot_preprocessed_input_data(preproc_file, show=False)
    else:
        print('No preprocessed data file found.\n\n\n')
        # save_preproc_metrics(RUN_DIR, train_data, test_data)
        plot_preprocessed_input_data(preproc_file, dest_dir=RUN_DIR, show=False)

    training_file = f'{METRICS_DIR}/all_epochs.pkl'
    if os.path.exists(training_file) and not recreate:
        plot_training_process([training_file], [parameters], dest_dir=RUN_DIR, show=True)
    else:
        training_file = create_lossovertime_pkl(METRICS_DIR)
        plot_training_process([training_file], [parameters], dest_dir=RUN_DIR, show=True)
    
def compare_two_runs(exp_name, runs, show=True):
    EXP_DIR = f'{OUTPUT_DIR}/runs/{exp_name}/'
    training_files = []
    params_list = []
    for run in runs:
        RUN_DIR = f'{EXP_DIR}/{[rd for rd in os.listdir(EXP_DIR) if run in rd][0]}'
        METRICS_DIR = f'{RUN_DIR}/metrics/'
        params_list.append(load_parameters(exp_name, run))
        training_files.append(f'{METRICS_DIR}/all_epochs.pkl')
    
    plot_training_process(training_files, params_list, dest_dir=None, show=show)


def run_experiment(exp_name, parameters, save_results=True):
    torch.manual_seed(parameters['SEED'])
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
    model, loss_fn, optimizer = setup_model(parameters)
    train_data, test_data = setup_data(parameters)

    print_log = []
    for epoch in range(parameters['EPOCHS']):
        print(f'\n\nEpoch {epoch}/{parameters["EPOCHS"]}', flush=True)
        
        # train and optimize
        epoch_loss = run_epoch(train_data, model, loss_fn, 
                               parameters, epoch, optimizer)
        # test data
        epoch_test_loss = run_epoch(test_data, model, loss_fn, 
                                    parameters, epoch)

        # keep track of the loss development
        epoch_log = pd.concat([epoch_loss, epoch_test_loss], axis=1)
        print_log.append(epoch_log.groupby(axis=1, level=1).mean().unstack().rename(epoch))
        
        if save_results:
            # save preprocessing data for plotting 
            if epoch == 0 and parameters['PLOT_PREPROC']:
                save_preproc_metrics(RUN_DIR, train_data, test_data)
            # save the current epoch loss
            epoch_log.to_csv(f'{METRICS_DIR}/E{epoch:0>3}.csv')
            # save the model every 100 Epochs
            if epoch and not (epoch % 100):
                save_checkpoint(model, optimizer, filename=f'{MODELS_DIR}/E{epoch:0>3}.pth')
            
            # check the model predictions every 20 Epochs
            # every 20th epoch, check how the model performs in eval model
            if epoch and epoch%50 == 0:
                epoch_dir = f'{METRICS_DIR}/{epoch:0>3}_results/'
                os.makedirs(epoch_dir)
                model.predict_all(train_data, parameters, dest_dir=epoch_dir)
                                  
                model.predict_all(test_data, parameters, dest_dir=epoch_dir)

        # print out current progress
        current_progress = pd.concat(print_log, axis=1).unstack(level=0)
        current_progress.index = [lbl[6:lbl.rfind('_loss')] for lbl in current_progress.index]
        print(parameters['NOTES'], parameters['DEVICE'])
        print(current_progress.stack(level=1).T.round(3).to_string())





if __name__ == '__main__':
    default_parameters = get_default_parameters()
    exp_name = 'v1Model_exp3_newcnn'
    exp_name = 'v1Model_exp4_NewCNN'

    # clean_rundirs(exp_name, delete_runs=15)
    # evaluate_run(exp_name, 'run09', recreate=False)
    # compare_two_runs(exp_name, ['run09', 'run11'])

    parameters = copy(default_parameters)
    parameters['FROM_CACHE'] = OUTPUT_DIR
    # parameters['CACHE'] = OUTPUT_DIR
    parameters['USE_MOTION_DATA'] = 'include'
    parameters['USE_TRANSFORMS'] = ['vflip', 'hflip', 'rot', 'translateY', 'translateX']
    parameters['USE_TRANSFORMS'] = []
    parameters['NUM_WORKERS'] = 4
    parameters['DROP_LAST'] = False
    parameters['BATCH_SIZE'] = 32
    parameters['SHUFFLE'] = False
    # parameters['LOAD_MODEL'] = [exp_name, 'run13', '100']
    # run_experiment(exp_name, parameters, save_results=False)



    # ========== GPU experiments ===========

    parameters = copy(default_parameters)
    # parameters['DEVICE'] = 'cuda:2'
    # parameters['CACHE'] = OUTPUT_DIR
    parameters['FROM_CACHE'] = OUTPUT_DIR
    parameters['USE_MOTION_DATA'] = 'temp_context'
    parameters['NUM_WORKERS'] = 4
    parameters['USE_TRANSFORMS'] = []

    parameters['BATCH_SIZE'] = 32
    parameters['LR'] = 1e-4
    parameters['NOTES'] = 'first temp context, new CNN'
    ARCHITECTURE = [
        #kernelsize, out_channels, stride, concat_to_feature_vector
        [(3, 20,  2,  5),     # y-x out: 256
         (3, 40,  2,  5),     # y-x out: 128
         (3, 80,  2,  5),     # y-x out: 64
         (3, 80,  1,  1)],     # y-x out: 64
        [(3, 80,  2,  1),     # y-x out: 32
         (3, 160, 2,  1)],     # y-x out: 16
    ]
    parameters['ARCHITECTURE'] = ARCHITECTURE
    run_experiment(exp_name, parameters, save_results=False)
    
    
    parameters = copy(default_parameters)
    parameters['DEVICE'] = 'cuda:0'
    # parameters['CACHE'] = OUTPUT_DIR
    parameters['FROM_CACHE'] = OUTPUT_DIR
    parameters['NUM_WORKERS'] = 4
    parameters['BATCH_SIZE'] = 32
    parameters['USE_MOTION_DATA'] = 'exclude'
    parameters['EPOCHS'] = 601
    parameters['NOTES'] = 'kinda exp4, excl motion'

    # run_experiment(exp_name, parameters, save_results=True)