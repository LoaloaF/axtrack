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
    to_device_specifc_params,
    )
from plotting import plot_preprocessed_input_data, plot_training_process

def evaluate_run(exp_name, run, recreate=True, use_prepend=False, 
                 include_processing=True, include_training=True,
                 include_ID_association=False, which_data='test'):

    EXP_DIR = f'{OUTPUT_DIR}/runs/{exp_name}/'
    RUN_DIR = f'{EXP_DIR}/{[rd for rd in os.listdir(EXP_DIR) if run in rd][0]}'
    METRICS_DIR = f'{RUN_DIR}/metrics/'
    PREPROC_DATA_DIR = f'{RUN_DIR}/preproc_data/'

    parameters = load_parameters(exp_name, run)
    torch.manual_seed(parameters['SEED'])
    params2img(f'{RUN_DIR}/params.png', parameters, show=False)
    print(f'Evaluating model: {exp_name}, {run}')
    print(params2text(parameters))

    if include_processing:
        preproc_file = f'{PREPROC_DATA_DIR}/preprocessed_data.csv'
        print(preproc_file)
        if os.path.exists(preproc_file):
            plot_preprocessed_input_data(preproc_file, show=False)
        else:
            print('No preprocessed data file found.\n\n\n')
            # save_preproc_metrics(RUN_DIR, train_data, test_data)
            plot_preprocessed_input_data(preproc_file, dest_dir=RUN_DIR, show=False)

    if include_training:
        training_file = f'{METRICS_DIR}/all_epochs.pkl'
        if use_prepend:
            training_file = f'{METRICS_DIR}/all_epochs_prepend.pkl'
        if os.path.exists(training_file) and not recreate:
            plot_training_process([training_file], [parameters], dest_dir=RUN_DIR, show=True)
        else:
            training_file = create_lossovertime_pkl(METRICS_DIR)
            plot_training_process([training_file], [parameters], dest_dir=RUN_DIR, show=False)
    
    if include_ID_association:
        parameters = to_device_specifc_params(parameters, get_default_parameters())
        parameters['LOAD_MODEL'] = [exp_name, run, 'latest']   
        parameters['USE_TRANSFORMS'] = []
        parameters['DEVICE'] = 'cpu'

        model, _, _, _ = setup_model(parameters)
        train_data, test_data = setup_data(parameters)
        data = train_data if which_data == 'train' else test_data
        data.construct_tiles(parameters['DEVICE'])

        os.makedirs(f'{RUN_DIR}/astar_dists', exist_ok=True)
        model.assign_ids(data, parameters['DEVICE'], RUN_DIR)

    
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
    # model.assign_ids(test_data)
    # exit()

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
            # save preprocessing data for plotting 
            if epoch == 0 and parameters['PLOT_PREPROC']:
                save_preproc_metrics(RUN_DIR, train_data, test_data)
            # save the current epoch loss
            epoch_log.to_csv(f'{METRICS_DIR}/E{epoch:0>3}.csv')
            # save the model every 100 Epochs
            if epoch and not (epoch % 500):
                save_checkpoint(model, optimizer, filename=f'{MODELS_DIR}/E{epoch:0>3}.pth')
            
            # check the model predictions every 20 Epochs
            # every 20th epoch, check how the model performs in eval model
            if epoch and epoch%500 == 0:
                epoch_dir = f'{METRICS_DIR}/{epoch:0>3}_results/'
                os.makedirs(epoch_dir)
                model.predict_all(train_data, parameters, dest_dir=epoch_dir, show=False)
                model.predict_all(test_data, parameters, dest_dir=epoch_dir, show=False)

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

    # clean_rundirs(exp_name, delete_runs=1, keep_only_latest_model=False)
    # prepend_prev_run(exp_name, 'run09', 'run23')
    # compare_two_runs(exp_name, ['run58', 'run59'])
    # evaluate_run(exp_name, 'run66', recreate=False)
    # evaluate_run(exp_name, 'run67', recreate=False)
    # evaluate_run(exp_name, 'run68', recreate=False)

    evaluate_run(exp_name, 'run26', include_processing=False, include_training=False, 
                 include_ID_association=True)

    # new_ARCHITECTURE = [
    #     #kernelsize, out_channels, stride, groups, 2nd list: concat to features inp
    #     [(3, 2,  2,  1),      # y-x out: 256
    #      (3, 4,  2,  1),      # y-x out: 1282
    #      (3, 8,  1,  1),      # y-x out: 128
    #      (3, 8,  2,  1)],     # y-x out: 64
    #     [(3, 8,  2,  1),      # y-x out: 32
    #      (3, 16, 2,  1)],     # y-x out: 16
    # ]

    parameters = copy(default_parameters)
    parameters['CACHE'] = OUTPUT_DIR
    # parameters['DEVICE'] = 'cpu'
    # parameters['FROM_CACHE'] = OUTPUT_DIR
    parameters['USE_MOTION_DATA'] = 'exclude'
    parameters['NUM_WORKERS'] = 4
    parameters['BATCH_SIZE'] = 32
    parameters['LR'] = 1e-4
    parameters['SHUFFLE'] = True
    parameters['DROP_LAST'] = False
    parameters['USE_TRANSFORMS'] = ['vflip', 'hflip', 'rot', 'translateY', 'translateX']
    parameters['USE_TRANSFORMS'] = []
    # parameters['ARCHITECTURE'] = 'resnet'
    parameters['NOTES'] = 'trying ID stuff'
    # run_experiment(exp_name, parameters, save_results=False)
    
    new_ARCHITECTURE = [
        #kernelsize, out_channels, stride, concat_to_feature_vector
        [(3, 20,  2,  1),      # y-x out: 256
         (3, 40,  2,  1),      # y-x out: 128

         (3, 80,  2,  1),      # y-x out: 64
         (3, 80,  1,  1),     # y-x out: 64
         (3, 80,  2,  1)],      # y-x out: 32
         [(3, 160, 2,  1)],     # y-x out: 16
    ]
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
    new_ARCHITECTURE_moreCs = [
        #kernelsize, out_channels, stride, concat_to_feature_vector
        [(3, 20,  2,  1),      # y-x out: 256
         (3, 40,  2,  1),      # y-x out: 128
         
         # everyone more Cs
         (3, 120,  2,  1),      # y-x out: 64
         (3, 120,  1,  1),     # y-x out: 64
         (3, 160,  2,  1)],      # y-x out: 32
         [(3, 240, 2,  1)],     # y-x out: 16
    ]
    new_ARCHITECTURE_evenDdeeper = [
        #kernelsize, out_channels, stride, concat_to_feature_vector
        [(3, 20,  2,  1),      # y-x out: 256
         (3, 40,  2,  1),      # y-x out: 128

         # eeryone more Cs2
         (3, 80,  2,  1),      # y-x out: 64
         (3, 80,  1,  1),     # y-x out: 64
         (3, 80,  2,  1),     # y-x out: 32
         
         # new
         (3, 80,  1,  1),     # y-x out: 32
         (3, 80,  1,  1),      # y-x out: 32
         
         (3, 120,  1,  1),      # y-x out: 32
         (3, 120,  1,  1)],      # y-x out: 32
         
         [(3, 120, 2,  1)],     # y-x out: 16
    ]
    
    # ========== GPU experiments ===========
    parameters = load_parameters(exp_name, 'run59')
    parameters['DEVICE'] = 'cuda:1'
    parameters['EPOCHS'] = 2001
    parameters['NUM_WORKERS'] = 4
    parameters['ARCHITECTURE'] = new_ARCHITECTURE_evenDdeeper
    parameters['NOTES'] = 'r59params deeper arch.'
    # run_experiment(exp_name, parameters, save_results=True)
    
    parameters = load_parameters(exp_name, 'run59')
    parameters['DEVICE'] = 'cuda:0'
    parameters['EPOCHS'] = 2001
    parameters['NUM_WORKERS'] = 4
    parameters['NOTES'] = 'r59params Dropout 0.2'
    # run_experiment(exp_name, parameters, save_results=True)
    
    parameters = load_parameters(exp_name, 'run59')
    parameters['DEVICE'] = 'cuda:2'
    parameters['EPOCHS'] = 2001
    parameters['NUM_WORKERS'] = 4
    parameters['WEIGHT_DECAY'] = 5e-5
    parameters['NOTES'] = 'r59params lower weight decay'
    # run_experiment(exp_name, parameters, save_results=True)
    
    parameters = load_parameters(exp_name, 'run59')
    parameters['DEVICE'] = 'cuda:3'
    parameters['EPOCHS'] = 2001
    parameters['NUM_WORKERS'] = 4
    parameters['WEIGHT_DECAY'] = 5e-3
    parameters['NOTES'] = 'r59params higher weight decay'
    # run_experiment(exp_name, parameters, save_results=True)
    





    
    parameters = copy(default_parameters)
    parameters['FROM_CACHE'] = OUTPUT_DIR
    parameters['NUM_WORKERS'] = 6
    parameters['EPOCHS'] = 3001
    parameters['LR'] = 5e-4
    parameters['USE_MOTION_DATA'] = 'exclude'
    parameters['ARCHITECTURE'] = new_ARCHITECTURE_deeper
    parameters['LR_DECAYRATE'] = 20
    parameters['NOTES'] = 'deeper.Arch., 1024x1024, TPR thr, Exp5-decay=20 (high LRs)'
    # run_experiment(exp_name, parameters, save_results=True)
    
    parameters = copy(default_parameters)
    parameters['DEVICE'] = 'cuda:2'
    parameters['FROM_CACHE'] = OUTPUT_DIR
    parameters['NUM_WORKERS'] = 3
    parameters['EPOCHS'] = 3001
    parameters['LR'] = 5e-4
    parameters['USE_MOTION_DATA'] = 'exclude'
    parameters['ARCHITECTURE'] = new_ARCHITECTURE_deeper
    parameters['LR_DECAYRATE'] = 15
    parameters['NOTES'] = 'deeper.Arch., 1024x1024, TPR thr, Exp5-decay=15 (mid LRs)'
    # run_experiment(exp_name, parameters, save_results=True)
    
    parameters = copy(default_parameters)
    parameters['DEVICE'] = 'cuda:3'
    parameters['FROM_CACHE'] = OUTPUT_DIR
    parameters['NUM_WORKERS'] = 3
    parameters['EPOCHS'] = 3001
    parameters['LR'] = 5e-4
    parameters['USE_MOTION_DATA'] = 'exclude'
    parameters['ARCHITECTURE'] = new_ARCHITECTURE_deeper
    parameters['LR_DECAYRATE'] = 10
    parameters['NOTES'] = 'deeper.Arch., 1024x1024, TPR thr, Exp5-decay=10 (low LRs)'
    # run_experiment(exp_name, parameters, save_results=True)
    










    parameters = copy(default_parameters)
    parameters['DEVICE'] = 'cuda:1'
    parameters['FROM_CACHE'] = OUTPUT_DIR
    parameters['NUM_WORKERS'] = 3
    parameters['EPOCHS'] = 1501
    parameters['LR'] = 1e-4
    parameters['USE_MOTION_DATA'] = 'exclude'
    parameters['ARCHITECTURE'] = new_ARCHITECTURE_deeper
    parameters['NOTES'] = 'deeper.Arch., 1024x1024'
    # run_experiment(exp_name, parameters, save_results=True)
    
    parameters = copy(default_parameters)
    parameters['DEVICE'] = 'cuda:2'
    parameters['FROM_CACHE'] = OUTPUT_DIR
    parameters['NUM_WORKERS'] = 3
    parameters['EPOCHS'] = 1501
    parameters['LR'] = 1e-4
    parameters['USE_MOTION_DATA'] = 'exclude'
    parameters['ARCHITECTURE'] = new_ARCHITECTURE_deeper
    parameters['NOTES'] = 'deeper.Arch., 1024xDx1024'
    # run_experiment(exp_name, parameters, save_results=True)
    
    parameters = copy(default_parameters)
    parameters['DEVICE'] = 'cuda:3'
    parameters['FROM_CACHE'] = OUTPUT_DIR
    parameters['NUM_WORKERS'] = 3
    parameters['EPOCHS'] = 1501
    parameters['LR'] = 1e-4
    parameters['USE_MOTION_DATA'] = 'exclude'
    parameters['ARCHITECTURE'] = new_ARCHITECTURE_deeper
    parameters['NOTES'] = 'deeper.Arch., 1024xDx1024xD'
    # run_experiment(exp_name, parameters, save_results=True)









    parameters = copy(default_parameters)
    parameters['DEVICE'] = 'cuda:1'
    parameters['FROM_CACHE'] = OUTPUT_DIR
    parameters['NUM_WORKERS'] = 3
    parameters['EPOCHS'] = 751
    parameters['LR'] = 1e-4
    parameters['ARCHITECTURE'] = 'mobilenet'
    parameters['USE_MOTION_DATA'] = 'exclude'
    parameters['NOTES'] = 'mobilenet, pretrained'
    # run_experiment(exp_name, parameters, save_results=True)

    parameters = copy(default_parameters)
    parameters['DEVICE'] = 'cuda:2'
    parameters['FROM_CACHE'] = OUTPUT_DIR
    parameters['NUM_WORKERS'] = 3
    parameters['EPOCHS'] = 751
    parameters['LR'] = 1e-4
    parameters['USE_MOTION_DATA'] = 'exclude'
    parameters['ARCHITECTURE'] = 'alexnet'
    parameters['NOTES'] = 'alexnet, pretrained'
    # run_experiment(exp_name, parameters, save_results=True)

    parameters = copy(default_parameters)
    parameters['DEVICE'] = 'cuda:3'
    parameters['FROM_CACHE'] = OUTPUT_DIR
    parameters['NUM_WORKERS'] = 3
    parameters['EPOCHS'] = 751
    parameters['LR'] = 1e-4
    parameters['USE_MOTION_DATA'] = 'include'
    parameters['ARCHITECTURE'] = new_ARCHITECTURE_moreCs
    parameters['NOTES'] = 'base, more Cs, incl motion'
    # run_experiment(exp_name, parameters, save_results=True)










    # parameters = copy(default_parameters)
    # parameters['DEVICE'] = 'cuda:1'
    # parameters['FROM_CACHE'] = OUTPUT_DIR
    # parameters['NUM_WORKERS'] = 1
    # parameters['EPOCHS'] = 1001
    # parameters['LR'] = 5e-5
    # parameters['ARCHITECTURE'] = old_ARCHITECTURE
    # parameters['USE_MOTION_DATA'] = 'include'
    # parameters['LOAD_MODEL'] = [exp_name, 'run10', '1000']   
    # parameters['NOTES'] = 'old arch, incl mot, contd.10, LR 5e-5, ' #run25
    # # run_experiment(exp_name, parameters, save_results=True)

    # parameters = copy(default_parameters)
    # parameters['DEVICE'] = 'cuda:2'
    # parameters['FROM_CACHE'] = OUTPUT_DIR
    # parameters['NUM_WORKERS'] = 3
    # parameters['EPOCHS'] = 2001
    # parameters['LR'] = 1e-4
    # parameters['ARCHITECTURE'] = new_ARCHITECTURE_nogrouping
    # parameters['USE_MOTION_DATA'] = 'exclude'

    # parameters['NOTES'] = 'new arch, no motion, no grouping' #run26
    # # run_experiment(exp_name, parameters, save_results=True)
    
    # parameters = copy(default_parameters)
    # parameters['DEVICE'] = 'cuda:3'
    # parameters['FROM_CACHE'] = OUTPUT_DIR
    # parameters['NUM_WORKERS'] = 3
    # parameters['EPOCHS'] = 2001
    # parameters['ARCHITECTURE'] = new_ARCHITECTURE_nofeaturecat
    # parameters['LR'] = 1e-4
    # parameters['USE_MOTION_DATA'] = 'exclude'
    # parameters['NOTES'] = 'new arch, no motion, no featureCat'  #run27
    # # run_experiment(exp_name, parameters, save_results=True)