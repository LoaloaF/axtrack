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
    )
from core_functionality import (
    setup_data, 
    setup_model, 
    one_epoch,
    setup_data_loaders,
    )
from utils import (
    create_logging_dirs,
    save_preproc_metrics,
    save_checkpoint,
    clean_rundirs,
    set_seed,
    )
from evaluation import (
    evaluate_preprocssing,
    evaluate_training,
    evaluate_precision_recall,
)
from config import OUTPUT_DIR, SPACER

def run_experiment(exp_name, parameters, save_results=True):
    set_seed(parameters['SEED'])

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
        epoch_loss, epoch_metrics = one_epoch(train_data, model, loss_fn, parameters, epoch, 
                                              optimizer, lr_scheduler)
        # test data
        model.eval()
        epoch_test_loss, epoch_test_metrics = one_epoch(test_data, model, loss_fn, 
                                                        parameters, epoch)
        model.train()

        # keep track of the loss development
        epoch_log = pd.concat([epoch_loss, epoch_test_loss], axis=1)
        epoch_log_mean = epoch_log.groupby(axis=1, level=1).mean().unstack().rename(epoch)
        metrics_log = pd.concat([epoch_metrics, epoch_test_metrics], axis=1)
        # metrics_log_maxF1 = metrics_log.loc['F1']
        print_log.append(epoch_log_mean)
        
        if save_results:
            # save the current epoch loss
            epoch_log.to_pickle(f'{METRICS_DIR}/E{epoch:0>4}.pkl')
            metrics_log.to_pickle(f'{METRICS_DIR}/E{epoch:0>4}_metrics.pkl')
            # save the model every 100 Epochs
            if epoch and not (epoch % 500):
                save_checkpoint(model, optimizer, filename=f'{MODELS_DIR}/E{epoch:0>4}.pth')
            
            # check the model predictions every 20 Epochs
            if epoch and epoch%500 == 0:
                epoch_dir = f'{METRICS_DIR}/{epoch:0>4}_results/'
                os.makedirs(epoch_dir)
                model.predict_all(train_data, parameters, dest_dir=epoch_dir)
                model.predict_all(test_data, parameters, dest_dir=epoch_dir)
            # save preprocessing of input data for plotting 
            if epoch == 0 and parameters['PLOT_PREPROC']:
                save_preproc_metrics(RUN_DIR, train_data, test_data)

        # print out current progress
        current_progress = pd.concat(print_log, axis=1).unstack(level=0)
        current_progress.index = [lbl[6:lbl.rfind('_loss')] for lbl in current_progress.index]
        print(SPACER)
        print(parameters['NOTES'], parameters['DEVICE'])
        print(current_progress.stack(level=1).T.round(3).to_string())
        print(metrics_log.loc['F1'])
        print(metrics_log.loc['F1'].max(0))



if __name__ == '__main__':
    default_parameters = get_default_parameters()
    exp2_name = 'v1Model_exp2_boxlossfocus'
    exp3_name = 'v1Model_exp3_newcnn'
    exp4_name = 'v1Model_exp4_NewCNN'
    exp5_name = 'v1Model_exp5_RefactIDFocus'
    
    # from utils import print_models
    # print_models()
    # exit()
    # clean_rundirs(exp4_name, keep_runs=[59,99,100], keep_only_latest_model=False)
    # clean_rundirs(exp5_name, delete_runs=5, keep_only_latest_model=True)
    # compare_precision_recall([(exp5_name, 'run18', 1600),(exp5_name, 'run20', 1600), (exp4_name, 'run59', 3000)])
    # evaluate_preprocssing(exp5_name, 'run17')
    # evaluate_precision_recall([(exp5_name, 'run18', 1500), (exp5_name, 'run18', 2000), (exp5_name, 'run18', 2500), (exp5_name, 'run18', 2985)])
    # evaluate_training([(exp5_name, 'run17'),])
    # evaluate_training([(exp5_name, 'run17'), (exp5_name, 'run18'), (exp5_name, 'run19'), (exp5_name, 'run20')], recreate=False)
    # evaluate_training([(exp5_name, 'run17')])


    parameters = copy(default_parameters)
    # parameters['CACHE'] = OUTPUT_DIR
    # parameters['FROM_CACHE'] = None
    parameters['USE_TRANSFORMS'] = []
    parameters['NOTES'] = 'implmenting FP1 metrics'
    # run_experiment(exp5_name, parameters, save_results=False)

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
    parameters['EPOCHS'] = 1501
    parameters['LOAD_MODEL'] = [exp5_name, 'run17', 3000]
    parameters['NOTES'] = 'T2_load17-E3000_baseline'
    # run_experiment(exp5_name, parameters, save_results=True)
    
    parameters = copy(default_parameters)
    parameters['DEVICE'] = 'cuda:2'
    parameters['EPOCHS'] = 1501
    parameters['ARCHITECTURE'] = maxp_arch
    parameters['LOAD_MODEL'] = [exp5_name, 'run18', 3000]
    parameters['NOTES'] = 'T2_load18-E3000_maxpool'
    # run_experiment(exp5_name, parameters, save_results=True)
    
    parameters = copy(default_parameters)
    parameters['DEVICE'] = 'cuda:3'
    parameters['EPOCHS'] = 1501
    parameters['ARCHITECTURE'] = dropout25_arch
    parameters['LOAD_MODEL'] = [exp5_name, 'run19', 3000]
    parameters['NOTES'] = 'T2_load19-E3000_dropout0.25'
    # run_experiment(exp5_name, parameters, save_results=True)
    
    parameters = copy(default_parameters)
    parameters['DEVICE'] = 'cuda:4'
    parameters['EPOCHS'] = 1501
    parameters['ARCHITECTURE'] = dropout35_arch
    parameters['LOAD_MODEL'] = [exp5_name, 'run20', 3000]
    parameters['NOTES'] = 'T2_load20-E3000_dropout0.35'
    run_experiment(exp5_name, parameters, save_results=True)