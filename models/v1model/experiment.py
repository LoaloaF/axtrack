import os
import sys
where = os.path.abspath(os.curdir)
sys.path.append('../../')
sys.path.append(where)
from copy import copy
# from os import environ as cuda_environment
print(where)

from  config import OUTPUT_DIR, SPACER
from plotting import draw_all

from AxonDetections import AxonDetections

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
    evaluate_model,
    evaluate_ID_assignment,
)


def run_experiment(exp_name, parameters, save_results=True):
    set_seed(parameters['SEED'])
    if torch.cuda.is_available():
        # torch.multiprocessing.set_start_method('spawn')
        torch.cuda.set_device(parameters['DEVICE'])

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
        # for printing average loss 
        epoch_log_mean = epoch_log.iloc[-3:-1].groupby(axis=1,level=1).mean().unstack().rename(epoch)
        
        metrics_log = pd.concat([epoch_metrics, epoch_test_metrics], axis=1)
        if not metrics_log.empty:
            maxF1_idx = metrics_log.loc['F1'].groupby(level=1).apply(lambda s: s.index[s.argmax()])
            metrics_log_maxF1 = metrics_log.loc[:, maxF1_idx].droplevel((0,2),1).unstack().rename(epoch)
            print_log.append(pd.concat([epoch_log_mean, metrics_log_maxF1]))

        if save_results:
            # save the current epoch loss
            epoch_log.to_pickle(f'{METRICS_DIR}/E{epoch:0>4}.pkl')
            metrics_log.to_pickle(f'{METRICS_DIR}/E{epoch:0>4}_metrics.pkl')
            # save the model every 100 Epochs
            if epoch and not (epoch % 500):
                save_checkpoint(model, optimizer, lr_scheduler, filename=f'{MODELS_DIR}/E{epoch:0>4}.pth')
            
            # check the model predictions every 20 Epochs
            # if epoch%500 == 0:
            if epoch and epoch%1000 == 0:
                epoch_dir = f'{METRICS_DIR}/{epoch:0>4}_results/'
                os.makedirs(epoch_dir)

                # predict everything in train data and plot result
                train_axon_dets = AxonDetections(model, train_data, parameters, None)
                train_fname = f'train_E:{epoch}_timepoint:---|{train_data.sizet}'
                draw_all(train_axon_dets, train_fname, dest_dir=epoch_dir, notes=parameters["NOTES"])
                
                # predict everything in test data and plot result
                test_axon_dets = AxonDetections(model, test_data, parameters, None)
                test_fname = f'test_E:{epoch}_timepoint:---|{test_data.sizet}'
                draw_all(test_axon_dets, test_fname, dest_dir=epoch_dir, notes=parameters["NOTES"])

            # save preprocessing of input data for plotting 
            if epoch == 0 and parameters['PLOT_PREPROC']:
                save_preproc_metrics(f'{RUN_DIR}/preproc_data/', train_data, test_data)

        # print out current progress
        current_progress = pd.concat(print_log, axis=1).unstack(level=0)
        print(SPACER)
        print(parameters['NOTES'], parameters['DEVICE'])
        print(current_progress.stack(level=1).T.round(3).tail(400).to_string())
        print(SPACER)
        print('Running mean (last 20 epochs):')
        print(current_progress.stack(level=1).iloc[:, -20:].mean(1).round(3).to_frame().T.to_string())


if __name__ == '__main__':
    default_parameters = get_default_parameters()
    exp2_name = 'v1Model_exp2_boxlossfocus'
    exp3_name = 'v1Model_exp3_newcnn'
    exp4_name = 'v1Model_exp4_NewCNN'
    exp5_name = 'v1Model_exp5_RefactIDFocus'
    exp6_name = 'v1Model_exp6_AxonDetClass'
    
    # from utils import print_models
    # print_models()
    # exit()

    # os.chdir('code')

    # clean_rundirs(exp6_name, delete_runs=1, keep_only_latest_model=False)
    # clean_rundirs(exp6_name, keep_runs=[12,17,18,20,22], keep_only_latest_model=False)
    # clean_rundirs(exp6_name, delete_runs=100, keep_only_latest_model=False)
    
    # evaluate_precision_recall([(exp5_name, 'run18', 3000),(exp5_name, 'run28', 0)])
    # evaluate_precision_recall([(exp6_name, 'run17', 3000), (exp6_name, 'run19', 15)], avg_over_t=30)
    # evaluate_training([(exp6_name, 'run20')])
    # evaluate_training([(exp6_name, 'run17'), (exp6_name, 'run18')])
    # evaluate_training([(exp6_name, 'run20'), (exp5_name, 'run17')], recreate=False)
    # evaluate_training([(exp6_name, 'run23'), 
    #                 #    (exp6_name, 'run24'), 
    #                    (exp6_name, 'run28'), 
    #                    (exp6_name, 'run29'), 
    #                    (exp6_name, 'run35'), 
    #                    (exp6_name, 'run36'), 
    #                    (exp6_name, 'run37'), 
                       
    #                    ], recreate=True)
    # evaluate_preprocssing(exp6_name, 'run35', show=True)
    # evaluate_precision_recall([(exp6_name, 'run17', 2000),
    #                            (exp6_name, 'run17', 2500),
    #                            (exp6_name, 'run17', 3000),
    #                            (exp6_name, 'run17', 1500),
    #                            (exp6_name, 'run17', 1000),
    #                            (exp6_name, 'run20', 500),
    #                            (exp6_name, 'run18', 2500),
    #                            (exp6_name, 'run18', 3000),
                            #    (exp6_name, 'run22', 2000),
                            #    (exp6_name, 'run22', 2500),
                            #    (exp6_name, 'run22', 3000),
    # ])
    # evaluate_model(exp5_name, 'run34', 4000, animated=True)
    # evaluate_model(exp5_name, 'run34', 3500, animated=True)   
    # evaluate_model(exp5_name, 'run34', 3000, animated=True)
    # evaluate_model(exp5_name, 'run34', 3000, assign_ids=True)
    # evaluate_model(exp6_name, 'run12', 3000,  animated=True, show=False)
    
    evaluate_model(exp6_name, 'run12', 3000,  animated=False, show=True, filter2FP_FN=False)
    # evaluate_model(exp6_name, 'run20', 500,  animated=False, show=True, filter2FP_FN=False)
    # evaluate_ID_assignment(exp6_name, 'run20', 500,  animated=False, show=False, 
    #                        cached_astar_paths='from')
    # evaluate_ID_assignment(exp6_name, 'run12', 3000,  animated=False, show=True, 
    #                        do_ID_lifetime_vis=False, cached_astar_paths='from')
    
    # parameters['CACHE'] = OUTPUT_DIR
    # parameters['FROM_CACHE'] = Nonemn
    # parameters['DEVICE'] = 'cuda:0'
    # parameters['USE_TRANSFORMS'] = []
    parameters = load_parameters(exp6_name, 'run12')
    parameters = to_device_specifc_params(parameters, get_default_parameters())
    parameters['EPOCHS'] = 1
    # parameters['NON_MAX_SUPRESSION_DIST'] = 23
    parameters['CACHE'] = OUTPUT_DIR
    parameters['FROM_CACHE'] = None
    parameters['NOTES'] = 'trying'
    # run_experiment(exp6_name, parameters, save_results=True)

    maxp_do10_arch = [
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
         ('dropout', 0.20),
         ('FC', 1024),
         ('activation', nn.Sigmoid()), 
         ('dropout', 0.20),
        ]     
    ]
    
    maxp_do10_deeper_arch = [
        #kernelsize, out_channels, stride, groups
        [(3, 20,  2,  1),      # y-x out: 256
         (3, 40,  2,  1),      # y-x out: 128
         (3, 40,  2,  1),      # y-x out: 128
         (3, 80,  1,  1),      # y-x out: 64
         'M',
         (3, 80,  1,  1),      # y-x out: 64
         (3, 80,  1,  1),      # y-x out: 32
         'M',
         (3, 80,  1,  1),      # y-x out: 32
         (3, 80,  1,  1),      # y-x out: 32
         (3, 80,  1,  1),      # y-x out: 32
         'M',
         ],      
        [(3, 160, 1,  1),      # y-x out: 16
         ],
        [('FC', 1024),
         ('activation', nn.Sigmoid()), 
         ('dropout', 0.20),
         ('FC', 1024),
         ('activation', nn.Sigmoid()), 
         ('dropout', 0.20),
        ]     
    ]
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
    maxp_3in_arch = [
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


    parameters = copy(default_parameters)
    parameters['DEVICE'] = 'cuda:0'
    parameters['NON_MAX_SUPRESSION_DIST'] = 23
    parameters['ARCHITECTURE'] = maxp_arch
    parameters['FROM_CACHE'] = OUTPUT_DIR
    parameters['NOTES'] = 'old - from_cache=True'
    # run_experiment(exp6_name, parameters, save_results=True)
    
    parameters = copy(default_parameters)
    parameters['DEVICE'] = 'cuda:1'
    parameters['NON_MAX_SUPRESSION_DIST'] = 23
    parameters['ARCHITECTURE'] = maxp_arch
    parameters['FROM_CACHE'] = OUTPUT_DIR
    parameters['USE_TRANSFORMS'] = ['vflip', 'hflip', 'rot', 'translateY', 'translateX', 'intensity_scaling']
    parameters['NOTES'] = 'old - from_cache=True, augmenting *0.5-1.5 intensity'
    # run_experiment(exp6_name, parameters, save_results=True)
    


    # ========== GPU experiments ===========
    parameters = copy(default_parameters)
    parameters['DEVICE'] = 'cuda:2'
    parameters['NON_MAX_SUPRESSION_DIST'] = 23
    parameters['ARCHITECTURE'] = maxp_arch
    parameters['FROM_CACHE'] = None
    parameters['CLIP_LOWERLIM'] = 55 /2**16
    parameters['STANDARDIZE'] = ('', None)
    parameters['NOTES'] = 'clip55, Non standardized data'
    # run_experiment(exp6_name, parameters, save_results=True)
    
    parameters = copy(default_parameters)
    parameters['DEVICE'] = 'cuda:3'
    parameters['NON_MAX_SUPRESSION_DIST'] = 23
    parameters['ARCHITECTURE'] = maxp_arch
    parameters['FROM_CACHE'] = None
    parameters['CLIP_LOWERLIM'] = 55 /2**16
    parameters['STANDARDIZE_FRAMEWISE'] = True
    parameters['NOTES'] = 'clip55, framewise std=1'
    # run_experiment(exp6_name, parameters, save_results=True)

    
    
    parameters = copy(default_parameters)
    parameters['DEVICE'] = 'cuda:4'
    parameters['NON_MAX_SUPRESSION_DIST'] = 23
    parameters['ARCHITECTURE'] = maxp_arch
    parameters['FROM_CACHE'] = None
    parameters['CLIP_LOWERLIM'] = 55 /2**16
    parameters['STANDARDIZE_FRAMEWISE'] = False
    parameters['NOTES'] = 'clip55, non-framewise,global std=1'
    # run_experiment(exp6_name, parameters, save_results=True)
    
    parameters = copy(default_parameters)
    parameters['DEVICE'] = 'cuda:5'
    parameters['NON_MAX_SUPRESSION_DIST'] = 23
    parameters['ARCHITECTURE'] = maxp_arch
    parameters['FROM_CACHE'] = None
    parameters['CLIP_LOWERLIM'] = 55 /2**16
    parameters['STANDARDIZE_FRAMEWISE'] = False
    parameters['STANDARDIZE'] = ('0to1', None)
    parameters['NOTES'] = 'clip55, non-framewise,global 0to1 (max) scaling'
    # run_experiment(exp6_name, parameters, save_results=True)
    