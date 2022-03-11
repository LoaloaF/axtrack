import os
import sys
# make runable from code basedir and v1model base dir
sys.path.extend(('./../../', '../../screen', os.path.abspath(os.curdir))) 
sys.path.extend(('./machinelearning/v1model/', './screen', os.path.abspath(os.curdir)))

import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import config
from config import OUTPUT_DIR, SPACER, TRAINING_DATA_DIR
from video_plotting import draw_all
from AxonDetections import AxonDetections

from exp_parameters import (
    get_default_parameters, 
    write_parameters, 
    check_parameters,
    load_parameters, 
    params2text, 
    compare_parameters,
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
    prepend_prev_run,
    turn_tex,
    )
from exp_evaluation import (
    setup_evaluation,
    evaluate_preprocssing,
    evaluate_training,
    evaluate_precision_recall,
    evaluate_model,
)
       
def run_experiment(exp_name, parameters, save_results=True):
    set_seed(parameters['SEED'])
    if torch.cuda.is_available():
        torch.cuda.set_device(parameters['DEVICE'])

    # Setup saving and parameter checking 
    print(f'Running Experiment: {exp_name}', flush=True)
    check_parameters(parameters, get_default_parameters())
    if save_results:
        dirs, run_label = create_logging_dirs(exp_name)
        RUN_DIR, MODELS_DIR, METRICS_DIR = dirs
        write_parameters(f'{RUN_DIR}/params.pkl', parameters)
        print('\tSaving: ', run_label)
    else:
        RUN_DIR, MODELS_DIR, METRICS_DIR = None, None, None
        print('\tRun is not saved!')
    print(params2text(parameters), flush=True)

    # setup model and data
    train_data, test_data = setup_data(parameters)
    model, loss_fn, optimizer, lr_scheduler = setup_model(parameters)
    # iterate epochs
    optimize(parameters, train_data, test_data, model, loss_fn, optimizer, 
             lr_scheduler, save_results, MODELS_DIR, METRICS_DIR, RUN_DIR)


def optimize(parameters, train_data, test_data, model, loss_fn, optimizer, 
             lr_scheduler, save_results, MODELS_DIR, METRICS_DIR, RUN_DIR):
    print_log = []
    for epoch in range(parameters['EPOCHS']):
        last_epoch_dt = round(time.time()-tstart) if epoch else ''
        tstart = time.time()
        print(f'\n\n\nEpoch {epoch}/{parameters["EPOCHS"]}, last epoch took: {last_epoch_dt}s\n{config.SPACER}', flush=True)
        # train and optimize
        epoch_train_info = one_epoch(train_data, model, loss_fn, parameters, 
                                     epoch, optimizer, lr_scheduler)
        # test data
        model.eval()
        epoch_test_info = one_epoch(test_data, model, loss_fn, parameters, epoch)
        model.train()
        
        # what to save 
        epoch_info = pd.concat([epoch_train_info, epoch_test_info], axis=1).stack(1).T
        epoch_info.index.name = 'epoch'

        # only loss , only current epoch
        if epoch_info.shape[1] == 10:
            epoch_info_str = epoch_info.loc[epoch, ['total_summed_loss', 
                                            'total_object_loss', 'total_pos_labels_rate']] 
            epoch_info_str = epoch_info_str.to_frame()
        # loss and metrics in epoch , print all training epochs so far
        else:
            metr_indexer = [(m, parameters['BBOX_THRESHOLD']) for m in ('precision', 'recall', 'F1')]
            epoch_info_str = epoch_info.loc[epoch, ['total_summed_loss', *metr_indexer]]
            print_log.append(epoch_info_str.to_frame().T)
            epoch_info_str = pd.concat(print_log)
        print(config.SPACER, '\n', epoch_info_str)
        
        if save_results:
            save_epoch_results(epoch_info, epoch, parameters, train_data, 
                               test_data, model, optimizer, lr_scheduler, 
                               MODELS_DIR, METRICS_DIR, RUN_DIR)

def save_epoch_results(epoch_info, epoch, parameters, train_data, test_data, model, optimizer, 
             lr_scheduler, MODELS_DIR, METRICS_DIR, RUN_DIR):
    # save the current epoch loss
    epoch_info.to_pickle(f'{METRICS_DIR}/E{epoch:0>4}.pkl')
    
    # save preprocessing of input data for plotting 
    if epoch == 0 and parameters['PLOT_PREPROC']:
        save_preproc_metrics(f'{RUN_DIR}/preproc_data/', train_data, test_data)
    
    # checkpoint
    if epoch in parameters['MODEL_CHECKPOINTS']:
        save_checkpoint(model, optimizer, lr_scheduler, 
                        filename=f'{MODELS_DIR}/E{epoch:0>4}.pth')
    
        epoch_dir = f'{METRICS_DIR}/{epoch:0>4}_results/'
        os.makedirs(epoch_dir)

        # predict everything in train data and plot result
        train_axon_dets = AxonDetections(model, train_data, parameters, None)
        train_fname = f'train_E{epoch}_timepoint---of{train_data.sizet}'
        draw_all(train_axon_dets, train_fname, dest_dir=epoch_dir, 
                    notes=parameters["NOTES"], **parameters['PERF_LOG_VIDEO_KWARGS'])
        
        # predict everything in test data and plot result
        test_axon_dets = AxonDetections(model, test_data, parameters, None)
        test_fname = f'test_E{epoch}_timepoint---of{test_data.sizet}'
        draw_all(test_axon_dets, test_fname, dest_dir=epoch_dir, 
                    notes=parameters["NOTES"], animated=False, 
                    draw_grid=True)
 
def optimize_MCF_params(exp_name, run, epoch='latest', MCF_param_vals={}):
    RUN_DIR, params = setup_evaluation(exp_name, run)
    params = to_device_specifc_params(params, get_default_parameters(), from_cache=OUTPUT_DIR)
    params['LOAD_MODEL'] = [exp_name, run, epoch]

    _, test_data = setup_data(params)
    model, _, _, _ = setup_model(params)

    axon_detections = AxonDetections(model, test_data, params, f'{RUN_DIR}/axon_dets')
    axon_detections.MCF_min_ID_lifetime = 1
    axon_detections.detect_dataset('from')
    axon_detections.assign_ids('from', 'to')
    axon_detections.search_MCF_params(**MCF_param_vals)
    


if __name__ == '__main__':
    default_parameters = get_default_parameters()
    # exp2_name = 'v1Model_exp2_boxlossfocus'
    # exp3_name = 'v1Model_exp3_newcnn'
    # exp4_name = 'v1Model_exp4_NewCNN'
    # exp5_name = 'v1Model_exp5_RefactIDFocus'
    # exp6_name = 'v1Model_exp6_AxonDetClass'
    exp7_name = 'v1Model_exp7_final'
    exp8_name = 'v1Model_exp8_postsubmission'
    
    # some general settings
    parameters = get_default_parameters()
    parameters['NOTES'] = 'GetThingsToRun'
    parameters['FROM_CACHE'] = None
    parameters['CACHE'] = False
    parameters['PERF_LOG_VIDEO_KWARGS'] = {'animated':True, 'draw_grid':True, 
                                           't_y_x_slice':[(0,40), None, None]}


    parameters['USE_TRANSFORMS'] = []
    parameters['LOAD_MODEL'] = (exp7_name, 'run35', 600)
    parameters['MODEL_CHECKPOINTS'] = [100,200]
    # on gpuserver settings
    # parameters['TEST_TIMEPOINTS'] = config.WHOLE_DATASET_TEST_FRAMES
    # parameters['TRAIN_TIMEPOINTS'] = config.WHOLE_DATASET_TRAIN_FRAMES

    old_arch = [
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
         'M',],
        [(3, 160, 1,  1)],      # y-x out: 16
        #  ],      
        [('FC', 1024),
         ('activation', nn.Sigmoid()), 
         ('FC', 1024),
         ('activation', nn.Sigmoid()), 
        ]
    ]
    parameters['ARCHITECTURE'] = old_arch

    # run training with a specifc parsameter set
    # run_experiment(exp8_name, parameters, save_results=False)

    turn_tex('on')
    # clean_rundirs(exp8_name, delete_runs_min_epochs=1, keep_only_latest_model=False)
    # prepend_prev_run(exp8_name, 'run01', 'run00')
    # evaluate_preprocssing(exp8_name, 'run00', show=True)
    # evaluate_training([[exp8_name, 'run00'], [exp8_name, 'run01']], show=True, use_prepend_ifavail=True)
    # evaluate_precision_recall([[exp8_name, 'run00', 70], [exp8_name, 'run01', 5]], show=True, recreate=True)
    # evaluate_model(exp8_name, 'run05', which_data='test', cache_detections='to',
    #                video_kwargs={'draw_grid':True, 'show':False, 'animated':True})
    # evaluate_model(exp7_name, 'run35', which_data='test', cache_detections='from',
    #                video_kwargs={'draw_grid':True, 'show':False, 'animated':True})
    evaluate_model(exp7_name, 'run35', cache_detections='to', astar_paths_cache='from', assigedIDs_cache='to')
    
    # hyperparams to search over
    
    
    MCF_param_vals = {'edge_cost_thr_values':  [.4, .6, .7, .8, .9, 1, 1.2, 3],
              'entry_exit_cost_values': [ .2, .8, .9, 1, 1.1,  2],
              'miss_rate_values':  [0.9, 0.6],
              'vis_sim_weight_values': [0, 0.1],
              'conf_capping_method_values': ['ceil' , 'scale_to_max'],}
    optimize_MCF_params(exp7_name, 'run35', MCF_param_vals=MCF_param_vals)


    # evaluate_training([[exp8_name, 'run41']], recreate=True, show=True)
    # p2 = load_parameters(exp7_name, 'run37')
    # o = compare_parameters(get_default_parameters(), p2)
    # print(o)

    # larger_arch = [
    #     #kernelsize, out_channels, stride, groups
    #     [(3, 20,  2,  1),      # y-x out: 256
    #      (3, 80,  1,  1),      # y-x out: 64
    #      'M',
    #      (3, 80,  1,  1),      # y-x out: 64
    #      (3, 80,  1,  1),      # y-x out: 64
    #      (3, 80,  1,  1),      # y-x out: 64
    #      'M',
    #      (3, 100,  1,  1),      # y-x out: 32
    #      (3, 40,  2,  1),      # y-x out: 128
    #      (3, 100,  1,  1),      # y-x out: 32
    #      (3, 100,  1,  1),      # y-x out: 32
    #      'M',
    #      ],      
    #     [(3, 180, 1,  1),      # y-x out: 16
    #      ],
    #     [('FC', 1024),
    #      ('activation', nn.Sigmoid()), 
    #      ('FC', 1024),
    #      ('activation', nn.Sigmoid()), 
    #     ]
    # ]