import os
import sys
where = os.path.abspath(os.curdir)
sys.path.append('../../')
sys.path.append(where)
from copy import copy
# from os import environ as cuda_environment

from  config import OUTPUT_DIR, SPACER, RAW_TRAINING_DATA_DIR
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
        # params2img(f'{RUN_DIR}/params.png', parameters, show=False)
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
            if epoch in (600, 1000, 1350, 1600):
            # if epoch in (400, 750, 1000):
            # if epoch and not (epoch % 800):
                save_checkpoint(model, optimizer, lr_scheduler, filename=f'{MODELS_DIR}/E{epoch:0>4}.pth')
            
            # check the model predictions every 20 Epochs
            # if epoch%500 == 0:
            # if epoch in (350, 600):
                epoch_dir = f'{METRICS_DIR}/{epoch:0>4}_results/'
                os.makedirs(epoch_dir)

                # predict everything in train data and plot result
                train_axon_dets = AxonDetections(model, train_data, parameters, None)
                train_fname = f'train_E{epoch}_timepoint---of{train_data.sizet}'
                draw_all(train_axon_dets, train_fname, dest_dir=epoch_dir, notes=parameters["NOTES"], animated=False)
                
                # predict everything in test data and plot result
                test_axon_dets = AxonDetections(model, test_data, parameters, None)
                test_fname = f'test_E{epoch}_timepoint---of{test_data.sizet}'
                draw_all(test_axon_dets, test_fname, dest_dir=epoch_dir, notes=parameters["NOTES"], animated=False)

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
    exp7_name = 'v1Model_exp7_final'
    
    # from utils import print_models
    # print_models()
    # exit()

    # clean_rundirs(exp7_name, delete_runs=3, keep_only_latest_model=False)
    # clean_rundirs(exp7_name, keep_runs=[2,35,36,38,41], keep_only_latest_model=False)
    # clean_rundirs(exp6_name, delete_runs=100, keep_only_latest_model=False)
    
    # evaluate_preprocssing(exp7_name, 'run34', show=False)
    # evaluate_precision_recall([(exp7_name, 'run02', 50), (exp7_name, 'run02', 600), (exp7_name, 'run02', 700), (exp7_name, 'run02', 800), (exp7_name, 'run02', 300)], avg_over_t=0)
    # evaluate_training([(exp7_name, 'run02')], recreate=True)
    # evaluate_training([(exp7_name, 'run24')], recreate=True)
    # prepend_prev_run(exp7_name, 'run08', 'run09', 500, 250)
    # evaluate_training([], recreate=False)
    # evaluate_training([(exp7_name, 'run24'), (exp7_name, 'run22'),]) #(exp7_name, 'run06')], recreate=False)
    # evaluate_training([(exp7_name, 'run08'), (exp7_name, 'run06')], recreate=False)


    # turn_tex('on')


    # evaluate_training([(exp7_name, 'run41'), ], recreate=False, show=True)
    # evaluate_training([(exp7_name, 'run38'), ], recreate=False, show=True)
    # evaluate_precision_recall([(exp7_name, 'run38', 1000), (exp7_name, 'run41', 900)], avg_over_t=42)
    evaluate_model(exp7_name, 'run38', 600, which_data='train', animated=False)
    # evaluate_ID_assignment(exp7_name, 'run38', 1000, cached_astar_paths='from', which_data='test')




    # prepend_prev_run(exp7_name, 'run35', 'run36', 600, 935)
    # evaluate_training([(exp7_name, 'run38')], recreate=False)
    # evaluate_training([(exp7_name, 'run41')], recreate=False)
    # evaluate_training([(exp7_name, 'run38'), (exp7_name, 'run41')], recreate=False)
    # evaluate_training([(exp6_name, 'run23'), 
    #                    (exp6_name, 'run17'), 
    #                    (exp6_name, 'run28'), 
    #                    (exp6_name, 'run29'), 
    #                    (exp6_name, 'run35'), 
    #                    (exp6_name, 'run36'), 
    #                    (exp6_name, 'run37'), 
    #                    (exp6_name, 'run39'), 
    #                    ], recreate=False)
    # evaluate_preprocssing(exp6_name, 'run35', show=True)
    # evaluate_precision_recall([(exp6_name, 'run17', 2000),
    #                            (exp6_name, 'run17', 250
    # 0),
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
    # evaluate_model(exp6_name, 'run39', 3000, animated=False, save_single_tiles=True)
    # evaluate_model(exp5_name, 'run34', 3000, animated=True)
    # evaluate_model(exp5_name, 'run34', 3000, assign_ids=True)
    # evaluate_model(exp6_name, 'run12', 3000,  animated=True, show=False)
    
    # evaluate_model(exp6_name, 'run35', 3000,  animated=True, show=True, filter2FP_FN=False)
    # evaluate_model(exp6_name, 'run39', 3000,  animated=True, show=True, filter2FP_FN=False)

    # evaluate_ID_assignment(exp6_name, 'run39', 3000, cached_astar_paths='from', 
    #                        do_draw_all_vis=False, animated=False, show=True, # which_axons=['Axon_016'],
    #                        hide_det2=False, draw_axons=True, color_det2_ids=False, 
    #                        color_det1_ids=True, )#t_y_x_slice=[None, [1000,2000], [2000,4000]])
    
    # evaluate_ID_assignment(exp6_name, 'run20', 500,  animated=False, show=False, 
    #                        cached_astar_paths='from')
    # evaluate_ID_assignment(exp6_name, 'run12', 3000,  animated=False, show=True, 
    #                        do_ID_lifetime_vis=False, cached_astar_paths='from')
    
    # parameters['CACHE'] = OUTPUT_DIR
    # parameters['FROM_CACHE'] = None
    # parameters['DEVICE'] = 'cuda:0'
    # parameters = load_parameters(exp6_name, 'run39')
    parameters = get_default_parameters()
    # parameters = to_device_specifc_params(parameters, get_default_parameters())
    # parameters['USE_TRANSFORMS'] = []
    parameters['CACHE'] = None
    parameters['FROM_CACHE'] = OUTPUT_DIR
    parameters['TIMELAPSE_FILE'] = RAW_TRAINING_DATA_DIR + 'training_timelapse.tif'
    parameters['MASK_FILE'] = RAW_TRAINING_DATA_DIR + 'training_mask.npy'
    parameters['LABELS_FILE'] = RAW_TRAINING_DATA_DIR + 'axon_anchor_labels.csv'
    parameters['DEVICE'] = 'cuda:1'

    parameters['TEST_TIMEPOINTS'] = list(range(37+80-20, 37+80+20))
    parameters['TRAIN_TIMEPOINTS'] = list(range(2, 37+80-20-4)) + list(range(37+80+20+4, 37+80+210-2))
    
    parameters['TEST_TIME_DISCONTINUITIES'] = [37+80]
    parameters['TRAIN_TIME_DISCONTINUITIES'] = [37, 37+80-20-4, 37+80+20+4]
    parameters['NUM_WORKERS'] = 2

    parameters['NOTES'] = 'fixing precision recall'
    # run_experiment(exp7_name, parameters, save_results=True)


    


    larger_arch = [
        #kernelsize, out_channels, stride, groups
        [(3, 20,  2,  1),      # y-x out: 256
         (3, 80,  1,  1),      # y-x out: 64
         'M',
         (3, 80,  1,  1),      # y-x out: 64
         (3, 80,  1,  1),      # y-x out: 64
         (3, 80,  1,  1),      # y-x out: 64
         'M',
         (3, 100,  1,  1),      # y-x out: 32
         (3, 40,  2,  1),      # y-x out: 128
         (3, 100,  1,  1),      # y-x out: 32
         (3, 100,  1,  1),      # y-x out: 32
         'M',
         ],      
        [(3, 180, 1,  1),      # y-x out: 16
         ],
        [('FC', 1024),
         ('activation', nn.Sigmoid()), 
         ('FC', 1024),
         ('activation', nn.Sigmoid()), 
        ]
    ]
    do_arch = [
        #kernelsize, out_channels, stride, groups
        [(3, 20,  2,  1),      # y-x out: 256
         (3, 40,  2,  1),      # y-x out: 128
         (3, 80,  1,  1),      # y-x out: 64
         'M',
         (3, 80,  1,  1),      # y-x out: 64
         (3, 80,  1,  1),      # y-x out: 64
         'M',
         (3, 80,  1,  1),      # y-x out: 32
         (3, 80,  1,  1),      # y-x out: 32
         'M',
         ],      
        [(3, 160, 1,  1),      # y-x out: 16
         ],
        [('FC', 1024),
         ('activation', nn.Sigmoid()), 
         ('dropout', .3),
         ('FC', 1024),
         ('activation', nn.Sigmoid()), 
        ]
    ]
    parameters = load_parameters(exp7_name, 'run24')
    parameters['NUM_WORKERS'] = 2
    parameters['BATCH_SIZE'] = 26
    parameters['DEVICE'] = 'cuda:2'
    parameters['OFFSET'] = None
    parameters['CACHE'] = None
    parameters['FROM_CACHE'] = OUTPUT_DIR
    parameters['EPOCHS'] = 1601
    parameters['TEST_TIMEPOINTS'] = list(range(37+80-20, 37+80+20))
    parameters['TRAIN_TIMEPOINTS'] = list(range(2, 37+80-20-4)) + list(range(37+80+20+4, 37+80+210-2))
    parameters['ARCHITECTURE'] = larger_arch
    parameters['NOTES'] = 'Larger CNN architecture'
    # run_experiment(exp7_name, parameters, save_results=True)


    parameters = load_parameters(exp7_name, 'run24')
    parameters['NUM_WORKERS'] = 1
    parameters['DEVICE'] = 'cuda:3'
    parameters['OFFSET'] = None
    parameters['CACHE'] = None
    parameters['FROM_CACHE'] = OUTPUT_DIR
    parameters['EPOCHS'] = 1601
    parameters['TEST_TIMEPOINTS'] = list(range(37+80-20, 37+80+20))
    parameters['TRAIN_TIMEPOINTS'] = list(range(2, 37+80-20-4)) + list(range(37+80+20+4, 37+80+210-2))
    parameters['NOTES'] = 'run24 params, final'
    # run_experiment(exp7_name, parameters, save_results=True)

    # final final final
    parameters = load_parameters(exp7_name, 'run24')
    parameters['LOAD_MODEL'] = [exp7_name, 'run35', 600]
    parameters['DEVICE'] = 'cuda:2'
    parameters['OFFSET'] = None
    parameters['NUM_WORKERS'] = 1
    parameters['EPOCHS'] = 1001
    parameters['CACHE'] = None
    parameters['FROM_CACHE'] = OUTPUT_DIR
    parameters['TEST_TIMEPOINTS'] = (2,3,4,5,6)
    parameters['TRAIN_TIMEPOINTS'] = list(range(2, 37+80+210-2))
    # parameters['L_OBJECT'] = 54.25
    # parameters['L_NOBJECT'] = 0.75
    # parameters['L_COORD_ANCHOR'] = 45
    # parameters['LR_DECAYRATE'] = 10
    parameters['NOTES'] = 'run24 params, final, train on all data, cntn E600 run35'
    # run_experiment(exp7_name, parameters, save_results=True)


    parameters = load_parameters(exp7_name, 'run02')
    parameters['LOAD_MODEL'] = [exp7_name, 'run09', 250]
    parameters['DEVICE'] = 'cuda:0'
    parameters['LR_DECAYRATE'] = 10
    parameters['FROM_CACHE'] = OUTPUT_DIR
    parameters['NUM_WORKERS'] = 1
    parameters['NOTES'] = 'LR_decay=10'


    parameters = load_parameters(exp7_name, 'run02')
    parameters['DEVICE'] = 'cuda:1'
    parameters['NUM_WORKERS'] = 3
    parameters['FROM_CACHE'] = OUTPUT_DIR
    parameters['L_OBJECT'] = 54.25
    parameters['L_NOBJECT'] = 0.75
    parameters['L_COORD_ANCHOR'] = 45
    parameters['LR_DECAYRATE'] = 10
    parameters['NOTES'] = 'LR_decay=10, Object Loss increased'
    # run_experiment(exp7_name, parameters, save_results=True)

    parameters = load_parameters(exp7_name, 'run02')
    parameters['DEVICE'] = 'cuda:2'
    parameters['ARCHITECTURE'] = larger_arch
    parameters['NOTES'] = 'Larger CNN architecture'
    # run_experiment(exp7_name, parameters, save_results=True)
    

    parameters = load_parameters(exp7_name, 'run02')
    parameters['DEVICE'] = 'cuda:2'
    parameters['LR_DECAYRATE'] = None
    parameters['LR'] = 1e-4
    parameters['NOTES'] = 'Constant LR=1e-4, no decay'
    # run_experiment(exp7_name, parameters, save_results=True)
    

    parameters = load_parameters(exp7_name, 'run02')
    parameters['DEVICE'] = 'cuda:3'
    parameters['ARCHITECTURE'] = do_arch
    parameters['FROM_CACHE'] = OUTPUT_DIR
    parameters['NOTES'] = 'Dropout CNN architecture'
    # run_experiment(exp7_name, parameters, save_results=True)
    

    parameters = load_parameters(exp7_name, 'run02')
    parameters['DEVICE'] = 'cuda:4'
    parameters['TEMPORAL_CONTEXT'] = 0
    parameters['FROM_CACHE'] = OUTPUT_DIR
    parameters['NOTES'] = 'No temporal context'
    # run_experiment(exp7_name, parameters, save_results=True)
    

    parameters = load_parameters(exp7_name, 'run02')
    parameters['DEVICE'] = 'cuda:5'
    parameters['USE_MOTION_DATA'] = 'include'
    parameters['TEMPORAL_CONTEXT'] = 1
    parameters['FROM_CACHE'] = OUTPUT_DIR
    parameters['NOTES'] = 'Precomputed motion, temporal context=1'
    # run_experiment(exp7_name, parameters, save_results=True)
    