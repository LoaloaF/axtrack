import os
import sys
sys.path.append((os.path.dirname(__file__) + '/../'))
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import axtrack.config as config
from axtrack.config import OUTPUT_DIR, SPACER, TRAINING_DATA_DIR
from axtrack.video_plotting import draw_all
from axtrack.AxonDetections import AxonDetections
from axtrack.exp_parameters import (
    get_default_parameters, 
    write_parameters, 
    check_parameters,
    load_parameters, 
    params2text, 
    compare_parameters,
    to_device_specifc_params,
    update_MCF_params,
    )
from axtrack.machinelearning.core_functionality import (
    setup_data, 
    setup_model, 
    one_epoch,
    setup_data_loaders,
    )
from axtrack.utils import (
    create_logging_dirs,
    save_preproc_metrics,
    save_checkpoint,
    clean_rundirs,
    set_seed,
    prepend_prev_run,
    turn_tex,
    )
from axtrack.exp_evaluation import (
    setup_evaluation,
    evaluate_preprocssing,
    evaluate_training,
    evaluate_precision_recall,
    evaluate_model,
    evaulate_ID_assignment
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
        print(f'\n\n\nEpoch {epoch}/{parameters["EPOCHS"]}, last epoch took: '
              f'{last_epoch_dt}s\n{config.SPACER}', flush=True)
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
            metr_indexer = [(m, parameters['BBOX_THRESHOLD']) 
                            for m in ('precision', 'recall', 'F1')]
            epoch_info_str = epoch_info.loc[epoch, ['total_summed_loss', *metr_indexer]]
            print_log.append(epoch_info_str.to_frame().T)
            epoch_info_str = pd.concat(print_log)
        print(config.SPACER, '\n', epoch_info_str)
        
        if save_results:
            save_epoch_results(epoch_info, epoch, parameters, train_data, 
                               test_data, model, optimizer, lr_scheduler, 
                               MODELS_DIR, METRICS_DIR, RUN_DIR)

def save_epoch_results(epoch_info, epoch, parameters, train_data, test_data, 
                       model, optimizer, 
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
        train_axon_dets = AxonDetections(model, train_data, parameters, epoch_dir)
        train_axon_dets.detect_dataset()
        draw_all(train_axon_dets, description=f'Notes: {parameters["NOTES"]}',
                 **parameters['PERF_LOG_VIDEO_KWARGS'])
        
        # predict everything in test data and plot result
        test_axon_dets = AxonDetections(model, test_data, parameters, epoch_dir)
        test_axon_dets.detect_dataset()
        draw_all(test_axon_dets, description=f'Notes: {parameters["NOTES"]}',
                 **parameters['PERF_LOG_VIDEO_KWARGS'])
 
def optimize_MCF_params(exp_name, run, epoch='latest', MCF_param_vals={},
                        update_tobest_MCF_params=False):
    RUN_DIR, params = setup_evaluation(exp_name, run)
    params = to_device_specifc_params(params, get_default_parameters(), 
                                      from_cache=OUTPUT_DIR)
    params['LOAD_MODEL'] = [exp_name, run, epoch]

    _, test_data = setup_data(params)
    model, _, _, _ = setup_model(params)

    axon_detections = AxonDetections(model, test_data, params, f'{RUN_DIR}/axon_dets')
    axon_detections.MCF_min_ID_lifetime = 1
    axon_detections.detect_dataset('from')
    axon_detections.assign_ids('from', 'from')
    axon_detections.search_MCF_params(update_tobest_MCF_params, **MCF_param_vals)

if __name__ == '__main__':
    """Set the Experiment Name. Matched with a directory."""
    # exp2_name = 'v1Model_exp2_boxlossfocus'
    # exp3_name = 'v1Model_exp3_newcnn'
    # exp4_name = 'v1Model_exp4_NewCNN'
    # exp5_name = 'v1Model_exp5_RefactIDFocus'
    # exp6_name = 'v1Model_exp6_AxonDetClass'
    exp7_name = 'v1Model_exp7_final'
    exp8_name = 'v1Model_exp8_postsubmission'

    """Delete a subset of experiment runs for tidiness."""
    # clean_rundirs(exp7_name, delete_all_except=(35,36,38,41))
    # clean_rundirs(exp8_name, delete_all_except=(8,9))
    
    """Get the default parameters defined in exp_parameters.py."""
    parameters = get_default_parameters()
    
    """Then adjust these defaults for the specifc experiment run."""
    parameters['NOTES'] = 'ReproduceExp7run35E400Results, trainonly'
    parameters['FROM_CACHE'] = None
    parameters['CACHE'] = None
    parameters['PERF_LOG_VIDEO_KWARGS'] = {'animated':True, 
                                           't_y_x_slice':[(0,50), None, None]}
    # on gpuserver settings
    # parameters['TEST_TIMEPOINTS'] = config.WHOLE_DATASET_TEST_FRAMES
    # parameters['TRAIN_TIMEPOINTS'] = config.WHOLE_DATASET_TRAIN_FRAMES
    parameters['TEST_TIMEPOINTS'] = config.ALLTRAIN_DATASET_TEST_FRAMES
    parameters['TRAIN_TIMEPOINTS'] = config.ALLTRAIN_DATASET_TRAIN_FRAMES
    
    """Run the experiment (model optimization) using the paramters."""
    # run_experiment(exp8_name, parameters, save_results=True)

    
    
    """======================================================================"""
    """======================  WHEN ALL EPOCHS DONE  ========================"""
    """======================================================================"""
    
    
    
    """Option to glue two experiment runs together"""
    # prepend_prev_run(exp8_name, 'run01', 'run00')
    
    """Compare the paramters of two runs"""
    # p2 = load_parameters(exp7_name, 'run36')
    # text = compare_parameters(get_default_parameters(), p2)
    # print(text)

    """Evalute an experiment run looking at:
        1. data preprocessing, model input data
        2. loss and metrics timeline over all epochs
        3. precision, recall, F1 at a particular epoch
        4. a saved model doing inference on test or train data 
        5. the quality of ID assignment over time    
        """
    evaluate_preprocssing(exp8_name, 'run08', show=True)
    # evaluate_training([[exp8_name, 'run08'], [exp8_name, 'run09']], show=True, recreate=True)
    # evaluate_precision_recall([[exp8_name, 'run08', 1000]], show=True, recreate=True)
    evaluate_model(exp8_name, 'run08', draw_true_dets=True, show=False, animated=True,
                   which_dets='IDed', t_y_x_slice=(None, (1500,3000), (1000,4000)))
    
    """Min cost flow assignment hyper parameter optimization"""
    # MCF_param_vals = {'edge_cost_thr_values':  [.1, .3, .4, .6, .7, .8, 1, 2],
    #                   'entry_exit_cost_values': [1, 1.1,  1.7, 2, 2.3, 3],
    #                   'miss_rate_values':  [0.9, 0.6],
    #                   'vis_sim_weight_values': [0, 0.1, .4],
    #                   'conf_capping_method_values': ['ceil' , 'scale_to_max'],}
    # optimize_MCF_params(exp8_name, 'run08', 1000, MCF_param_vals=MCF_param_vals)
    # update_MCF_params(exp8_name, 'run08', 1000)


    """Point 5. from above (ID assignment evaluation)"""
    evaulate_ID_assignment(exp8_name, 'run08', 1000)
    [evaulate_ID_assignment(exp8_name, 'run08', 1000, show=True, col_param=param[:-7])
     for param in MCF_param_vals.keys()]