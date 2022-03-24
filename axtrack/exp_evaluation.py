import os
import pandas as pd

from .exp_parameters import (
    get_default_parameters, 
    to_device_specifc_params,
    compare_parameters,
    )
from .machinelearning.core_functionality import (
    setup_data, 
    setup_model, 
    )
from .utils import (
    save_preproc_metrics,
    get_all_epoch_data,
    get_run_dir,
    set_seed,
    turn_tex,
    )
from .ml_plotting import (
    plot_preprocessed_input_data, 
    plot_training_process,
    plot_prc_rcl,
    plot_IDassignment_performance,
    )
from .video_plotting import  (
    draw_all,
    )
from .AxonDetections import AxonDetections
from .exp_parameters import load_parameters, params2text
from .config import OUTPUT_DIR, SPACER

def setup_evaluation(exp_name, run, print_params=True):
    EXP_DIR = f'{OUTPUT_DIR}/runs/{exp_name}/'
    RUN_DIR = get_run_dir(EXP_DIR, run)
    parameters = load_parameters(exp_name, run)
    turn_tex('on')
    if print_params:
        print(params2text(parameters))
    set_seed(parameters['SEED'])
    return RUN_DIR, parameters

def evaluate_preprocssing(exp_name, run, show=True):
    # get the matching directory for the comb of experiment name and run name
    RUN_DIR, params = setup_evaluation(exp_name, run)
    PREPROC_DATA_DIR = f'{RUN_DIR}/preproc_data/'

    # check if the preprocessed data file exists already, if not recreate
    # preproc_file = f'{RUN_DIR}/preprocessed_data.csv'
    preproc_file = f'{PREPROC_DATA_DIR}/preprocessed_data.csv'
    if not os.path.exists(preproc_file):
        train_data, test_data = setup_data(params)
        save_preproc_metrics(RUN_DIR, train_data, test_data)
    # plot the input data distribution over taken preprocessing steps
    data = pd.read_csv(preproc_file, header=[0,1,2], index_col=0)
    
    # plot the different preprocessing outcomes
    print('Evaluating preprocessing steps...', end='')
    plot_preprocessed_input_data(data, params['NOTES'], dest_dir=RUN_DIR, show=show)
    print('Done.')

def evaluate_training(exp_run_ids, recreate=False, use_prepend_ifavail=True, show=True):
    # aggregate the loss, metics and params for each of the passed exp-run combs
    training = {}
    for i, (exp_name, run) in enumerate(exp_run_ids):
        RUN_DIR, params = setup_evaluation(exp_name, run, print_params=len(exp_run_ids)==1)
        METRICS_DIR = f'{RUN_DIR}/metrics/'
        lbl = f"{run} - {params['NOTES']}"
        if i == 0:
            dest_dir = RUN_DIR
            base_params = params
        else:
            print(compare_parameters(base_params, params))
        
        # get the training data over time (loss + metrics)
        training[lbl], _ = get_all_epoch_data(exp_name, run, recreate, 
                                              use_prepend_ifavail)

    # plot the loss and metrics over epochs
    print(f'Evaluating training of {lbl}...', end='')
    plot_training_process(training, dest_dir=dest_dir, show=show)
    print('Done. ')
    
def evaluate_precision_recall(exp_run_epoch_ids, show=True, avg_over_t=30,
                              recreate=False, use_prepend_ifavail=True):
    metrics = {}
    for i, (exp_name, run, epoch) in enumerate(exp_run_epoch_ids):
        RUN_DIR, params = setup_evaluation(exp_name, run, print_params=False)
        METRICS_DIR = f'{RUN_DIR}/metrics/'
        lbl = f"{run} E{epoch:0>3} - {params['NOTES']}"
        
        if i == 0:
            base_params = params
            dest_dir = RUN_DIR
        else:
            print(compare_parameters(base_params, params))
        
        # get the training data over time, slice to certain epoch range, avg over epochs
        _, dat = get_all_epoch_data(exp_name, run, recreate, use_prepend_ifavail)
        metrics[lbl] = dat.loc[epoch-avg_over_t//2 :epoch+avg_over_t//2+1].dropna().mean()
    
    # plot precision recall
    print(f'Evaluating precision/recall of {lbl}...', end='')
    plot_prc_rcl(metrics, dest_dir=dest_dir, show=show)
    print('Done.')

def evaluate_model(exp_name, run, epoch='latest', which_data='test', 
                which_dets='confident', show=True,
                cache_detections='to', astar_paths_cache='to', 
                assigedIDs_cache='to', **kwargs):
    print('\nEvaluating model...', end='')
    RUN_DIR, params = setup_evaluation(exp_name, run)
    params = to_device_specifc_params(params, get_default_parameters(), 
                                      from_cache=OUTPUT_DIR)
    params['LOAD_MODEL'] = [exp_name, run, epoch]

    train_data, test_data = setup_data(params)
    data = test_data if which_data == 'test' else train_data
    model, _, _, _ = setup_model(params)

    dest_dir = f'{RUN_DIR}/axon_dets'
    axon_detections = AxonDetections(model, data, params, dest_dir)
    axon_detections.detect_dataset(cache=cache_detections)
    
    if which_dets == 'IDed':
        axon_detections.assign_ids(astar_paths_cache, assigedIDs_cache)
    
    description = f'{exp_name}, {run}, Epoch:{epoch}, Notes: {params["NOTES"]}'
    draw_all(axon_detections, which_dets=which_dets, show=show,
             description=description, **kwargs)

def evaulate_ID_assignment(exp_name, run, epoch='latest', show=True, **kwargs):
    RUN_DIR, params = setup_evaluation(exp_name, run)
    results_fname = f'{RUN_DIR}/axon_dets/MCF_params_results.csv'

    if not os.path.exists(results_fname):
        print('Run optimize_MCF_params() first to evaluate MCF parameters!')
        exit(1)

    results = pd.read_csv(results_fname, index_col=0)
    plot_IDassignment_performance(results, dest_dir=f'{RUN_DIR}/axon_dets/', 
                                  show=show, **kwargs)