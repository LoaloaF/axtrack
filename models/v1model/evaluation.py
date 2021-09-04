import os
from copy import copy

from exp_parameters import (
    get_default_parameters, 
    to_device_specifc_params,
    compare_parameters,
    )
from core_functionality import (
    setup_data, 
    setup_model, 
    )
from utils import (
    save_preproc_metrics,
    create_lossovertime_pkl,
    create_metricsovertime_pkl,
    get_run_dir,
    set_seed,
    )
from plotting import (
    plot_preprocessed_input_data, 
    plot_training_process,
    plot_prc_rcl,
    draw_all,
    plot_axon_IDs,
    )
from AxonDetections import AxonDetections
from exp_parameters import load_parameters, params2text
from config import OUTPUT_DIR, SPACER

def setup_evaluation(exp_name, run, print_params=True):
    EXP_DIR = f'{OUTPUT_DIR}/runs/{exp_name}/'
    RUN_DIR = get_run_dir(EXP_DIR, run)
    parameters = load_parameters(exp_name, run)
    if print_params:
        print(params2text(parameters))
    set_seed(parameters['SEED'])
    return RUN_DIR, parameters

def evaluate_preprocssing(exp_name, run, show=True):
    print('Evaluating preprocessing steps...', end='')
    # get the matching directory for the comb of experiment name and run name
    RUN_DIR, params = setup_evaluation(exp_name, run)
    PREPROC_DATA_DIR = f'{RUN_DIR}/preproc_data/'

    # check if the preprocessed data file exists already, if not recreate
    preproc_file = f'{PREPROC_DATA_DIR}/preprocessed_data.csv'
    if not os.path.exists(preproc_file):
        train_data, test_data = setup_data(parameters)
        save_preproc_metrics(RUN_DIR, train_data, test_data)
    # plot the input data distribution over taken preprocessing steps
    plot_preprocessed_input_data(preproc_file, dest_dir=RUN_DIR, show=show)
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
            print(f'Evaluating training of {lbl}...', end='')
        else:
            print(compare_parameters(base_params, params))

        
        loss_file = f'{METRICS_DIR}/loss_all_epochs.pkl'
        metrics_file = f'{METRICS_DIR}/metrics_all_epochs.pkl'
    
        # check if there exists a prepended version of that run
        loss_file_prd = loss_file.replace('.pkl', '_prepend.pkl')
        metrics_file_prd = metrics_file.replace('.pkl', '_prepend.pkl')
        if use_prepend_ifavail and os.path.exists(loss_file_prd) and os.path.exists(metrics_file_prd):
            loss_file, metrics_file = loss_file_prd, metrics_file_prd
        # if files files don't exists yet, make them here 
        if not os.path.exists(loss_file) or recreate:
            loss_file = create_lossovertime_pkl(METRICS_DIR)
            metrics_file = create_metricsovertime_pkl(METRICS_DIR)
        training[lbl] = [loss_file, metrics_file, params]

    # plot the loss and metrics over epochs
    plot_training_process(training, dest_dir=dest_dir, show=show)
    print('Done. ')
    
def evaluate_model(exp_name, run, epoch='latest', assign_ids=False, **kwargs):
    RUN_DIR, params = setup_evaluation(exp_name, run)
    params = to_device_specifc_params(params, get_default_parameters())
    params['LOAD_MODEL'] = [exp_name, run, epoch]
    params['DEVICE'] = 'cpu'

    model, _, _, _ = setup_model(params)
    train_data, test_data = setup_data(params)
    for data in test_data, train_data:
        
        axon_detections = AxonDetections(model, data, params, f'{RUN_DIR}/axon_detections', assign_id_at_init=assign_ids)
        
        os.makedirs(f'{RUN_DIR}/model_out', exist_ok=True)
        fname = f'{data.name}_E:{epoch}_timepoint:---|{data.sizet}'
        draw_all(axon_detections, fname, dest_dir=f'{RUN_DIR}/model_out', notes=params["NOTES"], **kwargs)
        break

        plot_axon_IDs(axon_detections, dest_dir=f'{RUN_DIR}/model_out', show=False)

def evaluate_precision_recall(exp_run_epoch_ids, show=True):
    metrics = {}
    for i, (exp_name, run, epoch) in enumerate(exp_run_epoch_ids):
        RUN_DIR, params = setup_evaluation(exp_name, run, print_params=False)
        lbl = f"{run} E{epoch} - {params['NOTES']}"
        
        if i == 0:
            print(f'Evaluating precision/recall of {lbl}...', end='')
            base_params = params
            dest_dir = RUN_DIR
        else:
            print(compare_parameters(base_params, params))

        metrics_files = [f'{RUN_DIR}/metrics/E{e:0>4}_metrics.pkl' for e in range(epoch-15,epoch+15)]
        # for the last Epoch, there are no 15 epochs into the future, delete those
        metrics_files = [file for file in metrics_files if os.path.exists(file)]
        metrics[lbl] = metrics_files
    plot_prc_rcl(metrics, dest_dir=dest_dir, show=show)
    print('Done.')