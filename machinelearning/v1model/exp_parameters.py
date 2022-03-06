import os
import pickle
from collections import OrderedDict

from torch import nn as nn
import matplotlib.pyplot as plt

from config import TRAINING_DATA_DIR, OUTPUT_DIR, DEFAULT_DEVICE, DEFAULT_NUM_WORKERS, SPACER
from utils import get_run_dir, architecture_to_text

def get_default_parameters():
    # DATA
    TIMELAPSE_FILE = TRAINING_DATA_DIR + 'training_timelapse.tif'
    LABELS_FILE = TRAINING_DATA_DIR + 'axon_anchor_labels.csv'
    MASK_FILE = TRAINING_DATA_DIR + 'training_mask.npy'
    TRAIN_TIMEPOINTS = range(4,33)
    TEST_TIMEPOINTS = list(range(2,4)) + list(range(33,35))
    
    LOG_CORRECT = True
    PLOT_PREPROC = True
    STANDARDIZE = ('zscore', None)

    STANDARDIZE_FRAMEWISE = False
    TEMPORAL_CONTEXT = 2
    USE_MOTION_DATA = 'exclude' #, 'include'  'only'
    USE_SPARSE = False
    USE_TRANSFORMS = ['vflip', 'hflip', 'rot', 'translateY', 'translateX']
    CLIP_LOWERLIM = 55 /2**16
    OFFSET = None
    PAD = [0,300,0,300]
    CACHE = None
    FROM_CACHE = OUTPUT_DIR
    SHUFFLE = True
    DROP_LAST = False

    # MODEL
    ARCHITECTURE = [
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
        (3, 160, 1,  1),      # y-x out: 16
         ],      
        [('FC', 1024),
         ('activation', nn.Sigmoid()), 
         ('FC', 1024),
         ('activation', nn.Sigmoid()), 
        ]
    ]

    IMG_DIM = 2920, 6364
    SY, SX = 12, 12
    TILESIZE = 512
    # GROUPING = False
    ACTIVATION_FUNCTION = nn.LeakyReLU(0.1)

    # ID stuff
    NON_MAX_SUPRESSION_DIST = 23

    WEIGHT_DECAY = 5e-4
    BATCH_SIZE = 32
    EPOCHS = 1501
    LOAD_MODEL = None # ['Exp1_yolo_only_firstfirst', 'run02', 'E0']   # [ExpName, #run, #epoch]
    BBOX_THRESHOLD = .7
    LR = 5e-4
    LR_DECAYRATE = 15

    # LOSS
    L_OBJECT = 49.5
    L_NOBJECT = 1
    L_COORD_ANCHOR = 49.5

    # RUN SETTINGS
    SEED = 42
    DEVICE = DEFAULT_DEVICE
    NUM_WORKERS = DEFAULT_NUM_WORKERS
    PIN_MEMORY = True
    NOTES = 'no notes - shame on you!'
    MODEL_CHECKPOINTS = (1, 250, 750, 1000, 1500)
    PERF_LOG_VIDEO_KWARGS = {}

    param_dict = OrderedDict({key: val for key, val in locals().items()})
    return param_dict

def write_parameters(file, params):
    with open(file.replace('pkl', 'txt'), 'w') as txt_file:  
        txt_file.writelines([(f'{key:20} {val}\n') for key, val in params.items()])
    pickle.dump(params, open(file, 'wb'))

def load_parameters(exp_name, run, from_directory=None):
    if exp_name and run:
        EXP_DIR = f'{OUTPUT_DIR}/runs/{exp_name}/'
        RUN_DIR = get_run_dir(EXP_DIR, run)
        file = f'{RUN_DIR}/params.pkl'
    elif from_directory:
        file = f'{from_directory}/params.pkl'
    return pickle.load(open(file, 'rb'))

def get_notes(exp_name, run):
    return load_parameters(exp_name, run)['NOTES']

def params2text(params):
    text = SPACER +'\n'
    for key, val in params.items():
        if key == 'TIMELAPSE_FILE':
            text += '\n\t>> data parameters <<\n'
        elif key == 'ARCHITECTURE':
            text += '\n\t>> model & training <<\n'
            text += f'\t\t{key}'
            text += architecture_to_text(val)
            continue
        elif key == 'LOSS':
            text += '\n\t>> loss <<\n'
        elif key == 'L_OBJECT':
            text += '\n\t>> loss <<\n'
        elif key == 'SEED':
            text += '\n\t>> run settings <<\n'
        if key.endswith('TIMEPOINTS') and len(val) >30:
            n = len(val)
            val = f'{val[:5]} ... {val[n//2:n//2+5]} ... {val[-5:]} (n={n})'
        text += f'\t\t{key:20} {val}\n'
    text += SPACER+'\n'
    return text

def check_parameters(passed_params, default_params):
    inval_keys = [key for key in passed_params if key not in default_params]
    if inval_keys:
        print(f'Invalid parameters passed: {inval_keys}')
        exit(1)

def to_device_specifc_params(model_parameters, local_default_params, 
                             from_cache=None, cache=None):
    to_update = ('TIMELAPSE_FILE', 'LABELS_FILE', 'MASK_FILE', 'DEVICE')
    [model_parameters.update({key: local_default_params[key]}) for key in to_update]
    if from_cache:
        model_parameters['FROM_CACHE'] = from_cache
    if cache:
        model_parameters['CACHE'] = cache
    return model_parameters

def compare_parameters(param1, param2):
    text = ''
    param1_only = [key for key in param1 if key not in param2]
    param2_only = [key for key in param2 if key not in param1]
    
    text += '\n'+SPACER
    text += '\nParamters only in P1:\n'
    if param1_only:
        text += '\n'.join([f'\t{key}: {param1[key]}'for key in param1_only])
        text += '\n'+SPACER+'\n'
    
    text += '\n'+SPACER
    text += '\nParamters only in P2:\n'
    if param2_only:
        text += '\n'.join([f'\t{key}: {param2[key]}'for key in param2_only])
        text += '\n'+SPACER+'\n'

    text += '\n'+SPACER
    text += '\nParamters that differ:\n'
    for key in param1.keys():
        if key in param1_only: 
            continue
        if param1[key] != param2[key]:
            text += f'\n{key}:'
            if key == 'ARCHITECTURE':
                text += f'\n\t\t\t\tP1: {architecture_to_text(param1[key])}:'
                text += f'\n\t\t\t\tP2: {architecture_to_text(param2[key])}:'
            else:
                text += f'\n\tP1: {param1[key]}:'
                text += f'\n\tP2: {param2[key]}:'
    text += '\n'+SPACER+'\n'
    return text