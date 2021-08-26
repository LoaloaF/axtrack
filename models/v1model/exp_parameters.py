
import os
import pickle
from collections import OrderedDict

from torch import nn as nn
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager

from config import RAW_DATA_DIR, OUTPUT_DIR, DEFAULT_DEVICE, DEFAULT_NUM_WORKERS, SPACER

def get_default_parameters():
    # DATA
    TIMELAPSE_FILE = RAW_DATA_DIR + 'G001_red_compr.deflate.tif'
    LABELS_FILE = OUTPUT_DIR + 'labelled_axons_astardists.csv'
    MASK_FILE = OUTPUT_DIR + 'mask_wells_excl.npy'
    TRAIN_TIMEPOINTS = range(4,33)
    TEST_TIMEPOINTS = list(range(2,4)) + list(range(33,35))
    LOG_CORRECT = True
    PLOT_PREPROC = True
    STANDARDIZE = ('zscore', None)
    # RETAIN_TEMPORAL_VAR = False

    STANDARDIZE_FRAMEWISE = False
    TEMPORAL_CONTEXT = 2
    USE_MOTION_DATA = 'exclude' #, 'include'  'only'
    USE_SPARSE = False
    USE_TRANSFORMS = ['vflip', 'hflip', 'rot', 'translateY', 'translateX']
    CLIP_LOWERLIM = 43 /2**16
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
         ('FC', 1024),
         ('activation', nn.Sigmoid()), 
        ]     
    ]
    IMG_DIM = 2920, 6364
    SY, SX = 12, 12
    TILESIZE = 512
    # GROUPING = False
    ACTIVATION_FUNCTION = nn.LeakyReLU(0.1)

    WEIGHT_DECAY = 5e-4
    BATCH_SIZE = 32
    EPOCHS = 301
    LOAD_MODEL = None
    # LOAD_MODEL = ['Exp1_yolo_only_firstfirst', 'run02', 'E0']   # [ExpName, #run, #epoch]
    BBOX_THRESHOLD = .7
    LR = 1e-4
    LR_DECAYRATE = 0
    # LR_DECAY_STARTEPOCH = 0

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

    param_dict = OrderedDict({key: val for key, val in locals().items()})
    return param_dict

def write_parameters(file, params):
    with open(file.replace('pkl', 'txt'), 'w') as txt_file:  
        txt_file.writelines([(f'{key:20} {val}\n') for key, val in params.items()])
    pickle.dump(params, open(file, 'wb'))

def load_parameters(exp_name, run):
    EXP_DIR = f'{OUTPUT_DIR}/runs/{exp_name}/'
    RUN_DIR = f'{EXP_DIR}/{[rd for rd in os.listdir(EXP_DIR) if run in rd][0]}'
    file = f'{RUN_DIR}/params.pkl'
    return pickle.load(open(file, 'rb'))

def params2text(params):
    text = SPACER +'\n'
    for key, val in params.items():
        if key == 'TIMELAPSE_FILE':
            text += '\n\t>> data parameters <<\n'
        elif key == 'ARCHITECTURE':
            text += '\n\t>> model & training <<\n'
            text += f'\t\t{key:20} {val[0][0]}\n'
            for layer in val[0][1:]:
                text += f'\t\t\t\t\t\t\t {layer}\n'
            text += '\t\t\t\t\t\t\t== cat CNN features ==\n'
            for layer in val[1]:
                text += f'\t\t\t\t\t\t\t {layer}\n'
            text += '\t\t\t\t\t\t\t== FullyConnected Head ==\n'
            for layer in val[2]:
                text += f'\t\t\t\t\t\t\t {layer}\n'
            continue
        elif key == 'LOSS':
            text += '\n\t>> loss <<\n'
        elif key == 'L_OBJECT':
            text += '\n\t>> loss <<\n'
        elif key == 'SEED':
            text += '\n\t>> run settings <<\n'
        text += f'\t\t{key:20} {val}\n'
    text += SPACER+'\n'
    return text

def params2img(fname, params, show=False):
    fontfiles = font_manager.findSystemFonts(fontpaths='/home/loaloa/.fonts/c/', fontext='ttf')
    [font_manager.fontManager.addfont(font_file) for font_file in fontfiles]

    fig, ax = plt.subplots(figsize=(9,9))
    ax.axis('off')
    txt = params2text(params).replace("\t", "    ")
    hfont = {'fontname':'CMU Typewriter Text'}
    ax.text(0,0, txt, fontsize=9, **hfont)

    if show:
        plt.show()
    fig.savefig(fname, dpi=600)
    plt.close()

def check_parameters(passed_params, default_params):
    inval_keys = [key for key in passed_params if key not in default_params]
    if inval_keys:
        print(f'Invalid parameters passed: {inval_keys}')
        exit(1)

def to_device_specifc_params(model_parameters, local_default_params):
    to_update = ('TIMELAPSE_FILE', 'LABELS_FILE', 'MASK_FILE', 'DEVICE', 'FROM_CACHE')
    [model_parameters.update({key: local_default_params[key]}) for key in to_update]
    return model_parameters

def compare_parameters(param1, param2):
    text = SPACER
    param1_only = [key for key in param1 if key not in param2]
    param2_only = [key for key in param2 if key not in param1]
    text += '\nParamters only in P1:\n'
    text += '\n'.join([f'\t{key}: {param1[key]}'for key in param1_only])
    text += '\nParamters only in P2:\n'
    text += '\n'.join([f'\t{key}: {param2[key]}'for key in param2_only])
    text += '\n'+SPACER

    for key in param1.keys():
        if key in param1_only: 
            continue
        if param1[key] != param2[key]:
            text += f'\n{key}:'
            text += f'\n\tP1: {param1[key]}:'
            text += f'\n\tP2: {param2[key]}:'
    
    print(text)