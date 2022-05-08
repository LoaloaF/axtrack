import sys
import os
from cycler import cycler
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

where = os.path.dirname(__file__) + '/../'
# print(where)

# eth gpu server
if os.path.exists('/home/ssteffens/'):
    REMOTE_DATA_DIR = '/srv/beegfs-data/projects/biohybrid-signal-p/data/'
    
    TRAINING_DATA_DIR = REMOTE_DATA_DIR + '/training_data/'
    TRAINING_DATA_DIR = REMOTE_DATA_DIR + '/training_data_subs/'
    OUTPUT_DIR = REMOTE_DATA_DIR + '/model_output/'
    BASE_DIR = '/home/ssteffens/'
    DEPLOYED_MODEL_DIR = BASE_DIR + '/code/axtrack/deployed_model/'

    DEFAULT_DEVICE = 'cuda:0'
    DEFAULT_NUM_WORKERS = 2
    VIDEO_ENCODER = 'ffmpeg'

# local machine
elif os.path.exists('/home/loaloa/gdriveeeeeeeee/'):
    BASE_DIR = '/home/loaloa/gdrive/projects/biohybrid MEA/'
    REMOTE_DATA_DIR = '/home/loaloa/lbbgpu_data/'

    TRAINING_DATA_DIR = BASE_DIR + 'training_data_subs/'
    OUTPUT_DIR = BASE_DIR + 'model_output/'
    DEPLOYED_MODEL_DIR = BASE_DIR + '/code/deployed_model/'
    
    DEFAULT_DEVICE = 'cpu'
    DEFAULT_NUM_WORKERS = 3
    VIDEO_ENCODER = 'ffmpeg'
else:
    BASE_DIR = where
    # DATA_DIR = '/run/media/loaloa/'
    DEPLOYED_MODEL_DIR = BASE_DIR + '/deployed_model/'
    # REMOTE_DATA_DIR = '/home/loaloa/lbbgpu_data/'

    TRAINING_DATA_DIR = BASE_DIR + 'training_data_subs/'
    OUTPUT_DIR = BASE_DIR + 'model_output/'
    
    DEFAULT_DEVICE = 'cpu'
    DEFAULT_NUM_WORKERS = 3
    VIDEO_ENCODER = 'ffmpeg'
    

SPACER = '========================================================'

# train test frames constants
# Dat1 37 frames - Dat2 80 frames - Dat3 210 frames 
WHOLE_DATASET_TRAIN_FRAMES = list(range(2, 37+80-20-4)) + list(range(37+80+20+4, 37+80+210-2))
WHOLE_DATASET_TEST_FRAMES = list(range(37+80-20, 37+80+20))
ALLTRAIN_DATASET_TRAIN_FRAMES = range(2,37+80+210-2)
ALLTRAIN_DATASET_TEST_FRAMES = range(2,6)  # dummy

# plotting 
TRAIN_Ps = {'linewidth':3, 'alpha':.8, 'linestyle':':'}
TEST_Ps = {'linewidth':3, 'alpha':.8}

PREDICTED_BOXES_KWARGS = {'alpha':1, 'linestyle':'dashed', 'linewidth':1, 
                          'facecolor':'none', 'edgecolor':'hsv'}
GROUNDTRUTH_BOXES_KWARGS = {'alpha':.6, 'linestyle':'solid', 'linewidth':1.2, 
                            'facecolor':'none', 'edgecolor':'white'}
FP_BOXES_KWARGS = {'alpha':.8, 'linestyle':'solid', 'linewidth':1, 
                   'facecolor':'none', 'edgecolor':'orange'}
FN_BOXES_KWARGS = {'alpha':.8, 'linestyle':'solid', 'linewidth':1, 
                   'facecolor':'none', 'edgecolor':'teal'}

FIGURE_FILETYPE = 'svg'
VIDEO_FILETYPE = 'mp4'
SMALL_FONTS = 14.5
FONTS = 18

SMALL_FIGSIZE = (4.5,3.5)
MEDIUM_FIGSIZE = (5.6,4.2)
LARGE_FIGSIZE = (14.6,6.8)
BARPLOT_HEIGHT = 4.2

DARK_GRAY = '#6b6b6b'
GRAY = '#858585'
LIGHT_GRAY = '#cfcfcf'
DEFAULT_COLORS = ['#729C27', '#A85C2A', '#1B6C5C', '#892259', '#273673', 
                  '#A0F40A', '#08C6A1', '#FF6B0B', '#E20A7C', '#2042CD']
plt.rcParams["axes.prop_cycle"] = cycler('color', DEFAULT_COLORS)