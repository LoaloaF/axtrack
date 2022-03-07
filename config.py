import sys
import os
from cycler import cycler
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

where = os.path.abspath(os.curdir)

# eth gpu server
if where.startswith('/home/ssteffens'):
    BASE_DIR = '/home/ssteffens/'
    REMOTE_DATA_DIR = '/srv/beegfs-data/projects/biohybrid-signal-p/data/'
    
    TRAINING_DATA_DIR = REMOTE_DATA_DIR + '/training_data/'
    SCREENING_OUT_DIR = REMOTE_DATA_DIR + 'PDMS_structure_screen_v2/'
    RAW_INFERENCE_DATA_DIR = REMOTE_DATA_DIR + '/tl13_tl14_all'
    OUTPUT_DIR = REMOTE_DATA_DIR + '/model_output/'
    DEPLOYED_MODEL_DIR = REMOTE_DATA_DIR + '/deployed_model/'

    DEFAULT_DEVICE = 'cuda:0'
    DEFAULT_NUM_WORKERS = 2
    VIDEO_ENCODER = 'ffmpeg'

# local machine
elif where.startswith('/home/loaloa/gdrive/'):
    BASE_DIR = '/home/loaloa/gdrive/projects/biohybrid MEA/'
    REMOTE_DATA_DIR = '/home/loaloa/lbbgpu_data/'

    TRAINING_DATA_DIR = BASE_DIR + 'training_data_subs/'
    SCREENING_OUT_DIR = REMOTE_DATA_DIR + '/PDMS_structure_screen_v2/'
    RAW_INFERENCE_DATA_DIR = TRAINING_DATA_DIR # locally for debugging/ testing
    # RAW_INFERENCE_DATA_DIR = BASE_DIR + '/timelapse02_processed' # locally for debugging/ testing
    OUTPUT_DIR = BASE_DIR + 'model_output/'
    DEPLOYED_MODEL_DIR = BASE_DIR + '/deployed_model/'
    
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

FIGURE_FILETYPE = 'svg'
VIDEO_FILETYPE = 'mp4'
SMALL_FONTS = 14.5
FONTS = 18

SMALL_FIGSIZE = (4.5,3.5)
MEDIUM_FIGSIZE = (5.6,4.2)
LARGE_FIGSIZE = (14.6,6.8)
BARPLOT_HEIGHT = 4.2
BAR_WIDTH = .6

# color constants 
PURPLE = '#D00AD4'
BLUE = '#80A8FF'
ORANGE = '#FFAE00'
GREEN = '#01FF6B'
GREEN2 = '#88ffba'
DARK_GRAY = '#6b6b6b'
GRAY = '#858585'
LIGHT_GRAY = '#cfcfcf'
WHITE = '#ffffff'
BLACK = '#000000'
RED = '#c22121'
RED2 = '#ba7c7c'
D21_COLOR = '#54aeb3'
D22_COLOR = DARK_GRAY
DEFAULT_COLORS = ['#729C27', '#A85C2A', '#1B6C5C', '#892259', '#273673', '#A0F40A', '#08C6A1', '#FF6B0B', '#E20A7C', '#2042CD']
DESIGN_CMAP = ListedColormap([D21_COLOR, *plt.cm.get_cmap('tab20b', 20).colors])
plt.rcParams["axes.prop_cycle"] = cycler('color', DEFAULT_COLORS)

# TEX labels constans
um_h = r'[$\frac{\mathrm{\upmu}\mathrm{m}}{\mathrm{h}}$]'
um = r'[$\upmu$m]'
delta = r'$\Delta$'
um_d = r'[$\frac{\mathrm{\upmu}\mathrm{m}}{\mathrm{day}}$]'

# threshold for axon growth
MIN_GROWTH_DELTA = 50
DESIGN_FEATURE_NAMES = ['n 2-joints', 'n rescue loops', '2-joint placement', 
                        'channel width', 'rescue loop design', '2-joint design', 
                        'final lane design', 'use spiky tracks']

# Original design names
# DESIGN_FEATURES = {
#     1: [0,       0, 'NA', 8,    'NA', 'NA', 'NA', 'no'], 
#     2: [0,       0, 'NA', 3,    'NA', 'NA', 'NA', 'no'],
#     3: [0,       0, 'NA', 1.5,  'NA', 'NA', 'NA', 'no'],
#     4: ['NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'no'],
   
#    21: [0,       0, 'NA', 8,    'NA', 'NA', 'neg. control', 'no'], 
    
#     5: [1, 0, 'late',  8, 'NA', '3um opening', 'normal', 'no'],
#     6: [3, 0, 'late',  8, 'NA', '3um opening', 'normal', 'no'],
#     7: [1, 0, 'early', 8, 'NA', '3um opening', 'normal', 'no'],
#     8: [3, 0, 'early', 8, 'NA', '3um opening', 'normal', 'no'],
    
#     9: [0, 1, 'NA',    8, 'angled', 'NA',          'normal', 'no'],
#    10: [0, 3, 'NA',    8, 'angled', 'NA',          'normal', 'no'],
#    11: [1, 1, 'early', 8, 'angled', '3um opening', 'normal', 'no'],
#    12: [1, 3, 'early', 8, 'angled', '3um opening', 'normal', 'no'],
   
#    13: [1, 3, 'early', 8, 'angled',   '5um opening',          'normal', 'no'],
#    14: [1, 3, 'early', 8, 'angled',   '1.5um opening',        'normal', 'no'],
#    15: [1, 3, 'early', 8, 'angled',   '3um opening, tangent', 'normal', 'no'],
#    16: [1, 3, 'early', 8, 'straight', '3um opening',          'normal', 'no'],

#    17: [1, 3, 'early', 8, 'angled',   '3um opening, insert', 'normal',   'no'],
#    18: [1, 3, 'early', 8, 'angled',   '3um opening',         'normal',   'yes'],
#    19: [1, 3, 'early', 8, 'angled',   '3um opening',         'wider',    'no'],
#    20: [1, 3, 'early', 8, 'angled',   '3um opening',         'narrower', 'no'],
# }

# after renaming for presentation puposes, structures_screen.identifier and ss.name 
# have a new numbers. This is reflected by the table below
DESIGN_FEATURES = {
    0: ['NA', 'NA', 'NA', 'NA',      'NA', 'NA',     'NA',           'no'],
    
    1: [0, 0, 'NA',    8,       'NA', 'NA',          'neg.Ctrl',     'no'], 
    2: [0, 0, 'NA',    8,       'NA', 'NA',          '3',            'no'], 
    3: [0, 0, 'NA',    3,       'NA', 'NA',          '3',            'no'],
    4: [0, 0, 'NA',    1.5,     'NA', 'NA',          '3',            'no'],
   
    5: [1, 0, 'late',  8,       'NA', '3',           '3',            'no'],
    6: [3, 0, 'late',  8,       'NA', '3',           '3',            'no'],
    7: [1, 0, 'early', 8,       'NA', '3',           '3',            'no'],
    8: [3, 0, 'early', 8,       'NA', '3',           '3',            'no'],
    
    9: [0, 1, 'NA',    8, 'angled',   'NA',          '3',            'no'],
   10: [0, 3, 'NA',    8, 'angled',   'NA',          '3',            'no'],
   11: [1, 1, 'early', 8, 'angled',   '3',           '3',            'no'],
   12: [1, 3, 'early', 8, 'angled',   '3',           '3',            'no'],
   
   14: [1, 3, 'early', 8, 'angled',   '5',           '3',            'no'],
   16: [1, 3, 'early', 8, 'angled',   '1.5',         '3',            'no'],
   15: [1, 3, 'early', 8, 'angled',   '3, tang.',    '3',            'no'],
   17: [1, 3, 'early', 8, 'straight', '3',           '3',            'no'],

   13: [1, 3, 'early', 8, 'angled',   '3, inlay',    '3',            'no'],
   18: [1, 3, 'early', 8, 'angled',   '3',           '3',            'yes'],
   19: [1, 3, 'early', 8, 'angled',   '3',           '3, wide',      'no'],
   20: [1, 3, 'early', 8, 'angled',   '3',           '1.5, wide',    'no'],
}
