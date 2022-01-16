import sys
import os

where = os.path.abspath(os.curdir)

# gpu server
if where.startswith('/home/ssteffens'):
    # eth gpu server
    BASE_DIR = '/home/ssteffens/'
    
    TRAINING_DATA_DIR = '/srv/beegfs-data/projects/biohybrid-signal-p/data/training_data/'
    # RAW_INFERENCE_DATA_DIR = '/srv/beegfs-data/projects/biohybrid-signal-p/data/timelapse13_processed_remote/'
    # RAW_INFERENCE_DATA_DIR = '/srv/beegfs-data/projects/biohybrid-signal-p/data/timelapse14_processed_remote'
    RAW_INFERENCE_DATA_DIR = '/srv/beegfs-data/projects/biohybrid-signal-p/data/tl13_tl14_all'
    # SCREENING_DIR = '/srv/beegfs-data/projects/biohybrid-signal-p/data/PDMS_structure_screen/'
    SCREENING_DIR = '/srv/beegfs-data/projects/biohybrid-signal-p/data/PDMS_structure_screen_v2/'
    
    OUTPUT_DIR = '/srv/beegfs-data/projects/biohybrid-signal-p/data/model_output/'

    DEFAULT_DEVICE = 'cuda:0'
    DEFAULT_NUM_WORKERS = 2
    VIDEO_ENCODER = 'ffmpeg'

# local
elif where.startswith('/home/loaloa/gdrive/'):
    BASE_DIR = '/home/loaloa/gdrive/projects/biohybrid MEA/'
    TRAINING_DATA_DIR = BASE_DIR + 'training_data_subs/'
    OUTPUT_DIR = BASE_DIR + 'model_output/'
    
    SCREENING_DIR = '/home/loaloa/ETZ_drive/biohybrid-signal-p/PDMS_structure_screen_v2/'
    RAW_INFERENCE_DATA_DIR = '/home/loaloa/ETZ_drive/biohybrid-signal-p/tl13_tl14_all/'
    
    DEFAULT_DEVICE = 'cpu'
    DEFAULT_NUM_WORKERS = 3
    VIDEO_ENCODER = 'ffmpeg'

SPACER = '========================================================'

# plotting 
TRAIN_Ps = {'linewidth':3, 'alpha':.8, 'linestyle':':'}
TEST_Ps = {'linewidth':3, 'alpha':.8}

PURPLE = '#D00AD4'
BLUE = '#80A8FF'
ORANGE = '#FFAE00'
GREEN = '#01FF6B'
DARK_GRAY = '#6b6b6b'
GRAY = '#858585'
LIGHT_GRAY = '#cfcfcf'
WHITE = '#ffffff'
BLACK = '#000000'
RED = '#c22121'

FIGURE_FILETYPE = 'svg'
SMALL_FONTS = 14.5
FONTS = 18

SMALL_FIGSIZE = (4.5,3.5)
MEDIUM_FIGSIZE = (5.6,4.2)
LARGE_FIGSIZE = (16,8)
D21_COLOR = '#54aeb3'
D22_COLOR = DARK_GRAY
DEFAULT_COLORS = ['#729C27', '#A85C2A', '#1B6C5C', '#892259', '#273673', '#A0F40A', '#08C6A1', '#FF6B0B', '#E20A7C', '#2042CD']

BARPLOT_HEIGHT = 4.2
BAR_WIDTH = .6

um_h = r'[$\frac{\mathrm{\upmu}\mathrm{m}}{\mathrm{h}}$]'
um = r'[$\upmu$m]'
delta = r'$\Delta$'
um_d = r'[$\frac{\mathrm{\upmu}\mathrm{m}}{\mathrm{day}}$]'

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
DESIGN_CMAP = ListedColormap([D21_COLOR, *plt.cm.get_cmap('tab20b', 20).colors])

from cycler import cycler
# plt.rcParams["axes.prop_cycle"] = cycler('color', ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'])
plt.rcParams["axes.prop_cycle"] = cycler('color', DEFAULT_COLORS)


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

MIN_GROWTH_DELTA = 50
DESIGN_FEATURE_NAMES = ['n 2-joints', 'n rescue loops', '2-joint placement', 
                        'channel width', 'rescue loop design', '2-joint design', 
                        'final lane design', 'use spiky tracks']

# after renaming for presentation puposes, structures_screen.identifier and ss.name 
# have a new numbers. This is reflected by the table below
DESIGN_FEATURES = {
    0: ['NA', 'NA', 'NA', 'NA',      'NA', 'NA',     'NA',           'no'],
    
    1: [0,       0, 'NA', 8,         'NA', 'NA',     'neg.Ctrl',     'no'], 
    2: [0,       0, 'NA', 8,         'NA', 'NA',     '3',            'no'], 
    3: [0,       0, 'NA', 3,         'NA', 'NA',     '3',            'no'],
    4: [0,       0, 'NA', 1.5,       'NA', 'NA',     '3',            'no'],
   
    5: [1, 0, 'late',  8, 'NA',      '3',            '3',            'no'],
    6: [3, 0, 'late',  8, 'NA',      '3',            '3',            'no'],
    7: [1, 0, 'early', 8, 'NA',      '3',            '3',            'no'],
    8: [3, 0, 'early', 8, 'NA',      '3',            '3',            'no'],
    
    9: [0, 1, 'NA',    8, 'angled',   'NA',          '3',            'no'],
   10: [0, 3, 'NA',    8, 'angled',   'NA',          '3',            'no'],
   11: [1, 1, 'early', 8, 'angled',   '3',           '3',            'no'],
   12: [1, 3, 'early', 8, 'angled',   '3',           '3',            'no'],
   
   14: [1, 3, 'early', 8, 'angled',   '5',           '3',            'no'],
   16: [1, 3, 'early', 8, 'angled',   '1.5',         '3',            'no'],
   15: [1, 3, 'early', 8, 'angled',   '3, tang.',  '3',            'no'],
   17: [1, 3, 'early', 8, 'straight', '3',           '3',            'no'],

   13: [1, 3, 'early', 8, 'angled',   '3, inlay',    '3',            'no'],
   18: [1, 3, 'early', 8, 'angled',   '3',           '3',            'yes'],
   19: [1, 3, 'early', 8, 'angled',   '3',           '3, wide',    'no'],
   20: [1, 3, 'early', 8, 'angled',   '3',           '1.5, wide',  'no'],

}









        # H24_designs = ['W:0.5um - H:24um',
        #                 'W:0.75um - H:24um',
        #                 'W:1um - H:24um',
        #                 'W:1.25um - H:24um',
        #                 'W:1.5um - H:24um',
        #                 'W:1.75um - H:24um',
        #                 'W:2um - H:24um',
        #                 'W:3um - H:24um',
        #                 'W:4um - H:24um',
        #                 'W:6um - H:24um',
        #                 'W:8um - H:24um',
        #                 'W:10um - H:24um',
        #                 'W:14um - H:24um',
        #                 'W:18um - H:24um',
        #                 'W:22um - H:24um',
        #                 'W:26um - H:24um']

        # H6_designs = ['W:0.5um - H:6um',
        # 'W:0.75um - H:6um',
        # 'W:1um - H:6um',
        # 'W:1.25um - H:6um',
        # 'W:1.5um - H:6um',
        # 'W:1.75um - H:6um',
        # 'W:2um - H:6um',
        # 'W:3um - H:6um',
        # 'W:4um - H:6um',
        # 'W:6um - H:6um',
        # 'W:8um - H:6um',
        # 'W:10um - H:6um',
        # 'W:14um - H:6um',
        # 'W:18um - H:6um',
        # 'W:22um - H:6um',
        # 'W:26um - H:6um',]
        