import sys
import os

where = os.path.abspath(os.curdir)

# gpu server
if where.startswith('/home/ssteffens'):
    # eth gpu server
    BASE_DIR = '/home/ssteffens/'
    
    RAW_TRAINING_DATA_DIR = '/srv/beegfs-data/projects/biohybrid-signal-p/data/timelapse01_40_min_processed/'
    RAW_INFERENCE_DATA_DIR = '/srv/beegfs-data/projects/biohybrid-signal-p/data/timelapse13_processed_remote/'
    # RAW_INFERENCE_DATA_DIR = '/srv/beegfs-data/projects/biohybrid-signal-p/data/tl13_tl14_all'
    SCREENING_DIR = '/srv/beegfs-data/projects/biohybrid-signal-p/data/PDMS_structure_screen/'
    
    DEFAULT_DEVICE = 'cuda:0'
    DEFAULT_NUM_WORKERS = 2
    VIDEO_ENCODER = 'ffmpeg'

# local
elif where.startswith('/home/loaloa/gdrive/'):
    BASE_DIR = '/home/loaloa/gdrive/projects/biohybrid MEA/'
    RAW_TRAINING_DATA_DIR = '/home/loaloa/Documents/'
    
    LOCAL_RAW_INFERENCE_DATA_DIR = '/home/loaloa/Documents/timelapse13_processed/'
    LOCAL_SCREENING_DIR = '/home/loaloa/Documents/PDMS_structure_screen/'
    
    SCREENING_DIR = '/home/loaloa/ETZ_drive/biohybrid-signal-p/PDMS_structure_screen/'
    RAW_INFERENCE_DATA_DIR = '/home/loaloa/ETZ_drive/biohybrid-signal-p/timelapse13_processed_remote/'
    
    DEFAULT_DEVICE = 'cpu'
    DEFAULT_NUM_WORKERS = 3
    VIDEO_ENCODER = 'ffmpeg'

OUTPUT_DIR = BASE_DIR + 'tl140_outputdata/'
SPACER = '========================================================'

# plotting 
TRAIN_Ps = {'linewidth':1, 'alpha':.7, 'linestyle':'-.'}
TEST_Ps = {'linewidth':1, 'alpha':.7}

PURPLE = '#D00AD4'
BLUE = '#80A8FF'
ORANGE = '#FFAE00'
GREEN = '#01FF6B'
DARK_GRAY = '#6b6b6b'
GRAY = '#858585'
LIGHT_GRAY = '#cfcfcf'
WHITE = '#ffffff'
BLACK = '#000000'
FIGURE_FILETYPE = 'png'
SMALL_FONTS = 12.5
FONTS = 16

SMALL_FIGSIZE = (4.5,3.5)
MEDIUM_FIGSIZE = (8.5,5.5)
D21_COLOR = '#54aeb3'
D22_COLOR = DARK_GRAY

BARPLOT_HEIGHT = 3.5

um_h = r'$[\frac{\mathrm{\upmu}\mathrm{m}}{\mathrm{h}}]$'
um_d = r'$[\frac{\mathrm{\upmu}\mathrm{m}}{\mathrm{day}}]$'

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
DESIGN_CMAP = ListedColormap([*plt.cm.get_cmap('tab20b', 20).colors, D21_COLOR, D22_COLOR])