import sys
import os

where = os.path.abspath(os.curdir)

# gpu server
if where.startswith('/home/ssteffens'):
    # eth gpu server
    BASE_DIR = '/home/ssteffens/'
    RAW_DATA_DIR = '/srv/beegfs-data/projects/biohybrid-signal-p/data/timelapse13_processed_remote/'
    SCREENING_DIR = '/srv/beegfs-data/projects/biohybrid-signal-p/data/PDMS_structure_screen/'
    DEFAULT_DEVICE = 'cuda:0'
    DEFAULT_NUM_WORKERS = 2
    VIDEO_ENCODER = 'ffmpeg'

# local
elif where.startswith('/home/loaloa/gdrive/'):
    BASE_DIR = '/home/loaloa/gdrive/projects/biohybrid MEA/'
    RAW_DATA_DIR = '/home/loaloa/Documents/timelapse13_processed/'
    SCREENING_DIR = '/home/loaloa/Documents/PDMS_structure_screen/'
    DEFAULT_DEVICE = 'cpu'
    DEFAULT_NUM_WORKERS = 3
    VIDEO_ENCODER = 'imagemagick'

OUTPUT_DIR = BASE_DIR + 'tl140_outputdata/'
SPACER = '========================================================'

# plotting 
TRAIN_Ps = {'linewidth':1, 'alpha':.7, 'linestyle':'-.'}
TEST_Ps = {'linewidth':1, 'alpha':.7}