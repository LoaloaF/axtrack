import sys
import os

where = os.path.abspath(os.curdir)

# gpu server
if where.startswith('/home/ssteffens'):
    # eth gpu server
    BASE_DIR = '/home/ssteffens/'
    RAW_DATA_DIR = BASE_DIR + 'raw_data_dir/'
    DEFAULT_DEVICE = 'cuda:0'
    DEFAULT_NUM_WORKERS = 2

# local
elif where.startswith('/home/loaloa/gdrive/'):
    BASE_DIR = '/home/loaloa/gdrive/projects/biohybrid MEA/'
    # RAW_DATA_DIR = '/run/media/loaloa/lbb_ssd/timelapse1_40_min_processed/'
    # RAW_DATA_DIR = '/run/media/loaloa/large_hdrive/timelapse1/40_min_processed/'
    RAW_DATA_DIR = '/home/loaloa/Documents/'
    DEFAULT_DEVICE = 'cpu'
    DEFAULT_NUM_WORKERS = 3

# colab
elif where.startswith('/content'):
    BASE_DIR = '/content/gdrive/MyDrive/bioMEA_colab/'
    RAW_DATA_DIR = BASE_DIR + 'raw_data_dir/'
    # OUTPUT_DIR = '/content/tl140_outputdata/'
    DEFAULT_DEVICE = 'cuda'
    DEFAULT_NUM_WORKERS = 2

OUTPUT_DIR = BASE_DIR + 'tl140_outputdata/'
SPACER = '========================================================'

# plotting 
TRAIN_Ps = {'linewidth':1, 'alpha':.7, 'linestyle':'-.'}
TEST_Ps = {'linewidth':1, 'alpha':.7}