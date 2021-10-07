import os
import pickle
from copy import copy

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix

from tifffile import imread

from Timelapse import Timelapse

class UnlabelledTimelapse(Timelapse):
    def __init__(self, path_to_files, imseq_fname, mask_fname, metadata_fname,
                 prepr_params, plot_preporcessing=True, cache=None, 
                 from_cache=None):
                
        # define the segmented data filenames
        metadata = self._load_metadata(f'{path_to_files}/{metadata_fname}')
        self.name = f'{metadata["design"]}_{metadata["CLSM_area"]}'
        imseq_path = f'{path_to_files}/{imseq_fname}'
        mask_path = f'{path_to_files}/{mask_fname}'
        labels_csv = None

        # slice the timepoints so that t0 and t-1 work with temporal context
        ntimepoints = metadata.get('ntimepoints')
        if not ntimepoints:
            ntimepoints = imread(imseq_path).shape[0]
            ntimepoints = 40
        timepoints = range(prepr_params['TEMPORAL_CONTEXT'], 
                           (ntimepoints-prepr_params['TEMPORAL_CONTEXT']))
        # no image augmentation, pad edges with 50 zeros
        use_transforms = []
        pad = [50,50,50,50]

        prepr_params['CLIP_LOWERLIM'] = 45/2**16

        # create a timelapse object with a dummy labels.csv
        super().__init__(imseq_path = imseq_path,
                        labels_csv = labels_csv,
                        mask_path = mask_path,
                        timepoints = timepoints,
                        use_transforms = use_transforms,
                        pad = pad, 
                        name = self.name,
                        log_correct = prepr_params['LOG_CORRECT'],
                        standardize_framewise = prepr_params['STANDARDIZE_FRAMEWISE'],
                        standardize = prepr_params['STANDARDIZE'],
                        use_motion_filtered = prepr_params['USE_MOTION_DATA'],
                        use_sparse = prepr_params['USE_SPARSE'],
                        temporal_context= prepr_params['TEMPORAL_CONTEXT'],
                        contrast_llim = prepr_params['CLIP_LOWERLIM'],
                        plot = prepr_params['PLOT_PREPROC'], 
                        cache = cache, 
                        from_cache = from_cache,
                        tilesize = prepr_params['TILESIZE'],
                        Sy = prepr_params['SY'], 
                        Sx = prepr_params['SX'])

        # unique attributes beyond Timelapse class. 
        self.notes = metadata['notes']
        self.dt = metadata['dt']
        self.pixelsize = metadata['pixelsize']
        self.structure_outputchannel_coo = metadata['target']
        
        self.goal_mask = np.load(mask_path.replace('mask1', 'mask1_goal'))
        self.goal_mask = np.pad(self.goal_mask, 50, mode='constant')
        self.start_mask = np.load(mask_path.replace('mask1', 'mask1_start'))
        self.start_mask = np.pad(self.start_mask, 50, mode='constant')

    def _load_metadata(self, filename):
        metadata = {}
        df = pd.read_csv(filename, index_col=0).infer_objects().iloc[:,0]
        metadata['target'] = tuple([int(coo) for coo in df.target[1:-1].split(', ')])
        metadata['notes'] = df.notes if str(df.notes) != 'nan' else ''
        metadata['design'] = f'D{df.design:0>2}'
        metadata['CLSM_area'] = f'G{df.CLSM_area:0>3}'
        metadata['dt'] = float(df["dt"])
        metadata['pixelsize'] = float(df.pixelsize)
        return metadata        

    