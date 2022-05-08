try:
    import axtrack
except ImportError:
    import sys
    import os
    sys.path.append(f'{os.path.abspath(os.path.dirname(__file__))}/../../')
    import axtrack

imseq_fname = 'timelapse42.tif'
inference_data_dir = '/run/media/loaloa/lbbSSD/test_axtrack_inference'
dest_dir = inference_data_dir
mask_fname = None
parameters, model, stnd_scaler = axtrack.setup_inference(dest_dir)

# parameters.update({'MCF_MIN_FLOW': 20})
parameters.update({'MCF_MAX_FLOW': 140})

use_cached_datasets = 'from'
check_preproc = False
input_metadata = {'dt': 31, 'pixelsize':.62, 'intensity_offset': 0, 
                    'clip_intensity': 45, 'incubation_time': 52, 'name':'mytl42'}
timelapse = axtrack.prepare_input_data(imseq_fname, parameters, dest_dir, inference_data_dir,
                            stnd_scaler, mask_fname=mask_fname, use_cached_datasets=use_cached_datasets, 
                            check_preproc=check_preproc, input_metadata=input_metadata)

cache_detections = 'from'
astar_paths_cache = 'from'
assigedIDs_cache = 'from'
axon_dets = axtrack.inference(timelapse, model, dest_dir, parameters, 
                                detections_cache=cache_detections,
                                astar_paths_cache=astar_paths_cache,
                                assigedIDs_cache=assigedIDs_cache)

dets = axon_dets.IDed_dets_all
print(dets)

axtrack.visualize_inference(axon_dets, which_dets='IDed', draw_scalebar=False, )