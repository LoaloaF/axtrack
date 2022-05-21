try:
    import axtrack
except ImportError:
    import sys
    import os
    sys.path.append(f'{os.path.abspath(os.path.dirname(__file__))}/../../')
    import axtrack

# first setup the directories for reading and writing data
inference_data_dir = f'{axtrack.PKG_DIR}/examples/'
dest_dir = inference_data_dir
imseq_fname = 'example_timelapse.tif'
# mask filename may also be None
# mask_fname = None
mask_fname = 'example_timelapse_mask.npy'
parameters, model, stnd_scaler = axtrack.setup_inference(dest_dir)

# adjust because example data is quite short (20 frames)
parameters.update({'MCF_MAX_FLOW': 140})    # reduce from 450 to 140

# create the input data object
use_cached_datasets = 'to'
check_preproc = True
# metadata specific for input data
input_metadata = {'dt': 31, 'pixelsize':.62, 'intensity_offset': 121,
                  'clip_intensity': 55, 'incubation_time': 52, 
                  'name':'example_timelapse'}
timelapse = axtrack.prepare_input_data(imseq_fname, parameters, dest_dir, inference_data_dir,
                            stnd_scaler, mask_fname=mask_fname, use_cached_datasets=use_cached_datasets, 
                            check_preproc=check_preproc, input_metadata=input_metadata)

# get axon detections
cache_detections = 'to'
astar_paths_cache = 'to'
assigedIDs_cache = 'to'
axon_dets = axtrack.inference(timelapse, model, dest_dir, parameters, 
                                detections_cache=cache_detections,
                                astar_paths_cache=astar_paths_cache,
                                assigedIDs_cache=assigedIDs_cache)

# print out the detections as a pd.DataFrame and make a final video 
dets = axon_dets.IDed_dets_all
print(dets)
axtrack.visualize_inference(axon_dets, which_dets='IDed', draw_scalebar=False, 
                            animated=True, show=False, draw_brightened_bg=True)