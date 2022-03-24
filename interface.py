import os
import sys

from .config import DEPLOYED_MODEL_DIR, DEFAULT_DEVICE, DEFAULT_NUM_WORKERS

from .machinelearning.utils import turn_tex, set_seed, get_data_standardization_scaler
from .machinelearning.exp_parameters import load_parameters, params2text
from .machinelearning.core_functionality import setup_model

from .machinelearning.UnlabelledTimelapse import UnlabelledTimelapse
from .machinelearning.AxonDetections import AxonDetections

def _get_params(num_workers=None, device=None):
    parameters = load_parameters(exp_name=None, run=None, from_directory=DEPLOYED_MODEL_DIR)
    # parameters = to_device_specifc_params(parameters, get_default_parameters(), 
    #                                       from_cache=config.OUTPUT_DIR)
    if num_workers:
        parameters['NUM_WORKERS'] = num_workers
    if device:
        parameters['DEVICE'] = device
    return parameters

def _get_model(parameters, print_params=True):
    parameters['LOAD_MODEL'] = DEPLOYED_MODEL_DIR
    model, _, _, _ = setup_model(parameters)
    if print_params:
        print(params2text(parameters))
    return model

def _get_train_data(parameters):
    return setup_data(parameters, skip_test=True)[0]

def setup_inference(dest_dir, num_workers=DEFAULT_NUM_WORKERS, device=DEFAULT_DEVICE):
    parameters = _get_params(num_workers, device)
    set_seed(parameters['SEED'])
    turn_tex('on')
    model = _get_model(parameters)
    stnd_scaler = get_data_standardization_scaler(f'{DEPLOYED_MODEL_DIR}/train_stnd_scaler.pkl')
    os.makedirs(dest_dir, exist_ok=True)
    return parameters, model, stnd_scaler

def prepare_input_data(imseq_fname, mask_fname, metadata_fname, parameters, 
                       dest_dir, stnd_scaler, use_cached_datasets, check_preproc,
                       inference_data_dir):
    cache_dir = None if not use_cached_datasets else inference_data_dir
    timelapse = UnlabelledTimelapse(inference_data_dir, imseq_fname, 
                                    mask_fname, metadata_fname, parameters, 
                                    standardize=stnd_scaler, from_cache=cache_dir, 
                                    cache=inference_data_dir)
    timelapse.structure_outputchannel_coo = (0,3500)
    if check_preproc:
        preproc_file = save_preproc_metrics(dest_dir, timelapse, _get_train_data(parameters))
        plot_preprocessed_input_data(preproc_file, dest_dir=dest_dir, show=True, 
                                     fname_prefix=structure_name)
    print(timelapse.dt)
    return timelapse

def inference(timelapse, model, dest_dir, parameters, use_cached_det_astar_paths):
    axon_detections = AxonDetections(model, timelapse, parameters, f'{dest_dir}/axon_dets')
    cache = 'to' if not use_cached_det_astar_paths else 'from'
    axon_detections.assign_ids(astar_paths_cache=cache)
    return axon_detections

# def visualize_inference(timelapse, axon_detections, dest_dir, 
#                         use_cached_target_astar_paths, make_tracking_video,
#                         make_tracking_video_finaldest, tracking_video_kwargs, 
#                         make_dist_over_time_video,
#                         cache_screen):

#     # PDMS structure screen object composed of axon detections and basic timelapse properties
#     cache = 'to' if not use_cached_target_astar_paths else 'from'
#     dest_dir = f'{dest_dir}/screen/'
#     screen = StructureScreen(timelapse, dest_dir, axon_detections, cache_target_distances=cache)
    
#     if make_dist_over_time_video:
#         os.makedirs(f'{screen.dir}/target_distance_frames/', exist_ok=True)
#         plot_target_distance_over_time_video(screen, subtr_init_dist=True)

#     if cache_screen:
#         import pickle
#         pickle.dump(screen, open(f'{dest_dir}/{structure_name}_screen.pkl', 'wb'))
#         print(f'{structure_name} screen successfully saved.')
    
#     if make_tracking_video:
#         fname = f'timepoint---of{timelapse.sizet}'
#         dest_dir = f'{dest_dir}/det_videos'
#         os.makedirs(dest_dir, exist_ok=True)

#         if tracking_video_kwargs.get('insert_distance_plot'):
#             tracking_video_kwargs['insert_distance_plot'] = screen
#             if tracking_video_kwargs.get('which_axons'):
#                 screen.set_axon_subset(tracking_video_kwargs.get('which_axons'))

#         draw_all(axon_detections, fname, dest_dir=dest_dir, 
#                  dt=timelapse.dt, pixelsize=timelapse.pixelsize, hide_det2=True,
#                  color_det1_ids=True, use_IDed_dets=True, 
#                  **tracking_video_kwargs)

#     if make_tracking_video_finaldest:
#         dists = screen.get_distances()
#         target_axons, cg_axons, stag_axons = screen.detect_axons_final_location(dists)
        
#         fname = f'timepoint---of{timelapse.sizet}'
#         draw_all(axon_detections, fname, dest_dir=dest_dir, which_axons=target_axons+cg_axons,
#                  dt=timelapse.dt, pixelsize=timelapse.pixelsize, hide_det2=True, 
#                  color_det1_ids=True, use_IDed_dets=True, anim_fname_postfix='_special_axons',
#                  **tracking_video_kwargs)
#     return screen

def main():
    basename = 'timel2_Des101_G1'
    imseq_fname = f'training_timelapse.tif'
    mask_fname = None
    metadata_fname = f'{basename}_metadata.csv'

    # RAW_INFERENCE_DATA_DIR = REMOTE_DATA_DIR + '/tl13_tl14_all'
    RAW_INFERENCE_DATA_DIR = DEPLOYED_MODEL_DIR + '/../../training_data_subs/' # locally for debugging/ testing
    dest_dir = RAW_INFERENCE_DATA_DIR+'_inference'

    use_cached_datasets = True
    check_preproc = False

    use_cached_det_astar_paths = True
    use_cached_target_astar_paths = True
    make_tracking_video = True
    make_tracking_video_finaldest = False
    make_dist_over_time_video = True

    
    parameters, model, stnd_scaler = setup_inference(dest_dir)

    timelapse = prepare_input_data(imseq_fname, mask_fname, metadata_fname, 
                                   parameters, dest_dir, stnd_scaler, 
                                   use_cached_datasets, check_preproc,
                                   RAW_INFERENCE_DATA_DIR)
    
    axon_detections = inference(timelapse, model, dest_dir, parameters, use_cached_det_astar_paths)

    # # tracking_video_kwargs = {'show':True, 'animated':True,
    # #                          't_y_x_slice':[None,(0, 1250), (2250, 3100)]}
    # tracking_video_kwargs = {'draw_axons':None, 'show':False, 'animated':True, 
    #                          'draw_grid':False, 'hide_det1':False,
    #                         #  'which_axons': ['Axon_001'], 
    #                          'anim_fname_postfix':'_Ax064_dist',
    #                          'draw_trg_distances': True,
    #                          'fps':4, 
    #                          'dpi':300, 
    #                          'draw_scalebar':False, 
    #                          'insert_distance_plot':True,
    #                          't_y_x_slice':[None,None, (2500, 3500)]}
    # visualize_inference(timelapse, axon_detections, dest_dir, 
    #                     use_cached_target_astar_paths, make_tracking_video,
    #                     make_tracking_video_finaldest, tracking_video_kwargs, 
    #                     make_dist_over_time_video,
    #                     cache_screen=True)



if __name__ == '__main__':
    main()
    print('\n\n\nExit(0)')