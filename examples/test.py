try:
    import axtrack
except ImportError:
    import sys
    import os
    sys.path.append(f'{os.path.dirname(__file__)}/../../')
    import axtrack

def main():
    imseq_fname = 'timelapse42.tif'
    inference_data_dir = '/run/media/loaloa/lbbSSD/test_axtrack_inference'
    # inference_data_dir = '/srv/beegfs-data/projects/biohybrid-signal-p/data/test_axtrack_inference'
    dest_dir = inference_data_dir
    mask_fname = None
    parameters, model, stnd_scaler = axtrack.setup_inference(dest_dir)
    
    parameters.update({'MCF_MIN_FLOW': 20})
    # parameters.update({'MCF_MAX_FLOW': 140})
    
    use_cached_datasets = 'from'
    check_preproc = False
    input_metadata = {'dt': 31, 'pixelsize':.62, 'intensity_offset': 0, 
                      'clip_intensity': 45, 'incubation_time': 52, 'name':'mytl42'}
    timelapse = axtrack.prepare_input_data(imseq_fname, parameters, dest_dir, inference_data_dir,
                               stnd_scaler, mask_fname=mask_fname, use_cached_datasets=use_cached_datasets, 
                               check_preproc=check_preproc, input_metadata=input_metadata)
    
    cache_detections = None
    astar_paths_cache = 'from'
    assigedIDs_cache = 'to'
    axon_dets = axtrack.inference(timelapse, model, dest_dir, parameters, 
                          cache_detections=cache_detections,
                          astar_paths_cache=astar_paths_cache,
                          assigedIDs_cache=assigedIDs_cache)
    # all_dets = axon_dets.get_frame_dets(which_dets='IDed', t=None)

    axtrack.visualize_inference(axon_dets, which_dets='confident')

















    # basename = 'timel2_Des101_G1'
    # imseq_fname = f'training_timelapse.tif'
    # mask_fname = None
    # metadata_fname = f'{basename}_metadata.csv'

    # # RAW_INFERENCE_DATA_DIR = REMOTE_DATA_DIR + '/tl13_tl14_all'
    # RAW_INFERENCE_DATA_DIR = DEPLOYED_MODEL_DIR + '/../../training_data_subs/' # locally for debugging/ testing
    # dest_dir = RAW_INFERENCE_DATA_DIR+'_inference'

    # use_cached_datasets = True
    # check_preproc = False

    # use_cached_det_astar_paths = True
    # MCF_MIN_FLOW
    # make_tracking_video = True
    # make_tracking_video_finaldest = False
    # make_dist_over_time_video = True

    # parameters, model, stnd_scaler = axtrack.setup_inference(dest_dir)
    # print(parameters)
    # print('wtf1')

    # print('wtf12')
    # print(parameters)
    # print('wtf13')
    # exit()
    # print('wtf14')
    # timelapse = axtrack.prepare_input_data(imseq_fname, mask_fname, metadata_fname, 
    #                                parameters, dest_dir, stnd_scaler, 
    #                                use_cached_datasets, check_preproc,
    #                                RAW_INFERENCE_DATA_DIR)
    
    # axon_detections = axtrack.inference(timelapse, model, dest_dir, parameters, use_cached_det_astar_paths)

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