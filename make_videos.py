from evaluate_PDMS_structures import get_PDMSscreen
from utils import turn_tex


def get_video_args():
    all_videos_kwargs = []
    # raw video showcase
    tracking_video_kwargs = {'draw_axons':None, 'show':False, 'animated':True, 
                             'draw_grid':False, 'hide_det1':True, 
                             'anim_fname_postfix':'_unlbled_outgrowth',
                             'draw_scalebar':'top_right', 
                             't_y_x_slice':[(0,140), (250, 1900), (2100, 3850)]}
    # all_videos_kwargs.append(tracking_video_kwargs)

    # detected video showcase
    tracking_video_kwargs = tracking_video_kwargs.copy()
    tracking_video_kwargs['draw_grid'] = True
    tracking_video_kwargs['hide_det1'] = False
    tracking_video_kwargs['anim_fname_postfix'] = '_lbled_outgrowth'
    # all_videos_kwargs.append(tracking_video_kwargs)
    
    # Design 20
    tracking_video_kwargs = tracking_video_kwargs.copy()
    tracking_video_kwargs['anim_fname_postfix'] = '_showcase'
    tracking_video_kwargs['t_y_x_slice'] = [None, None, None]
    # all_videos_kwargs.append(tracking_video_kwargs)

    # # Axon 064 without distance drawn
    # tracking_video_kwargs = {'draw_axons':None, 'show':False, 'animated':True, 
    #                          'draw_grid':False, 'hide_det1':False,
    #                          'which_axons': ['Axon_064'], 
    #                          'anim_fname_postfix':'_Ax064_nodist',
    #                          'draw_trg_distances': False,
    #                          'draw_scalebar':False, 
    #                          'fps':4, 
    #                          'dpi':300, 
    #                          't_y_x_slice':[(80,140),(550, 1200), (2250, 3100)]}
    # all_videos_kwargs.append(tracking_video_kwargs)

    # # # Axon 064 with distance drawn
    # tracking_video_kwargs = tracking_video_kwargs.copy()
    # tracking_video_kwargs = {'draw_axons':None, 'show':False, 'animated':True, 
    #                          'draw_grid':True, 'hide_det1':False,
    #                          'which_axons': ['Axon_064'], 
    #                          'anim_fname_postfix':'_Ax064_dist',
    #                          'draw_trg_distances': False,
    #                          'draw_scalebar':False, 
    #                          'fps':4, 
    #                          'dpi':300, 
    #                          't_y_x_slice':[(80,140),(0, 1250), (2250, 3100)]}
    # all_videos_kwargs.append(tracking_video_kwargs)
    
    # # Axon 064 with inserted plot drawn
    tracking_video_kwargs = {'draw_axons':None, 'show':False, 'animated':True, 
                             'draw_grid':False, 'hide_det1':False,
                             'which_axons': ['Axon_064'], 
                             'anim_fname_postfix':'_Ax064_dist',
                             'draw_trg_distances': True,
                             'fps':4, 
                             'dpi':300, 
                             'draw_scalebar':False, 
                             'insert_distance_plot':True,
                             't_y_x_slice':[(80,160),(0, 1250), (2250, 3100)]}
    all_videos_kwargs.append(tracking_video_kwargs)
    
    # # Only Axon 64 withut drawn distance
    # structure_names_subset = [0]
    # tracking_video_kwargs = {'draw_axons':None, 'show':False, 'animated':True, 
    #                          'draw_grid':True, 'hide_det1':False, 
    #                          'anim_fname_postfix':'_showcase',
    #                          'draw_scalebar':'top_right', 
    #                          't_y_x_slice':[(140,210), (300, 2100), (0,1800)]}
    # all_videos_kwargs.append(tracking_video_kwargs)

    # # Only Axon 64 with drawn distance
    # structure_names_subset = [5]
    # tracking_video_kwargs = {'draw_axons':None, 'show':False, 'animated':True, 
    #                          'draw_grid':True, 'hide_det1':False, 
    #                          'anim_fname_postfix':'_showcase',
    #                          'draw_scalebar':'top_right', 
    #                          't_y_x_slice':[(0,70), (0, 1800), (1900,3600)]}
    # all_videos_kwargs.append(tracking_video_kwargs)
    
    return all_videos_kwargs

def main():
    turn_tex('on')
    # device run settings, which model used for inference
    num_workers = 28
    device = 'cuda:4'
    exp_name, run, epoch = 'v1Model_exp7_final', 'run36', 400

    # Only design 12, timelapse13, G013
    structure_names_subset = [19]
    exclude_names = []

    use_cached_datasets = True
    use_cached_det_astar_paths = True
    use_cached_target_astar_paths = True
    # Inefficent, but for every video one needs to create a new screen...
    use_cached_screen = False
    
    # for a new screen which additional functions should be executed 
    check_preproc = False
    make_tracking_video = True
    make_tracking_video_finaldest = False
    make_dist_over_time_video = False
    # where to save the analysis output
    PDMS_screen_dest_dir = 'presentation_videos'

    all_videos_kwargs = get_video_args()
    for i, arg_set in enumerate(all_videos_kwargs):
        screen = get_PDMSscreen(num_workers, 
                                device, 
                                exp_name, 
                                run, 
                                epoch, 
                                structure_names_subset, 
                                exclude_names, 
                                use_cached_datasets, 
                                use_cached_det_astar_paths, 
                                use_cached_target_astar_paths, 
                                use_cached_screen, 
                                check_preproc, 
                                make_tracking_video, 
                                make_tracking_video_finaldest, 
                                arg_set, 
                                make_dist_over_time_video,
                                PDMS_screen_dest_dir
        )

if __name__ == '__main__':
    main()
    print('\n\n\nExit(0)')