import sys
import pickle
import glob


import os
# os.chdir('../')
sys.path.append('./models/v1model')

from UnlabelledTimelapse import UnlabelledTimelapse
from StructureScreen import StructureScreen

from exp_parameters import load_parameters

from evaluation import setup_evaluation
from core_functionality import setup_model
from AxonDetections import AxonDetections

import warnings
warnings.filterwarnings("ignore")

from plotting import draw_all

from exp_parameters import (
    get_default_parameters, 
    to_device_specifc_params,
    )

# from config import RAW_DATA_DIR, SCREENING_DIR, OUTPUT_DIR
import config

from utils import set_seed
from utils import turn_tex
turn_tex('on')

from utils import save_preproc_metrics
from utils import set_seed
from plotting import plot_preprocessed_input_data
from exp_parameters import params2text

from PDMSDesignScreen import PDMSDesignScreen
from core_functionality import setup_data


def get_params(exp_name, run, num_workers=None, device=None):
    parameters = load_parameters(exp_name, run)
    parameters = to_device_specifc_params(parameters, get_default_parameters(), 
                                          from_cache=config.OUTPUT_DIR)
    if num_workers:
        parameters['NUM_WORKERS'] = num_workers
    if device:
        parameters['DEVICE'] = device
    return parameters

def get_model(exp_name, run, epoch, parameters, print_params=True):
    parameters['LOAD_MODEL'] = [exp_name, run, epoch]
    model, _, _, _ = setup_model(parameters)
    if print_params:
        print(params2text(parameters))
    return model

def get_data_standardization_scaler(parameters):
    if 'OFFSET' not in parameters:
        parameters['OFFSET'] = None
    train_data, _ = setup_data(parameters)
    return train_data.stnd_scaler, train_data

def get_structure_screen_names(names_subset, exclude_names):
    segmted_timelapses = glob.glob(config.RAW_INFERENCE_DATA_DIR+'/*mask1.npy')
    names = sorted([f[f.rfind('/')+1:f.rfind('/')+14] for f in segmted_timelapses])
    if names_subset:
        names = [names[i] for i in names_subset]
    if exclude_names:
        names = [name for name in names if name not in exclude_names]

    print(f'{len(names)} structure timelapses found:\n{names}', flush=True)
    return names


def create_structure_screen(structure_name, directory, model, parameters, 
                            train_data, check_preproc,
                            use_cached_datasets, use_cached_det_astar_paths, 
                            use_cached_target_astar_paths, make_tracking_video,
                            make_tracking_video_finaldest, tracking_video_kwargs, 
                            cache_screen):
    
    imseq_fname = f'{structure_name}_GFP_compr.deflate.tif'
    mask_fname = f'{structure_name}_Transmission_compr.deflate_mask1.npy'
    metadata_fname = f'{structure_name}_Transmission_compr.deflate_metadata.csv'
    from_cache = None if not use_cached_datasets else directory
    timelapse = UnlabelledTimelapse(config.RAW_INFERENCE_DATA_DIR, imseq_fname, 
                                    mask_fname, metadata_fname, parameters, 
                                    standardize=train_data.stnd_scaler,
                                    from_cache=from_cache, cache=directory)

    if check_preproc:
        preproc_file = save_preproc_metrics(directory, timelapse, train_data)
        plot_preprocessed_input_data(preproc_file, dest_dir=directory, show=False, 
                                     fname_prefix=structure_name)

    axon_detections = AxonDetections(model, timelapse, parameters, f'{directory}/axon_dets')
    cache = 'to' if not use_cached_det_astar_paths else 'from'
    axon_detections.assign_ids(cache=cache)
    
    if make_tracking_video:
        fname = f'timepoint---of{timelapse.sizet}'
        dest_dir = f'{directory}/det_videos'
        os.makedirs(dest_dir, exist_ok=True)
        draw_all(axon_detections, fname, dest_dir=dest_dir, 
                 dt=timelapse.dt, pixelsize=timelapse.pixelsize, hide_det2=True,
                 color_det1_ids=True, use_IDed_dets=True, 
                 **tracking_video_kwargs)

    # PDMS structure screen object composed of axon detections and basic timelapse properties
    cache = 'to' if not use_cached_target_astar_paths else 'from'
    dest_dir = f'{directory}/screen/'
    screen = StructureScreen(timelapse, dest_dir, axon_detections, cache_target_distances=cache)
    if cache_screen:
        pickle.dump(screen, open(f'{directory}/{structure_name}_screen.pkl', 'wb'))
        print(f'{structure_name} screen successfully saved.')
    
    if make_tracking_video_finaldest:
        dists = screen.get_distances()
        target_axons, cg_axons = screen.detect_axons_final_location(dists)
        
        fname = f'timepoint---of{timelapse.sizet}'
        draw_all(axon_detections, fname, dest_dir=dest_dir, which_axons=target_axons+cg_axons,
                 dt=timelapse.dt, pixelsize=timelapse.pixelsize, hide_det2=True, 
                 color_det1_ids=True, use_IDed_dets=True, anim_fname_postfix='_special_axons',
                 **tracking_video_kwargs)
    return screen
    
def get_PDMSscreen(num_workers, device, exp_name, run, epoch, 
                   structure_names_subset, exclude_names, use_cached_datasets, 
                   use_cached_det_astar_paths, use_cached_target_astar_paths, 
                   use_cached_screen, check_preproc, make_tracking_video, 
                   make_tracking_video_finaldest, tracking_video_kwargs, 
                   PDMS_screen_dest_dir):

    parameters = get_params(exp_name, run, num_workers, device)
    set_seed(parameters['SEED'])
    if not use_cached_screen:
        model = get_model(exp_name, run, epoch, parameters)
        stnd_scaler, train_data = get_data_standardization_scaler(parameters)
    
    all_screens = []
    structure_screen_names = get_structure_screen_names(structure_names_subset, exclude_names)
    for i, structure_name in enumerate(structure_screen_names):
        print(f'Processing structure screen {structure_name} now ({i}/'
              f'{len(structure_screen_names)})...', flush=True, end='')
        directory = f'{config.SCREENING_DIR}/{structure_name}/'
        os.makedirs(directory, exist_ok=True)
        
        if not use_cached_screen:
            screen = create_structure_screen(structure_name, directory, model, 
                                             parameters, train_data, 
                                             check_preproc, 
                                             use_cached_datasets, 
                                             use_cached_det_astar_paths, 
                                             use_cached_target_astar_paths, 
                                             make_tracking_video,
                                             make_tracking_video_finaldest,
                                             tracking_video_kwargs,
                                             cache_screen=True)
        else:
            try:
                screen = pickle.load(open(f'{directory}/{structure_name}_screen.pkl', 'rb'))
            except FileNotFoundError:
                print(f'No cached screen found. Skipping.')
                continue
            
            print('Screen loaded from cache.')
            screen.dir = f'{config.SCREENING_DIR}/{structure_name}/screen/'

            # if screen.dir.startswith('/srv/beegfs-data/') and \
            #     config.BASE_DIR.startswith('/home/loaloa'):
            #     print('in')
            #     screen.dir = f'{config.SCREENING_DIR}/{structure_name}/screen/'
        
            # remove me later when caching new screening objects
            screen.hacky_adhoc_param_setter()
            # pickle.dump(screen, open(f'{directory}/{structure_name}_screen.pkl', 'wb'))
        all_screens.append(screen)
    return PDMSDesignScreen(all_screens, f'{config.SCREENING_DIR}/{PDMS_screen_dest_dir}')







def validate_screens(screen, symlink_results, plot_kwargs):

    screen.sss_validate_masks(symlink_results=symlink_results, plot_kwargs=plot_kwargs)
    
    screen.sss_validate_IDlifetime(symlink_results=symlink_results, plot_kwargs=plot_kwargs)
    









def main():
    # device run settings, which model used for inference
    num_workers = 28
    device = 'cuda:0'
    exp_name, run, epoch = 'v1Model_exp7_final', 'run36', 400

    # which timelapses to include in analysis, None = all
    structure_names_subset = None
    # structure_names_subset = range(34)  # all timelapse13 
    # structure_names_subset = range(34, 75)  # all timelapse14
    # structure_names_subset = range(25,45)
    exclude_names = ['tl13_D04_G004', 'tl13_D19_G035', # training data
                     'tl14_D02_G002', 'tl14_D02_G014', 'tl14_D03_G015', # bad detections
                     'tl13_D20_G036'] # only one exists for D20.... changed on 2nd wafer....
    # exclude_names.extend(['tl13_D02_G002',  # not in both datasets
    #                       'tl13_D02_G006',
    #                       'tl13_D03_G003',
    #                       'tl13_D03_G007',
    #                       'tl14_D04_G041',
    #                       'tl14_D04_G042',
    #                       'tl14_D21_G004',
    #                       'tl14_D21_G016',
    #                       'tl14_D08_G008',
    #                       'tl14_D08_G012',
    #                       ])
    # exclude_names = []
    # recompute things or load from disk
    use_cached_datasets = True
    use_cached_det_astar_paths = True
    use_cached_target_astar_paths = True
    use_cached_screen = True
    
    # for a new screen which additional functions should be executed 
    check_preproc = True
    make_tracking_video = True
    make_tracking_video_finaldest = True
    tracking_video_kwargs = {'draw_axons':None, 'show':False, 'animated':True, 
                             'draw_scalebar':'top_right', }#'t_y_x_slice':[(0,20), None, None]}
    
    # where to save the analysis output
    # PDMS_screen_dest_dir = 'tl13_results'
    # PDMS_screen_dest_dir = '../ETZ_drive/biohybrid-signal-p/PDMS_structure_screen_v2/all_results_v6'
    PDMS_screen_dest_dir = 'all_results_figure1'
    PDMS_screen_dest_dir = 'all_results_figure2'
    PDMS_screen_dest_dir = 'all_results_figure3'

    
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
                            tracking_video_kwargs, 
                            PDMS_screen_dest_dir
    )

    rank = False
    show = True if config.BASE_DIR.startswith('/home/loaloa/') else False # on server, show is always False
    # show = False
    symlink_results = False
    DIV_range = (3.5,6.5)
    plot_kwargs = {'show': show}
    
    """Validate the single screens, checking the masks and ID lifetime"""
    # validate_screens(screen, symlink_results, plot_kwargs)


    """Results figure 1 `Information obtained through tracking` : 
    Take a first look at single-structure distance over time plots, 
    base line subtr and not"""
    # screen.sss_target_distance_timeline(symlink_results=symlink_results, plot_kwargs=plot_kwargs)
    # _plot_kwargs = {**plot_kwargs, 'subtr_init_dist': False, 'fname_postfix':'_nodelta'}
    # screen.sss_target_distance_timeline(symlink_results=symlink_results, plot_kwargs=_plot_kwargs)

    """Compare the two datasets tl13, and 1l14 wrt growth speed variance & dist to target var"""
    # # structure_names_subset = range(34)  # all timelapse13        |   <-- uncomment one of the two above (line215)
    # # structure_names_subset = range(34, 75)  # all timelapse14    |            and run
    # _plot_kwargs = {**plot_kwargs, 'fname_postfix': '_tl13'}
    # screen.cs_target_distance_timeline(speed=False, plot_kwargs=_plot_kwargs)
    # screen.cs_target_distance_timeline(speed=True, plot_kwargs=_plot_kwargs)
    
    """Quantify the above the two datasets tl13, and 1l14 wrt growth speed variance & dist to target var"""
    # _plot_kwargs = {**plot_kwargs, 'split_by':'timelapse', 'custom_colors':(config.DARK_GRAY, config.DARK_GRAY)}
    # screen.cs_axon_growth(DIV_range=DIV_range, which_metric='speed', plot_kwargs=_plot_kwargs)
    # screen.cs_axon_growth(DIV_range=DIV_range, which_metric='growth_direction', plot_kwargs=_plot_kwargs)






    """Results figure 2 `Growth speed across designs` : 
    Take a second look at growth speed with highlighting specific feature effects
    """
    design_subsets = {
        'channel width': (2,3,4),
        'n 2-joints': (9,10,11,12,5,8,6,7),
        'n rescue loops': (2,9,10,11,12),
        '2-joint placement': (5,6,7,8),
        'rescue loop design': (12, 17),
        'use spiky tracks': (12, 18),
        'final lane design': (1, 12,19,20),
        '2-joint design': (12,13,14,15,16),
    }
    """all designs"""
    screen.cs_axon_growth(DIV_range=DIV_range, which_metric='speed', plot_kwargs=plot_kwargs)
    screen.cs_axon_growth_counted(DIV_range=DIV_range, which_metric='n_axons', plot_kwargs={**plot_kwargs, })
    

    """subset of designs specifically differing in a design feature"""
    plot_kwargs['order'] = rank
    for design_feature in  design_subsets.keys():
        plot_kwargs['split_by'] = design_feature
        plot_kwargs['design_subset'] = design_subsets[design_feature]
        plot_kwargs['mixed_colorbars'] = True
        
        # growth speed
        screen.cs_axon_growth(DIV_range=DIV_range, which_metric='speed', 
                              plot_kwargs=plot_kwargs)
        # n axons
        screen.cs_axon_growth_counted(DIV_range=DIV_range, which_metric='n_axons', 
                                      plot_kwargs={**plot_kwargs, 'draw_bar_singles':True})
    
    
    
    




    """Results figure 3 `directionality` : 
    Take a third look. Now the core directionality. counted v1, v2, and distribution
    """
    design_subsets = {
        'channel width': (2,3,4),
        'n 2-joints': (9,10,11,12,5,8,6,7),
        'n rescue loops': (2,9,10,11,12),
        '2-joint placement': (5,6,7,8),
        'rescue loop design': (12, 17),
        'use spiky tracks': (12, 18),
        'final lane design': (1, 12,19,20),
        '2-joint design': (12,13,14,15,16),
    }
    """all designs"""
    screen.cs_axon_growth(DIV_range=DIV_range, which_metric='speed', plot_kwargs=plot_kwargs)
    screen.cs_axon_growth_counted(DIV_range=DIV_range, which_metric='n_axons', plot_kwargs={**plot_kwargs, })
    

    """subset of designs specifically differing in a design feature"""
    plot_kwargs['order'] = rank
    for design_feature in  design_subsets.keys():
        plot_kwargs['split_by'] = design_feature
        plot_kwargs['design_subset'] = design_subsets[design_feature]
        plot_kwargs['mixed_colorbars'] = True
        
        # growth speed
        screen.cs_axon_growth(DIV_range=DIV_range, which_metric='speed', 
                              plot_kwargs=plot_kwargs)
        # n axons
        screen.cs_axon_growth_counted(DIV_range=DIV_range, which_metric='n_axons', 
                                      plot_kwargs={**plot_kwargs, 'draw_bar_singles':True})
    
    



    """Bonus: Growth direction distribution """
    # screen.cs_axon_growth(DIV_range=DIV_range, which_metric='growth_direction', plot_kwargs=plot_kwargs)
    # screen.cs_axon_growth(DIV_range=DIV_range, which_metric='growth_direction', plot_kwargs=plot_kwargs)
    
    







    """Results figure 2: structure,- and feature wise comparison of directionality 
       based on reached target and reached neighbor"""
    # plot_kwargs['fname_postfix'] = 'all_designs'
    # screen.cs_axon_destinations(DIV_range=DIV_range, plot_kwargs=plot_kwargs)
    # plot_kwargs['design_subset'] = [0,1,2,3,4]
    # plot_kwargs['fname_postfix'] = '_group1'
    # screen.cs_axon_destinations(DIV_range=DIV_range, plot_kwargs=plot_kwargs)
    # plot_kwargs['design_subset'] = [2,5,6,7,8]
    # plot_kwargs['fname_postfix'] = '_group2'
    # screen.cs_axon_destinations(DIV_range=DIV_range, plot_kwargs=plot_kwargs)
    # plot_kwargs['design_subset'] = [2,9,10,11,12]
    # plot_kwargs['fname_postfix'] = '_group3'
    # screen.cs_axon_destinations(DIV_range=DIV_range, plot_kwargs=plot_kwargs)
    # plot_kwargs['fname_postfix'] = '_group4'
    # screen.cs_axon_destinations(DIV_range=DIV_range, plot_kwargs=plot_kwargs)
    # plot_kwargs['design_subset'] = None
    

    # design_subsets = {
    #     'channel width': (2,3,4),
    #     'n 2-joints': (9,10,11,12,5,8,6,7),
    #     'n rescue loops': (2,9,10,11,12),
    #     '2-joint placement': (5,6,7,8),
    #     'rescue loop design': (12, 17),
    #     'use spiky tracks': (12, 20),
    #     'final lane design': (1, 12,18,19),
    #     '2-joint design': (12,13,14,15,16),
    # }
    # plot_kwargs['order'] = 'rank'
    # plot_kwargs['draw_bar_singles'] = True
    # plot_kwargs['draw_naxons_counts'] = False
    # for design_feature in  ['timelapse'] + config.DESIGN_FEATURE_NAMES:
    #     plot_kwargs['split_by'] = design_feature
    #     if design_feature != 'timelapse':
    #         plot_kwargs['design_subset'] = design_subsets[design_feature]
        
    #     plot_kwargs['which_bars'] = 'metric'
    #     plot_kwargs['fname_postfix'] = f'_des_subset_{design_feature}'
    #     screen.cs_axon_destinations(DIV_range=DIV_range, plot_kwargs=plot_kwargs)
    #     plot_kwargs['which_bars'] = 'pos_metric'
    #     plot_kwargs['fname_postfix'] = f'_des_subset_{design_feature}_pos'
    #     screen.cs_axon_destinations(DIV_range=DIV_range, plot_kwargs=plot_kwargs)
    #     plot_kwargs['which_bars'] = 'neg_metric'
    #     plot_kwargs['fname_postfix'] = f'_des_subset_{design_feature}_neg'
    #     screen.cs_axon_destinations(DIV_range=DIV_range, plot_kwargs=plot_kwargs)
    




if __name__ == '__main__':
    main()
    print('\n\n\nExit(0)')