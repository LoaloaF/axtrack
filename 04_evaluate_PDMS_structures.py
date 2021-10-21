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

from plotting import draw_all

from exp_parameters import (
    get_default_parameters, 
    to_device_specifc_params,
    )

# from config import RAW_DATA_DIR, SCREENING_DIR, OUTPUT_DIR
import config

from utils import set_seed

from utils import save_preproc_metrics
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
    set_seed(parameters['SEED'])
    parameters['LOAD_MODEL'] = [exp_name, run, epoch]
    model, _, _, _ = setup_model(parameters)
    if print_params:
        print(params2text(parameters))
    return model

def get_data_standardization_scaler(parameters):
    train_data, _ = setup_data(parameters)
    return train_data.stnd_scaler, train_data

def get_structure_screen_names(names_subset):
    segmted_timelapses = glob.glob(config.RAW_INFERENCE_DATA_DIR+'/*mask1.npy')
    names = sorted([f[f.rfind('/')+1:f.rfind('/')+14] for f in segmted_timelapses])
    if names_subset:
        names = [names[i] for i in names_subset]

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
        plot_preprocessed_input_data(preproc_file, dest_dir=directory, show=False)

    axon_detections = AxonDetections(model, timelapse, parameters, f'{directory}/axon_dets')
    cache = 'to' if not use_cached_det_astar_paths else 'from'
    axon_detections.assign_ids(cache=cache)
    
    if make_tracking_video:
        fname = f'timepoint:---of{timelapse.sizet}'
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
        
        fname = f'timepoint:---of{timelapse.sizet}'
        draw_all(axon_detections, fname, dest_dir=dest_dir, which_axons=target_axons+cg_axons,
                 dt=timelapse.dt, pixelsize=timelapse.pixelsize, hide_det2=True, 
                 color_det1_ids=True, use_IDed_dets=True, anim_fname_postfix='_special_axons',
                 **tracking_video_kwargs)
    return screen
    

















def main():
    # general setup 
    structure_names_subset = None
    # structure_names_subset = range(26,28)
    # structure_names_subset = (3,4,5,6,9,12,13,15,16)
    # structure_names_subset = [0,25]
    check_preproc = True
    make_tracking_video = True
    make_tracking_video_finaldest = True
    tracking_video_kwargs = {'draw_axons':None, 'show':False, 'animated':True, 
                             'draw_scalebar':'top_right', }#'t_y_x_slice':[(0,20), None, None]}
    use_cached_datasets = True
    use_cached_det_astar_paths = True
    use_cached_target_astar_paths = False
    use_cached_screen = True
    num_workers = 10
    device = 'cuda:0'
    exp_name, run, epoch = 'v1Model_exp6_AxonDetClass', 'run39', 3000
    
    if not use_cached_screen:
        parameters = get_params(exp_name, run, num_workers, device)
        model = get_model(exp_name, run, epoch, parameters)
        stnd_scaler, train_data = get_data_standardization_scaler(parameters)
    
    all_screens = []
    structure_screen_names = get_structure_screen_names(structure_names_subset)
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
            if screen.dir.startswith('/srv/beegfs-data/') and \
                config.BASE_DIR.startswith('/home/loaloa'):
                screen.dir = f'{config.SCREENING_DIR}/{structure_name}/screen/'
        
            # remove me later when caching new screening objects
            screen.hacky_adhoc_param_setter()
            # pickle.dump(screen, open(f'{directory}/{structure_name}_screen.pkl', 'wb'))
        
        all_screens.append(screen)

    
    screen = PDMSDesignScreen(all_screens, f'{config.SCREENING_DIR}/all_screens')
    # screen.show_masks()
    
    # screen.sss_ID_lifetime(symlink_results=True, plot_kwargs={'show':False})
    
    # screen.sss_target_distance_over_time(symlink_results=False, plot_kwargs={'show':True})
    # screen.sss_target_distance_over_time(symlink_results=False, plot_kwargs={'show':True, 'subtr_init_dist':False})
    
    # screen.sss_target_distance_over_time(symlink_results=False, plot_kwargs={'show':True, 'draw_until_t':200, 'subtr_init_dist':False})
    # screen.sss_target_distance_over_time(symlink_results=False, plot_kwargs={'show':True, 'draw_until_t':140, 'subtr_init_dist':True})
    

    
    # screen.cs_naxons(DIV_range=(2.5,6), plot_kwargs={'show':False, 'fname_postfix':'_MCF_5-550'})
    # screen.cs_naxons(DIV_range=None, plot_kwargs={'show':False, 'fname_postfix':'_MCF_5-550'})

    # screen.cs_axon_growthspeed(DIV_range=(2.5,6), percentile=None, plot_kwargs={'show':True, 'fname_postfix':'20percentile_MCF_5-550'})
    # screen.cs_axon_growthspeed(DIV_range=None, percentile=None, absolute=False, plot_kwargs={'show':False, 'fname_postfix':'non-absolute'})

    screen.cs_axon_growth_direction(plot_kwargs={'show':False, 'fname_postfix':'_min500um_prop'})

    # screen.cs_axon_destinations(plot_kwargs={'show':False})#DIV_range=(5.8,6.57))




if __name__ == '__main__':
    main()
    print('\n\n\nExit(0)')