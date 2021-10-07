import sys
import pickle

import os
# os.chdir('../')
sys.path.append('./models/v1model')

from UnlabelledTimelapse import UnlabelledTimelapse
from PDMSDesignScreen import PDMSDesignScreen

from exp_parameters import load_parameters

from evaluation import setup_evaluation
from core_functionality import setup_model
from AxonDetections import AxonDetections

from plotting import draw_all

from exp_parameters import (
    get_default_parameters, 
    to_device_specifc_params,
    )

from config import RAW_DATA_DIR, SCREENING_DIR, OUTPUT_DIR
import config

from utils import set_seed

from utils import save_preproc_metrics
from plotting import plot_preprocessed_input_data

from core_functionality import setup_data


def main():
    exp6_name = 'v1Model_exp6_AxonDetClass'
    parameters = load_parameters(exp6_name, 'run17')
    parameters = to_device_specifc_params(parameters, get_default_parameters())
    set_seed(parameters['SEED'])
    check_preproc = False
    use_cached = False

    name = 'D01_G001'
    
    if not use_cached:
        run = 'run17'
        epoch = 2000
        parameters['LOAD_MODEL'] = [exp6_name, run, epoch]
        model, _, _, _ = setup_model(parameters)

        directory = f'{SCREENING_DIR}/{name}/'

        imseq_fname = f'{name}_GFP_compr.deflate.tif'
        mask_fname = f'{name}_Transmission_compr.deflate_mask1.npy'
        metadata_fname = f'{name}_Transmission_compr.deflate_metadata.csv'
        timelapse = UnlabelledTimelapse(config.RAW_DATA_DIR, imseq_fname, 
                                        mask_fname, metadata_fname, parameters, 
                                        from_cache=config.SCREENING_DIR)

        if check_preproc:
            train_data, _ = setup_data(parameters)
            preproc_file = save_preproc_metrics(directory, timelapse, train_data)
            plot_preprocessed_input_data(preproc_file, dest_dir=directory, show=True)

        axon_detections = AxonDetections(model, timelapse, parameters, f'{directory}/axon_dets')
        axon_detections.assign_ids(cache='from')
        
        fname = f'{timelapse.name}_E:{epoch}_timepoint:---|{timelapse.sizet}'
        draw_all(axon_detections, fname, dest_dir=directory, 
                 notes=parameters["NOTES"], show=False, animated=True, hide_det2=False, 
                 color_det1_ids=True, use_IDed_dets=True, )

        screen = PDMSDesignScreen(timelapse, directory, axon_detections, cache_target_distances='to')
        pickle.dump(open(f'{directory}/{name}_screen.pkl', 'rb'), screen)
    
    else:
        screen = pickle.load(open(f'{directory}/{name}_screen.pkl', 'wb'))
    
    screen.compute_metrics(show=True)
    

    




if __name__ == '__main__':
    main()














# import sys
# import os
# # os.chdir('../')
# sys.path.append('./models/v1model')

# from UnlabelledTimelapse import UnlabelledTimelapse
# from exp_parameters import load_parameters

# from evaluation import setup_evaluation
# from core_functionality import setup_model
# from AxonDetections import AxonDetections

# from plotting import draw_all

# from exp_parameters import (
#     get_default_parameters, 
#     to_device_specifc_params,
#     )
# from config import RAW_DATA_DIR, SCREENING_DIR


# def main():
#     exp6_name = 'v1Model_exp6_AxonDetClass'
#     epoch = 2000
#     run = 'run17'
#     parameters = load_parameters(exp6_name, run)

#     # timelapse_processed_dir = '/run/media/loaloa/lbb_ssd/timelapse13_processed'
#     # timelapse_processed_dir = '/home/loaloa/Documents/timelapse13_processed'
#     name = 'D01_G001'
#     imseq_fname = f'{name}_GFP_compr.deflate.tif'
#     mask_fname = f'{name}_Transmission_compr.deflate_mask1.npy'
#     metadata_fname = f'{name}_Transmission_compr.deflate_metadata.csv'
#     timelapse = UnlabelledTimelapse(RAW_DATA_DIR, imseq_fname, 
#                                     mask_fname, metadata_fname, parameters, 
#                                     from_cache=SCREENING_DIR)
    
#     from utils import (
#         create_logging_dirs,
#         save_preproc_metrics,
#         save_checkpoint,
#         clean_rundirs,
#         set_seed,
#     )

#     parameters = to_device_specifc_params(parameters, get_default_parameters())
#     parameters['LOAD_MODEL'] = [exp6_name, run, epoch]
#     # params['DEVICE'] = 'cpu'

#     model, _, _, _ = setup_model(parameters)
        
#     directory = f'{SCREENING_DIR}/axon_detections'
#     axon_detections = AxonDetections(model, timelapse, parameters, directory)
#     axon_detections.assign_ids(cache='to')
    
#     os.makedirs(f'{RUN_DIR}/model_out', exist_ok=True)
#     fname = f'{timelapse.name}_E:{epoch}_timepoint:---|{timelapse.sizet}'
#     draw_all(axon_detections, fname, dest_dir=f'{RUN_DIR}/model_out', 
#                 notes=parameters["NOTES"], show=True, animated=True, hide_det2=False, 
#                 color_det1_ids=True, use_IDed_dets=True, )

    




# if __name__ == '__main__':
#     main()

