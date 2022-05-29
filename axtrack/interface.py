import os

from .config import DEPLOYED_MODEL_DIR, DEFAULT_DEVICE, DEFAULT_NUM_WORKERS

from .exp_parameters import load_parameters, params2text
from .machinelearning.core_functionality import setup_model
from .ml_plotting import plot_preprocessed_input_data
from .video_plotting import draw_all
from .Timelapse import Timelapse
from .AxonDetections import AxonDetections
from .utils import (
        turn_tex, 
        set_seed, 
        get_data_standardization_scaler, 
        save_preproc_metrics)

import pandas as pd

def _get_params(num_workers=None, device=None):
    parameters = load_parameters(exp_name=None, run=None, 
                                 from_directory=DEPLOYED_MODEL_DIR)
    if num_workers:
        parameters['NUM_WORKERS'] = num_workers
    if device:
        parameters['DEVICE'] = device
    return parameters

def _get_model(parameters, print_params=False):
    parameters['LOAD_MODEL'] = DEPLOYED_MODEL_DIR
    model, _, _, _ = setup_model(parameters)
    if print_params:
        print(params2text(parameters))
    return model

def _get_train_data(parameters):
    return setup_data(parameters, skip_test=True)[0]

def setup_inference(dest_dir, print_params=False, num_workers=DEFAULT_NUM_WORKERS, 
                    device=DEFAULT_DEVICE):
    """
    Setup model and parameters for detecting axons. 

    Arguments
    ---------
    dest_dir: str
        The directory to which all model outputs will be saved. 
    
    print_params: bool
        Weather to print out the model parameters. Defaults to False.
    
    num_workers: int
        How many CPU cores to utilize. Defaults to config.DEFAULT_NUM_WORKERS.

    device: str
        Which device to us in PyTorch. Defaults to config.DEFAULT_DEVICE.

    Returns
    -------
        parameters: dict
            Collection of all model and inference parameters.

        model: axtrack.axtrack.machinelearning.YOLO_AXTrack(torch.Module)
            The growth cone detecter.

        stnd_scaler: (str, (float, float))
            The standardization used for training data. By default this will
            be ('zscore', (0.015176106, 0.009456525))
    """

    parameters = _get_params(num_workers, device,)
    set_seed(parameters['SEED'])
    turn_tex('on')
    model = _get_model(parameters, print_params)
    scaler_fname = f'{DEPLOYED_MODEL_DIR}/train_stnd_scaler.pkl'
    stnd_scaler = get_data_standardization_scaler(scaler_fname)
    os.makedirs(dest_dir, exist_ok=True)
    return parameters, model, stnd_scaler

def prepare_input_data(imseq_fname, parameters, dest_dir, inference_data_dir,
                        stnd_scaler, mask_fname=None, use_cached_datasets='to', 
                        check_preproc=False, input_metadata={}):
    """
    Generate the input data object to pass to the model.

    Arguments
    ---------
    imseq_fname: str
        The filename of the timelapse (.tif), excluding the path.
    
    parameters: dict
        Collection of all model and inference parameters.

    dest_dir: str
        The directory to which all model outputs will be saved. 
    
    inference_data_dir: str
        The path to the input data.

    stnd_scaler: (str, (float, float))
        The standardization used for training data. By default this will be
        ('zscore', (0.015176106, 0.009456525))
    
    mask_fname: str 
        The segmentation mask (.npy) with the region to detect. Defaults to
        None, detecting in the whole timelapse.

    use_cached_datasets: str 
        Pass None to not cache (save as file in dest_dir), pass 'to' to cache
        dataset, or pass 'from' to skip compute and load previous results from
        file. Default to 'to'.
    
    check_preproc: bool
        Weather to compare the processed input data with the model training
        data. A plot is produced and saved in dest_dir if True is passed.
        Defaults to False.
    
    input_metadata: dict
        Metadata for the input data timelapse. May include `name`, `dt_min`,
        `pixelsize`, `incubation_time_min`, `seeding_datetime`, 
        `notes`, `intensity_offset`, `clip_intensity`.
    
    Returns
    -------
        Timelapse: axtrack.axtrack.Timelapse(torch.Dataset)
            The input data to the model.
    """

    pad = input_metadata.get('pad')
    if pad:
        pad = [pad] *4
    # create a timelapse object with a dummy labels.csv
    timelapse = Timelapse(imseq_path = f'{inference_data_dir}/{imseq_fname}',
                    mask_path = f'{inference_data_dir}/{mask_fname}',
                    labels_csv = None,
                    timepoints = None,
                    pad = pad,  
                    use_transforms = [],
                    cache = dest_dir if use_cached_datasets=='to' else None,
                    from_cache = dest_dir if use_cached_datasets=='from' else None,
                    name = input_metadata.get('name'),
                    dt = input_metadata.get('dt_min'),
                    pixelsize = input_metadata.get('pixelsize'),
                    incubation_time = input_metadata.get('incubation_time_min'),
                    seeding_datetime = input_metadata.get('seeding_datetime'),
                    notes = input_metadata.get('notes'),
                    offset = input_metadata.get('intensity_offset'),
                    contrast_llim = input_metadata.get('clip_intensity'),
                    log_correct = parameters['LOG_CORRECT'],
                    standardize_framewise = parameters['STANDARDIZE_FRAMEWISE'],
                    standardize = stnd_scaler,
                    use_motion_filtered = parameters['USE_MOTION_DATA'],
                    use_sparse = parameters['USE_SPARSE'],
                    temporal_context= parameters['TEMPORAL_CONTEXT'],
                    plot = parameters['PLOT_PREPROC'], 
                    tilesize = parameters['TILESIZE'],
                    Sy = parameters['SY'],
                    Sx = parameters['SX'])
    
    if check_preproc:
        infrc_preproc_fname = save_preproc_metrics(dest_dir, timelapse)
        infrc_prproc = pd.read_csv(infrc_preproc_fname, index_col=0, header=[0,1,2])
        train_preproc_fname = f'{DEPLOYED_MODEL_DIR}/train_preproc_data.csv'
        train_preproc = pd.read_csv(train_preproc_fname, index_col=0, 
                                    header=[0,1,2]).loc[:,['train']]
        plot_preprocessed_input_data(pd.concat([infrc_prproc, train_preproc], 
                                     axis=1), name=timelapse.name, 
                                     dest_dir=dest_dir, show=False)
    return timelapse

def inference(timelapse, model, dest_dir, parameters, detections_cache='to', 
              astar_paths_cache='to', assigedIDs_cache='to',):
    """
    Detect growth cones in timelapse and associate to axon identities.

    Arguments
    ---------
    Timelapse: axtrack.axtrack.Timelapse(torch.Dataset)
        The input data to the model.
    
    model: axtrack.axtrack.machinelearning.YOLO_AXTrack(torch.Module)
            The growth cone detecter.
    
    dest_dir: str
        The directory to which all model outputs will be saved. 
    
    parameters: dict
        Collection of all model and inference parameters.
    
    detections_cache: str
        Pass None to not cache (save as file in dest_dir), pass 'to' to cache
        detections, or pass 'from' to skip compute and load previous results
        from file. Default to 'to'.
    
    astar_paths_cache: str
        Pass None to not cache (save as file in dest_dir), pass 'to' to cache A*
        paths between detections, or pass 'from' to skip compute and load
        previous results from file. Default to 'to'.
    
    assigedIDs_cache: str
        Pass None to not cache (save as file in dest_dir), pass 'to' to cache
        IDed detections, or pass 'from' to skip compute and load previous
        results from file. Default to 'to'.
    
    Returns
    -------
        axon_detections: axtrack.axtrack.AxonDetections
            Object wrapping detected axons. Final detections are accessed 
            through axon_detections.IDed_dets_all.
    """
    dets_dest_dir = f'{dest_dir}/axon_dets'
    axon_detections = AxonDetections(model, timelapse, parameters, dets_dest_dir)
    axon_detections.detect_dataset(cache=detections_cache)
    
    axon_detections.assign_ids(astar_paths_cache, assigedIDs_cache)
    return axon_detections

def visualize_inference(axon_dets, which_dets='IDed', 
                        description='', t_y_x_slice=[None,None,None], dets_kwargs=None, 
                        scnd_dets_kwargs=None, show=False, axon_subset=None, 
                        save_single_tiles=False, animated=False, dpi=160, fps=6, 
                        anim_fname_postfix='', draw_true_dets=False, 
                        draw_grid=True, draw_scalebar=False, 
                        draw_axon_reconstructions=False, draw_trg_paths=None, 
                        draw_brightened_bg=False):
    """
    Draw the detections on the timelapse. Produce single frames or a rendered
    .mp4. Results are saved into the directory associated with the AxonDetection
    object.

    Arguments
    ---------
    axon_dets: axtrack.axtrack.AxonDetections
        Object wrapping detected axons. Final detections are accessed through
        axon_detections.IDed_dets_all.

    which_dets: str
        Which detections to draw. Options are `all` (unfiltered detections),
        `confident` (hard-threshold to parameters["BBOX_THRESHOLD"], default
        0.7), and `IDed` (axon identities assigned through min-cost-flow
        algorithm). Defaults to `IDed`.
    
    description: str
        A description text to draw on each frame.
    
    t_y_x_slice: [(int, int), (int, int), (int, int)]
        Option to slice the data to be drawn in three dimensions (time, height,
        width). Tuples represent start and stop indices. Defaults to (None,
        None, None), i.e. no slicing.

    dets_kwargs: dict
        Drawing arguments for bounding boxes. Defaults to None, in which case
        config.PREDICTED_BOXES_KWARGS is used.

    scnd_dets_kwargs: dict
        Drawing arguments for secondary bounding boxes, usually ground truth
        boxes. Defaults to None, in which case config.GROUNDTRUTH_BOXES_KWARGS
        is used.
    
    show: bool
        Weather to show the video/ frames. Occasionally doesn't display video
        properly. Defaults to False.
        
    save_single_tiles: bool
        Weather to save the single tiles on which the detector operates.
        Defaults to False.

    animated: bool
        Weather to merge frames in a video. Defaults to False.
        
        
    dpi: int
        Dots per inch for video. Defaults to 160.
        
    fps: int
        Frames per second for video. Defaults to 6.


    anim_fname_postfix: str
        Filename postfix for video file.
        
    draw_true_dets: bool
        Weather to draw ground truth detections. Requires AxonDetection.labelled
        to be True. Defaults to False,

    draw_grid: bool
        Weather to draw tile boundaries. Defaults to True.as_integer_ratio()
        
    draw_scalebar: bool
        Weather to draw a scale bar. Requires metadata input when creating the
        Timelapse object (pixelsize). Defaults to False.
    
    draw_axon_reconstructions: bool
        Draw reconstructed axons. Not implemented, defaults to False.
    
    draw_trg_paths=None,
        Draw reconstructed paths to target. Not implemented, defaults to False.
                        
    draw_brightened_bg: bool
        Brighten the background using the segmentation mask of the Timelapse 
        object. Defaults to False.
    """
    draw_all(axon_dets, 
             which_dets = which_dets,
             description = description,
             t_y_x_slice = t_y_x_slice,
             dets_kwargs = dets_kwargs,
             scnd_dets_kwargs = scnd_dets_kwargs,
             show = show,
             axon_subset = axon_subset,
             save_single_tiles = save_single_tiles,
             animated = animated,
             dpi = dpi,
             fps = fps,
             anim_fname_postfix = anim_fname_postfix,
             draw_true_dets = draw_true_dets,
             draw_grid = draw_grid,
             draw_scalebar = draw_scalebar,
             draw_axon_reconstructions = draw_axon_reconstructions,
             draw_trg_paths = draw_trg_paths,
             draw_brightened_bg = draw_brightened_bg,)
