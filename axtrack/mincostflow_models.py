from copy import deepcopy

import cv2
import numpy as np

def observation_model(**kwargs):
    """Compute Observation Cost for one frame

    Arguments
    ---------
    scores: array like
        (N,) matrix of detection's confidence

    Returns
    -------
    costs: ndarray
        (N,) matrix of detection's observation costs
    """
    scores = np.array(kwargs['scores']).astype(np.float)
    
    # why not like in paper where beta (uncertainty) is used instead of 
    # confidence? Maybe relaxation because conf == 1 gives neg infinity
    scores = (scores-1) *-1 +1e-6  # conf to beta
    scores = (np.log(scores/(1-scores)))
    scores[scores>kwargs['max_conf_cost']] = kwargs['max_conf_cost']
    scores[scores<-kwargs['max_conf_cost']] = -kwargs['max_conf_cost']
    return scores
    # return np.log(-0.446 * np.array(kwargs['scores']).astype(np.float) + 0.456)

def feature_model(**kwargs):
    """Compute each object's K features in a frame

    Arguments
    ---------
    image: ndarray
        bgr image of ndarray
    boxes: array like
        (N,4) matrix of boxes in (x,y,w,h)

    Returns
    -------
    features: array like
        (N,K,1) matrix of features
    """
    assert 'image' in kwargs and 'boxes' in kwargs, 'Parameters must contail image and boxes'

    boxes = kwargs['boxes']
    image = kwargs['image']
    if len(boxes) == 0:
        return np.zeros((0,))

    boxes = np.atleast_2d(deepcopy(boxes))
    features = np.zeros((boxes.shape[0], 180, 1), dtype=np.float32)

    for i, roi in enumerate(boxes):
        x1 = max(roi[1], 0)
        x2 = max(roi[0], 0)
        x3 = max(x1 + 1, x1 + roi[3])
        x4 = max(x2 + 1, x2 + roi[2])
        cropped = image[x1:x3, x2:x4]
        # cropped = np.moveaxis(image[:, x1:x3, x2:x4], 0, -1)[:,:,0]
        hist = cv2.calcHist([cropped], [0], None, [180], [0, 1])
        cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        features[i] = deepcopy(hist)
    return features

def transition_model(**kwargs):
    """Compute costs between track and detection

    Arguments
    ---------
    miss_rate: float
        the similarity for track and detection will be multiplied by miss_rate^(time_gap - 1)
    time_gap: int
        number of frames between track and detection
    predecessor_boxes: ndarray
        (N,4) matrix of boxes in track's frame
    boxes: ndarray
        (M,4) matrix of boxes in detection's frame
    predecessor_features: ndarray
        (N,180, 256) matrix of features in track's frame
    features: ndarray
        (M,180, 256) matrix of features in detection's frame
    frame_idx: int
        frame id, begin from 1

    Returns
    -------
    costs: ndarray
        (N,M) matrix of costs between track and detection
    """
    miss_rate = kwargs['miss_rate']
    time_gap = kwargs['time_gap']
    boxes = kwargs['boxes']
    predecessor_boxes = kwargs['predecessor_boxes']
    features = kwargs['features']
    predecessor_features = kwargs['predecessor_features']
    frame_idx = kwargs['frame_idx']

    # A* distances component of cost (major component)
    lbl = f'{kwargs["dataset_name"]}_t:{frame_idx:0>3}-t:{frame_idx-(time_gap):0>3}'
    # scale distances 0to1 - 0 ist best, 1 is worst
    distances = ((kwargs['astar_dists'][lbl] /kwargs['max_px_assoc_dist']) -1) *-1
    inf_dist = distances == 0

    # visual similarity component of cost
    vis_sim = []
    for f1 in predecessor_features:
        vis_sim_f1 = []
        for f2 in features:
            vis_sim_f1.append(1 - cv2.compareHist(f1, f2, cv2.HISTCMP_BHATTACHARYYA))
        vis_sim.append(vis_sim_f1)
    vis_sim = np.nan_to_num(np.array(vis_sim))
    
    costs = -np.log((1-kwargs['vis_sim_weight']) * distances*(miss_rate ** (time_gap-1)) \
                    + kwargs['vis_sim_weight'] * vis_sim \
                    + 1e-6)
    costs[inf_dist] = np.inf
    return costs
    