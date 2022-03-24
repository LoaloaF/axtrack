# -*- coding: utf-8 -*-
# Author : HuangPiao
# Email  : huangpiao2985@163.com
# Date   : 1/12/2019

import numpy as np
import cv2
import matplotlib.pyplot as plt

from libmot.data_association  import MinCostFlowTracker
from copy import deepcopy
import time
from libmot.utils import iou
import motmetrics as mm
import pandas as pd
import colorsys
import pickle


def observation_model(**kwargs):
    """Compute Observation Cost for one frame

    Parameters
    ------------
    scores: array like
        (N,) matrix of detection's confidence

    Returns
    -----------
    costs: ndarray
        (N,) matrix of detection's observation costs
    """
    scores = np.array(kwargs['scores']).astype(np.float)
    

    scores = np.log(-0.446 * scores + 0.456)
    return scores
    # why not like in paper where beta (uncertainty) is used instead of 
    # confidence? Maybe relaxation because conf == 1 gives neg infinity
    scores = (scores-1) *-1   # conf to beta
    scores = (np.log(scores/(1-scores)))
    scores[scores>max_conf_cost] = max_conf_cost
    scores[scores<-max_conf_cost] = -max_conf_cost
    print(scores)
    return scores

def feature_model(**kwargs):
    """Compute each object's K features in a frame

    Parameters
    ------------
    image: ndarray
        bgr image of ndarray
    boxes: array like
        (N,4) matrix of boxes in (x,y,w,h)

    Returns
    ------------
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
        cropped = np.moveaxis(image[:, x1:x3, x2:x4], 0, -1)[:,:,0]
        hist = cv2.calcHist([cropped], [0], None, [180], [0, 1])
        cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        features[i] = deepcopy(hist)
    return features

def transition_model(**kwargs):
    """Compute costs between track and detection

    Parameters
    ------------
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
    ------------
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

    # # A* distances component of cost
    # lbl = f't{frame_idx:0>3}-t_before{frame_idx-(time_gap):0>3}'
    # distances = ((astar_dists[lbl] /max_px_assoc_dist) -1) *-1
    # inf_dist = distances == 0

    # # visual similarity component of cost
    # vis_sim = []
    # for f1 in predecessor_features:
    #     vis_sim_f1 = []
    #     for f2 in features:
    #         vis_sim_f1.append(1 - cv2.compareHist(f1, f2, cv2.HISTCMP_BHATTACHARYYA))
    #     vis_sim.append(vis_sim_f1)
    # vis_sim = np.array(vis_sim).T
    
    # costs = -np.log((1-vis_sim_weight) * distances*(miss_rate ** (time_gap-1)) \
    #                 + vis_sim_weight * vis_sim \
    #                 + 1e-5)
    # costs[inf_dist] = np.inf
    # return costs.T


    assert len(boxes) == len(features) and len(predecessor_boxes) == len(predecessor_features), \
        "each boxes and features's length must be same"

    # warp_matrix, _ = ECC(images[frame_idx - time_gap - 2], images[frame_idx - 1],
    #                    warp_mode=cv2.MOTION_EUCLIDEAN,
    #                    eps=0.01, max_iter=100, scale=0.1, align=False)

    if len(boxes) == 0 or len(predecessor_boxes) == 0:
        return np.zeros((0,))
    costs = np.zeros((len(predecessor_boxes), len(boxes)))
    for i, (box1, feature1) in enumerate(zip(predecessor_boxes, predecessor_features)):
        """
        points = np.array([box1[:2],
                           box1[:2] + box1[2:] - 1])
        points_aligned = AffinePoints(points.reshape(2, 2), warp_matrix)
        points_aligned = points_aligned.reshape(1, 4)
        boxes_aligned = np.c_[points_aligned[:, :2],
                              points_aligned[:, 2:] - points_aligned[:, :2] + 1]
        boxes_aligned[:, 2:] = np.clip(boxes_aligned[:, 2:], 1, np.inf)
        box1 = boxes_aligned.squeeze()
        """
        for j, (box2, feature2) in enumerate(zip(boxes, features)):
            if max(abs(box1[0] - box2[0]), abs(box1[1] - box2[1])) > max(box1[2], box1[3], box2[2], box2[3]):
                costs[i, j] = np.inf
            else:
                costs[i, j] = -np.log(1.0 * float(iou(box1, box2)) * (miss_rate ** time_gap) \
                                      + 0 * (1 - cv2.compareHist(feature1, feature2,
                                                                   cv2.HISTCMP_BHATTACHARYYA)) + 1e-5)
    print('Cost: ', '\n', costs, costs.shape)
    return costs


def get_tracks():
    entry_exit_cost = default['entry_exit_cost']
    miss_rate = default['miss_rate']
    max_num_misses = default['max_num_misses']
    thresh = default['thresh']
    
    record = []
    track_model = MinCostFlowTracker(observation_model=observation_model,
                                     transition_model=transition_model, feature_model=feature_model,
                                     entry_exit_cost=entry_exit_cost, min_flow=min_flow,
                                     max_flow=max_flow, miss_rate=miss_rate, max_num_misses=max_num_misses,
                                     cost_threshold=thresh)

    for i in range(track_len):
        print('Processing frame: ', i, end='...')
        det = dets[(dets[:, 0] == i), :]
        track_model.process(boxes=det[:, 2: 6].astype(np.int32),
                            scores=det[:, 6], image=images[i],
                            frame_idx=i)
        print('Done.\n\n\n')

    trajectory = track_model.compute_trajectories()
    # trajectory is a list [IDs] of list of boxes (in frames) with format 
    # frame_idx, box_idx, box_coo (4 values)
    # [print(t, '\n') for t in trajectory]
    for i, t in enumerate(trajectory):
        # i = which ID
        for j, box in enumerate(t):
            record.append([box[0] + 1, i + 1, box[2][0], box[2][1], box[2][2], box[2][3]])
    record = np.array(record)
    record = record[np.argsort(record[:, 0])]
    return record

def make_video(tracks):
    from matplotlib.animation import ArtistAnimation
    from matplotlib.patches import Rectangle

    fig, ax = plt.subplots()

    framewise_artists = []
    for frame in range(track_len):
        im = images[frame]
        print(im.shape)
        if im.shape[0] == 1:
            im = im[0]
        im = ax.imshow(im, animated=True)

        boxes = tracks[tracks[:,0] == frame]
        recs = []
        for box in boxes:
            ID, x, y, width, height = box[1:]
            col = plt.cm.get_cmap('hsv', 10)(ID%10)
            recs.append(Rectangle((x,y), width, height, edgecolor=col, facecolor='none'))
        [ax.add_patch(rec) for rec in recs]
        t = ax.text(0,0, frame, zorder=1)
        
        framewise_artists.append([im, *recs, t])
    an = ArtistAnimation(fig, framewise_artists, blit=True)
    plt.show()

def evaluation(tracks):
    gts = pd.DataFrame(gt[:, :6], columns=['FrameId', 'Id', 'X', 'Y', 'Width', 'Height'])
    gts = gts.set_index(['FrameId', 'Id'])
    gts[['X', 'Y']] -= (1, 1)

    box = pd.DataFrame(np.array(tracks), columns=['FrameId', 'Id', 'X', 'Y', 'Width', 'Height'])
    box = box.set_index(['FrameId', 'Id'])
    box[['X', 'Y']] -= (1, 1)

    acc = mm.utils.compare_to_groundtruth(gts, box, 'iou', distth=0.5)
    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=mm.metrics.motchallenge_metrics, \
                         return_dataframe=False)
    return abs(summary['mota']), abs(summary['motp']), abs(summary['idf1']), abs(summary['num_switches'])







# parameters
default = {'entry_exit_cost': 5, 'thresh': 1.8,
           'miss_rate': 0.8, 'max_num_misses': 12}
# test
default = {'entry_exit_cost': 2, 'thresh': .8,
           'miss_rate': 0.8, 'max_num_misses': 1}
min_flow = 1
max_flow = 120
vis_sim_weight = 0.1
max_conf_cost = 4.6
track_len = 4   # excluding last number , starting from 1
max_px_assoc_dist = 500



# MOT data
base_dir = '/home/loaloa/datasets'
images = []
for i in range(1, track_len+1):
    images.append(cv2.imread(f'{base_dir}/MOT17/train/MOT17-10-SDP/img1/00%04d.jpg' % i))
dets = np.genfromtxt(f'{base_dir}/MOT17/train/MOT17-10-SDP/det/det.txt', delimiter=',')
dets = dets[(dets[:, 0] <= track_len) & (dets[:, 6] > 0.7), :]
dets[:, 0] -= 1
# dets format: frame - ID - x_topleft, y_topleft, width, height, ratio occluded 
dets = dets.astype(np.int32)
dets[:,-1] = 1
print(dets[dets[:,-1]==1])
print(dets[dets[:,-1]==1].shape)



# groundtruth for evaluation
gt = np.genfromtxt(f'{base_dir}/MOT17/train/MOT17-10-SDP/gt/gt.txt', delimiter=',')
gt = gt[(gt[:, 0] < track_len) & (gt[:, 6] == 1), :]
mask = (gt[:, 7] == 1) | (gt[:, 7] == 2) | (gt[:, 7] == 7)
gt = gt[mask].astype(np.int32)



# # testing on axon detection data
# images = pickle.load(open('./axon_test_imgs.pkl', 'rb'))
# astar_dists = pickle.load(open('./axon_test_dists.pkl', 'rb'))
# dets = np.load('./axon_test_dets.npy', allow_pickle=True)
# # dets[:, 0] -= 1
# dets[:, -1][dets[:, -1]>1] = 1
# print(dets)



# do the association using min cost flow 
time0 = time.time()
track = np.array(get_tracks())
track[:,0] -= 1
print(track)
print(track.shape)
make_video(track)
# print(evaluation(track))
















# rgb = lambda x: colorsys.hsv_to_rgb((x * 0.41) % 1, 1., 1. - (int(x * 0.41) % 4) / 5.)
# colors = lambda x: (int(rgb(x)[0] * 255), int(rgb(x)[1] * 255), int(rgb(x)[2] * 255))
# # draw = np.concatenate((images[0], images[15]), axis=1)
# sz = images[0].shape
# boxes = track[track[:, 0] == 1]
# id_list = list(boxes[:, 1])
# boxes = boxes[:, 2:6]

# print()
# print(boxes)
# print(id_list)

# track_boxes = []

# for i in id_list:
#     t = track[(track[:, 0] == 16) & (track[:, 1] == i)]
#     if t.size > 0:
#         track_boxes.append(t[:, 2:6].squeeze())
#     else:
#         track_boxes.append(None)

# for i, (bbox, tracked_bbox) in enumerate(zip(boxes, track_boxes)):
#     x_tl = (bbox[0], bbox[1])
#     x_br = (bbox[0] + bbox[2], bbox[1] + bbox[3])
#     x_center = (int(bbox[0] + bbox[2] / 2), int(bbox[1] + bbox[3] / 2))
#     cv2.rectangle(draw, x_tl, x_br, colors(i), 5)

#     if tracked_bbox is not None:
#         y_tl = (tracked_bbox[0] + sz[1], tracked_bbox[1])
#         y_br = (tracked_bbox[0] + tracked_bbox[2] + sz[1], tracked_bbox[1] + tracked_bbox[3])
#         y_center = (int(tracked_bbox[0] + tracked_bbox[2] / 2 + sz[1]), int(tracked_bbox[1] + tracked_bbox[3] / 2))

#         cv2.rectangle(draw, y_tl, y_br, colors(i), 5)
#         cv2.line(draw, x_center, y_center, colors(i), 3)

# plt.imshow(cv2.cvtColor(draw, cv2.COLOR_BGR2RGB))
# plt.title("offline MinCostFlow Tracker")
# plt.show()
