import os

import numpy as np
import pandas as pd

from scipy import sparse
from scipy.optimize import linear_sum_assignment

from concurrent.futures import ThreadPoolExecutor
from itertools import repeat

import pyastar

def cntn_id_cost(t1_conf, t0_conf, dist, alpha, beta):
    return ((alpha*dist) * (1/(1 + np.e**(beta*(t1_conf+t0_conf-1))))).astype(float)

def create_id_cost(conf, alpha, beta):
    return 1/(1 + np.e**(beta*(conf-.5))).astype(float)

def generate_axon_id():
    axon_id = 0
    while True:
        yield axon_id
        axon_id += 1

def hungarian_ID_assignment(cntd_cost_detections, alpha, beta):
    t0_conf = pd.DataFrame([])
    axon_id_generator = generate_axon_id()

    # iterate detections: [(conf+anchor coo) + distances to prv frame dets]
    assigned_ids = []
    for t, cntn_c_dets in enumerate(cntd_cost_detections):
        # unpack confidences of current t 
        t1_conf = cntn_c_dets.iloc[:,0]
        # each detections will have a new ID assigned to it, create empty here
        ass_ids = pd.Series(-1, index=cntn_c_dets.index, name=t)
        
        if t == 0:
            # at t=0 every detections is a new one, special case
            ass_ids[:] = [next(axon_id_generator) for _ in range(len(ass_ids))]
        else:
            # t1 and t0 have different lengths, save shorter n of the two
            nt1 = cntn_c_dets.shape[0]
            ndet_min = min((nt1, cntn_c_dets.shape[1]-3))

            # get the costs for creating t1 ID and killing t0 ID (dist indep.)
            create_t1_c = create_id_cost(t1_conf, alpha, beta).values
            kill_t0_c = create_id_cost(t0_conf, alpha, beta).values
            
            # get the costs for continuing IDs
            cntn_c_matrix = cntn_c_dets.iloc[:,3:].values
            # hungarian algorithm
            t1_idx_cntn, t0_idx_cntn = linear_sum_assignment(cntn_c_matrix)
            # index costs to assignments, associating t1 with t0, dim == ndet_min
            cntn_t1_c = cntn_c_matrix[t1_idx_cntn, t0_idx_cntn]

            # an ID is only NOT continued if the two associated detections both 
            # have lower costs on their own (kill/create)
            cntn_t1_mask = (cntn_t1_c<create_t1_c[:ndet_min]) | (cntn_t1_c<kill_t0_c[:ndet_min])
            # pad t1 mask to length of t1 (in case t1 was shorter than t0)
            cntn_t1_mask = np.pad(cntn_t1_mask, (0, nt1-ndet_min), constant_values=False)
            created_t1_mask = ~cntn_t1_mask

            # created IDs simply get a new ID from the generator
            ass_ids[created_t1_mask] = [next(axon_id_generator) for _ in range(created_t1_mask.sum())]
            # for cntn ID, mask the t0_idx (from hung. alg.) and use it as 
            # indexer of previous assignments timpoint
            assoc_id = t0_idx_cntn[cntn_t1_mask[:ndet_min]]
            ass_ids[cntn_t1_mask] = assigned_ids[-1].iloc[assoc_id].values

        t0_conf = t1_conf
        assigned_ids.append(ass_ids)
    assigned_ids = pd.concat(assigned_ids, axis=1).stack().swaplevel().sort_index()
    assigned_ids = assigned_ids.apply(lambda ass_id: f'Axon_{int(ass_id):0>3}')
    return assigned_ids

def greedy_linear_assignment(cntd_cost_detections):
    # alternative to hacky hungarian method, not implemented rn
    pass

def dist2cntn_cost(dists_detections, alpha, beta):
    for i in range(len(dists_detections)):
        dist = dists_detections[i]
        if i == 0:
            t0_conf = dist.conf
            continue
        t1_conf = dist.conf

        # t1 detection confidence vector to column-matrix
        t1_conf_matrix = np.tile(t1_conf, (len(t0_conf),1)).T
        # t0 detection confidence vector to row-matrix
        t0_conf_matrix = np.tile(t0_conf, (len(t1_conf),1))
        # get the distance between t0 and t1 detections 
        t1_t0_dists_matrix = dist.iloc[:, 3:].values
        
        # pass distances and anchor conf to calculate assoc. cost between detections
        dist.iloc[:,3:] = cntn_id_cost(t1_conf_matrix, t0_conf_matrix, 
                                            t1_t0_dists_matrix, alpha, beta)
        t0_conf = t1_conf
    return dists_detections

def compute_astar_path(source, target, weights, return_path=True):
    path_coo = pyastar.astar_path(weights, source, target)
    if return_path:
        ones, row, cols = np.ones(path_coo.shape[0]), path_coo[:,0], path_coo[:,1]
        path = sparse.coo_matrix((ones, (row, cols)), shape=weights.shape, dtype=bool)
        return path_coo.shape[0], path
    return path_coo.shape[0]

def get_detections_distance(t1_det, t0_det, weights, max_dist, fname):
    # print(t1_det, t0_det, weights, max_dist)
    t1id, t1y, t1x = t1_det 
    t0id, t0y, t0x = t0_det
    dist = np.sqrt((t1y-t0y)**2 + (t1x-t0x)**2)
    if dist < max_dist:
        dist, path = compute_astar_path((t1y,t1x), (t0y,t0x), weights)
        sparse.save_npz(f'{fname}_id{t0id}-id{t1id}.npz', path)
    dist = min((dist, max_dist))
    return dist

def compute_astar_distances(t1_dets, t0_dets, mask, max_dist, num_workers, filename):
    # check if a cached file exists, must hace the correct shape (lazy check)
    if os.path.exists(filename+'.npy'):
        distances = np.load(filename+'.npy')
        if distances.shape == (t1_dets.shape[0], t0_dets.shape[0]):
            return distances
    
    # get the cost matrix for finding the astar path
    weights = np.where(mask==1, 1, 2**16).astype(np.float32)
    # get the current ids (placeholders)
    t1_ids = [int(idx[-3:]) for idx in t1_dets.index]
    t0_ids = [int(idx[-3:]) for idx in t0_dets.index]
    
    # iterate t1_detections, compute its distance to all detections in t0_dets
    distances = []
    t0_dets = np.stack([t0_ids, t0_dets.anchor_y, t0_dets.anchor_x], -1)
    for i, t1_det in enumerate(np.stack([t1_ids, t1_dets.anchor_y, t1_dets.anchor_x], -1)):
        print(f'{i}/{len(t1_dets)}', end='...')

        with ThreadPoolExecutor(num_workers) as executer:
            args = repeat(t1_det), t0_dets, repeat(weights), repeat(max_dist), repeat(filename)
            dists = executer.map(get_detections_distance, *args)
        distances.append([d for d in dists])
    distances = np.array(distances)
    np.save(filename+'.npy', distances)
    return distances

def assign_detections_ids(data, model, params, run_dir):
    alpha, beta, gamma = 0.1, 6, 0.2
    max_dist = 400
    device, conf_thr, num_workers = params['DEVICE'], params['BBOX_THRESHOLD'], params['NUM_WORKERS']
    # conf_thr = np.log((1/gamma) -1)/beta +0.5
    max_cost = cntn_id_cost(conf_thr, conf_thr, np.array([max_dist]), alpha, beta)
    print('Assigning axon IDs ...', end='')

    # aggregate all confidences+anchor coo and distances over timepoints
    dists_detections = []
    for t in range(data.sizet):
        if t == 0:
            dets_t0 = model.get_frame_detections(data, t, device, conf_thr)[3]
            dets_t0 = dets_t0.sort_values('conf', ascending=False)
            dists_detections.append(dets_t0)
            continue
        lbl = f'{data.name}_t1:{t:0>3}-t0:{t-1:0>3}'
        print(lbl, end='...')

        # get detections and stitch tiles to frame, only get stitched prediction
        dets_t1 = model.get_frame_detections(data, t, device, conf_thr)[3]
        # reorder detections by confidence
        dets_t1 = dets_t1.sort_values('conf', ascending=False)
        
        # compute a* dists_detections between followup and current frame detections 
        dists = compute_astar_distances(dets_t1, dets_t0, data.mask, max_dist,
                                        num_workers, filename=f'{run_dir}/astar_dists/{lbl}')

        dists = pd.DataFrame(dists, index=dets_t1.index, columns=dets_t0.index)
        dists_detections.append(pd.concat([dets_t1, dists], axis=1))
        dets_t0 = dets_t1

    # pass confidences, distances and hyperparameters to compute cost matrix
    cntd_cost_detections = dist2cntn_cost(dists_detections, alpha, beta)

    assigned_ids = hungarian_ID_assignment(cntd_cost_detections, alpha, beta)
    assigned_ids.to_csv(f'{run_dir}/model_out/{data.name}_assigned_ids.csv')
    print('Done.')