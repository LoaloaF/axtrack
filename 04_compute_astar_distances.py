"""
This updates the axon_labels.csv with aastar distances towards the output channel.
The new axon labels csv is saved as axon_labels_astardists.csv. This path is computed 
on the previously segmented background, more specifically the mask_wells_excl.npy 
file with some added erosion to find paths in the center of the structure instead 
of hugging the edges. On top of that, the axon path from its first to its last 
timepoint is saved as a sparse matrix in axon_reconstructions/. 
"""

import numpy as np
import pandas as pd
from skimage import morphology
from scipy import sparse

import pyastar

def get_astar_path(source, target, weights):
    path_coo = pyastar.astar_path(weights, source, target)
    ones, row, cols = np.ones(path_coo.shape[0]), path_coo[:,0], path_coo[:,1]
    return sparse.coo_matrix((ones, (row, cols)), shape=weights.shape, dtype=bool)

def main():
    # axons_index = range(78)
    # sizet = 37
    target = 991, 2746
    outputdir = '../tl140_outputdata'

    axon_labels = pd.read_csv(f'{outputdir}/labelled_axons_eucldists.csv', index_col=0, header=[0,1])
    which_axons = axon_labels.columns.unique(0)
    excl_mask = np.load(f'{outputdir}/mask_wells_excl.npy')
    excl_mask = morphology.erosion(excl_mask, morphology.selem.square(6))
    weights = np.where(excl_mask==1, 1, 2**16).astype(np.float32)

    for axon in which_axons:
        print(f'{axon}...')

        # make dataframe to save all information 
        timepoints = axon_labels.loc[:, axon].dropna().index.values.astype(int)
        anchor_x = axon_labels.loc[timepoints, (axon, 'anchor_x')].values.astype(int)
        anchor_y = axon_labels.loc[timepoints, (axon, 'anchor_y')].values.astype(int)

        # compute the shortest path between first growth cone point and last point
        # this is purely for visualization
        coo_path = get_astar_path((anchor_y[0], anchor_x[0]), (anchor_y[-1],anchor_x[-1]), weights)
        sparse.save_npz(f'{outputdir}/axon_reconstructions/{axon}.npz', coo_path)
        
        distances = []
        for t in range(len(timepoints)):
            print(f't:{t}...', end='')
            coo_path = get_astar_path((anchor_y[t], anchor_x[t]), target, weights)
            # in case you want to save every timestep
            sparse.save_npz(f'{outputdir}/axon_dists/{axon}_{timepoints[t]}.npz', coo_path)
            distances.append(len(coo_path.data))
        dist = np.array(distances)

        axon_labels.loc[timepoints, (axon, 'distance')] = dist
        axon_labels.loc[timepoints, (axon, 'rel_distance')] = dist[np.where(timepoints == min(timepoints))] - dist
        print('Ok.\n\n')
    axon_labels.to_csv(f'{outputdir}/labelled_axons_astardists.csv')

if __name__ == '__main__':
    main()