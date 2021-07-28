"""
Construct axons from astar computed path. axons_labels_astardists.csv is also
used to color the reconstructed axons according to the relative distance towards
the output channel (red= growing away, green=growing towards). The reconstructed 
axons are drawn onto the transmission channel, masked by the background 
segmentation. At the last point, a growth cone is drawn as well. Overall this
code is quite crappy, the rgb conversion is a little ugly. Final reconstructions
are saved as cropped, compressed .tifs in axon_reconstructions/
"""
import numpy as np
import pandas as pd
from matplotlib import cm
from matplotlib import colors

from tifffile import imsave, imread
from scipy import sparse

import skimage.color
from skimage.util import img_as_uint
from skimage import morphology
from skimage import filters

# load all the relevent files
inp_path = '/run/media/loaloa/lbb_ssd/timelapse1_40_min_processed/'
outputdir = '../tl140_outputdata'
axon_labels = pd.read_csv(f'{outputdir}/labelled_axons_astardists.csv', 
                          index_col=0, header=[0,1])
excl_mask = np.load(f'{outputdir}/mask_wells_excl.npy')
background = imread(inp_path+'G001_grey_compr.deflate.tif')[0] * 16
mask_growthcone = imread('growthcone.tif')[:,:,0].astype(bool)

# don't remember why I am doing all the multiplications here and above
background[~excl_mask] = 2**13
background = skimage.color.gray2rgb(background)

# this is for coloring the axon according to the growth directionality
cmap = cm.get_cmap('RdYlGn')
vmin, vmax = -800, 800
norm_col = colors.Normalize(vmin=vmin, vmax=vmax)
for axon in axon_labels.columns.unique(0):
    print(f'{axon}...', end='')
    trend = axon_labels.loc[:,(axon, 'rel_distance')].dropna().values[-1]
    color = np.array(cmap(norm_col(trend))[:-1])

    # get the path from growthcone start point to end point
    coo_dists = sparse.load_npz(f'{outputdir}/axon_reconstructions/{axon}.npz')
    dashed_coo_y = [r for i, r in enumerate(coo_dists.row)]
    dashed_coo_x = [c for i, c in enumerate(coo_dists.col)]

    # make a binary image of the path, then dilate to desired thickness, here 6 px
    mask_dashed_line = np.zeros(background.shape[:-1], dtype=bool)
    mask_dashed_line[dashed_coo_y, dashed_coo_x] =  1
    mask_dashed_line = morphology.binary_dilation(mask_dashed_line, np.ones((6,6)))
    
    # convert to rgb and draw the growthcone 
    dashed_line = skimage.color.gray2rgb(mask_dashed_line).astype(float)
    dashed_line[mask_dashed_line, :] = color
    growthcone = skimage.color.gray2rgb(mask_growthcone).astype(float)
    growthcone[mask_growthcone, :] = color

    # blur out the growth cone?
    gc_y, gc_x = slice(dashed_coo_y[-1]-12, dashed_coo_y[-1]+13), slice(dashed_coo_x[-1]-12,dashed_coo_x[-1]+13) 
    dashed_line[gc_y, gc_x, :] = .3 * dashed_line[gc_y, gc_x, :] + .7 * growthcone

    # do some weird  thresholding?
    dashed_line = filters.gaussian(dashed_line, 1.4,  multichannel=True)
    mask_dashed_line = dashed_line.mean(axis=2) > .06

    final_img = background.copy()
    final_img[mask_dashed_line] = .3*background[mask_dashed_line] + .7*img_as_uint(dashed_line[mask_dashed_line])
    # crop to axon + 400 px padding:
    ymin, ymax = min(dashed_coo_y)-400, max(dashed_coo_y)+400
    ymin, ymax = max(0, ymin), min(ymax, final_img.shape[0])
    xmin, xmax = min(dashed_coo_x)-400, max(dashed_coo_x)+400
    xmin, xmax = max(0, xmin), min(xmax, final_img.shape[1])
    
    imsave(f'{outputdir}/axon_reconstructions/{axon}.tif', final_img[ymin:ymax, xmin:xmax], compression='deflate')