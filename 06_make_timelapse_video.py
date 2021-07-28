"""
THis produces the final, most elaborate visualization of moving growth cones.
Using the rgb_seq_vanilla.npy file from 02*.py, the single timepoint astar paths
from 04*.py, and the axon_labels_astardists.csv for the relative distance
trends, this produces an .mp4 video with the lablled axon and its shortest 
distance path towards the outputchannel and a graph that shows this ditances
over time. Each of these compontens can be turned of to produce the desired 
video. You can also choose to output a video for every single axon, or one video
with all of them. 
"""

import numpy as np
import pandas as pd
from skimage.util import img_as_uint
from skimage.util import img_as_ubyte

from skimage.draw import rectangle_perimeter
from skimage.draw import rectangle
from skimage.draw import line_aa
from skimage import morphology
from skimage import filters
import skimage.color

from tifffile import imsave, imread

import skvideo.io
import cv2


import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from scipy import sparse

def make_video(rgb_image, which_axons):
    time = 0
    to_plot = []
    print('Drawing...')
    for t in which_timepoints:
        print(f'\n\nt:{t}... ', end='\n')

        fig, ax = plt.subplots(figsize=(21, 13), facecolor='k',)
        canvas = FigureCanvas(fig)
        ax.set_facecolor('k')

        all_dists = axon_labels.loc[:, (slice(None), 'rel_distance')].unstack()
        ymin, ymax = all_dists.min()-200, all_dists.max()+200
        xmin, xmax = -1, sizet+1
        ax.set_ylim(ymin, ymax)
        ax.set_xlim(xmin, xmax)

        ax.axline((0, ymin), (0, ymax), color='grey', linewidth=3)
        ax.axline((xmin, 0), (xmax, 0), color='grey', linewidth=3)
        ax.vlines(range(1,sizet), ymin+400, ymin+430, color='grey', linewidth=2)
        ax.set_ylabel('distance to outputchannel', color='white', fontsize=50)
        ax.text(-1, 0, 'increases                  decreases', ha='center', va='center', color='white', fontsize=32, rotation=90)

        hours = int(time/60)
        minutes = time % 60
        text = f'{hours:0>2}:{minutes:0>2}h'
        ax.text(t, ymin+300, text, ha='center', va='center', color='white', fontsize=50)
        ax.text(sizet, 0, 'initial\ndistance', ha='center', va='center', color='white', fontsize=32)
        time += 40

        line_masks = []
        for axon_id in which_axons:
            first_tpoint = axon_labels.loc[:, axon_id].dropna().index[0]
            last_tpoint = axon_labels.loc[:, axon_id].dropna().index[-1]
            
            # timestep for which a label exists but also a label exists for t-1 
            if first_tpoint < t <= last_tpoint :
                data_t = axon_labels.loc[t,(axon_id, 'rel_distance')]
                data_tbefore = axon_labels.loc[t-1,(axon_id, 'rel_distance')]

                if draw_colored:
                    t_past = tau if t-first_tpoint > tau else t-first_tpoint
                    data_longbefore = axon_labels.loc[t-t_past,(axon_id, 'rel_distance')]
                    trend = (data_longbefore - data_t)/t_past
                    color = np.array(cmap(norm_col(trend))[:-1])
                    # color = axon_labels.loc[t, (axon_id, ['col_r', 'col_g', 'col_b'])]
                else:
                    color = 'grey'
                to_plot.append([(t-1, t), (data_tbefore, data_t), color])
            
            # this is the first timepoint of that axon
            elif first_tpoint == t:
                color = np.array(cmap(norm_col(0))[:-1])
            # this is a timestamp that the axon has no label for
            else:
                continue

            y, x = axon_labels.loc[t, (axon_id, ['anchor_y', 'anchor_x'])].astype(int)
            y_start, x_start = axon_labels.loc[t, (axon_id, ['topleft_y', 'topleft_x'])]
            y_end, x_end = axon_labels.loc[t, (axon_id, ['bottomright_y', 'bottomright_x'])]
            
            ### draw axon distance to target line into rgb_image ###
            if draw_target_path:
                coo_dists = sparse.load_npz(f'{outp_dir}/axon_dists/{axon_id}_{t}.npz')
                dashed_coo_y = [r for i, r in enumerate(coo_dists.row) if i%50<30]
                dashed_coo_x = [c for i, c in enumerate(coo_dists.col) if i%50<30]

                mask_dashed_line = np.zeros(rgb_image.shape[1:-1], dtype=bool)
                mask_dashed_line[dashed_coo_y, dashed_coo_x] =  1

                mask_dashed_line = morphology.binary_dilation(mask_dashed_line, np.ones((5,5)))
                dashed_line = skimage.color.gray2rgb(mask_dashed_line).astype(float)
                dashed_line[mask_dashed_line, :] = color
                dashed_line = filters.gaussian(dashed_line, 2,  multichannel=True)
                
                rgb_image[t, mask_dashed_line, :] = img_as_uint(dashed_line[mask_dashed_line])
                
            ### draw axon bounding box into rgb_image ###
            if draw_bbox:
                for w in range(4):
                    rr, cc = rectangle_perimeter((y_start+w, x_start+w), end=(y_end+w, x_end+w), 
                                                 shape=rgb_image.shape[1:])
                    rgb_image[t, rr, cc] = img_as_uint(color)
                if annotate_bbox:
                    font = cv2.FONT_HERSHEY_DUPLEX
                    bottomLeftCornerOfText = (int(x_start), int(y_start-10))
                    fontScale = .7
                    fontColor = img_as_uint(color).tolist()
                    lineThickness = 2
                    cv2.putText(rgb_image[t],axon_id, 
                        bottomLeftCornerOfText, 
                        font, 
                        fontScale,
                        fontColor,
                        lineThickness)
        
        for oneaxon_onet in to_plot:
            ax.plot(*oneaxon_onet[:-1], color=oneaxon_onet[2], linewidth=2)

        if draw_figure:
            canvas.draw()
            s, (width, height) = canvas.print_to_buffer()
            img = np.frombuffer(s, np.uint8).reshape((height, width, 4))[:,:,:-1]
            img = img_as_uint(img)

            y_size, x_size = img.shape[:-1]
            insert_y, insert_x = 0,0
            
            rgb_image[t, insert_y:insert_y+y_size, insert_x:insert_x+x_size, :] = img
        plt.close()
        
    rgb_image = img_as_ubyte(rgb_image)
    writer = skvideo.io.FFmpegWriter(fname, inputdict={'-r': '3'})
    for i in range(sizet):
        print(f'frame:{i}..', end='')
        writer.writeFrame(rgb_image[i, :, :, :])
    [writer.writeFrame(np.zeros(rgb_image[0].shape)) for _ in range(4)]
    writer.close()
    print('Done writing video.')

outp_dir = '../tl140_outputdata'
axon_labels = pd.read_csv(f'{outp_dir}/labelled_axons_astardists.csv', 
                          index_col=0, header=[0,1])

# what to draw paramaters
draw_target_path = True
draw_bbox = True
annotate_bbox = True
draw_figure = True
draw_colored = True
single_axons_videos = False

# parameters when using colored
tau = 3 # when computing the trend, take the average over tau timepoints
cmap = cm.get_cmap('RdYlGn_r')
vmin, vmax = -100, 100
norm_col = colors.Normalize(vmin=vmin, vmax=vmax)

# # Below for all axons
if not single_axons_videos:
    print('Loading image...', end='')
    rgb_image = np.load(f'{outp_dir}/rgb_seq_vanilla.npy')
    print('Done.')
    sizet = rgb_image.shape[0]
    # how many timepoints and how many axons
    which_timepoints = range(sizet)
    which_axons = axon_labels.columns.unique(0)
    # filename for video
    fname = f'{outp_dir}/videos/tl1_40min.mp4'
    print('\n', fname)
    make_video(rgb_image, which_axons)

# Below for single axons
else:
    for axon in axon_labels.columns.unique(0):
        fname = f'{outp_dir}/videos/{axon}.mp4'
        print('\n\n\n\n', fname)
        which_axons = axon,

        print('Loading image...', end='')
        rgb_image = np.load(f'{outp_dir}/rgb_seq_vanilla.npy')
        print('Done.')
        sizet = rgb_image.shape[0]
        # how many timepoints and how many axons
        which_timepoints = range(sizet)
        make_video(rgb_image, which_axons)