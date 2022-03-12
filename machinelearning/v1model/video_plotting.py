import pickle

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.animation import ArtistAnimation, writers

from skimage.filters import gaussian
from skimage.morphology import dilation

from utils import torch_data2drawable
import config

def draw_all(axon_dets, structure_screen=None, which_dets='confident', 
             description='', t_y_x_slice=[None,None,None], dets_kwargs=None, 
             scnd_dets_kwargs=None, show=False, axon_subset=None, 
             save_single_tiles=False, animated=False, dpi=160, fps=6, 
             anim_fname_postfix='', draw_true_dets=False, draw_grid=False,  
             draw_scalebar=False, draw_axon_reconstructions=False,  
             draw_trg_paths=None, draw_brightened_bg=True):

    # slicing if passed
    tmin, tmax = t_y_x_slice[0] if t_y_x_slice[0] is not None else (0, axon_dets.dataset.sizet)
    ymin, ymax = t_y_x_slice[1] if t_y_x_slice[1] is not None else (0, axon_dets.dataset.sizey)
    xmin, xmax = t_y_x_slice[2] if t_y_x_slice[2] is not None else (0, axon_dets.dataset.sizex)
    yslice = slice(ymin, ymax)
    xslice = slice(xmin, xmax)
    tslice = slice(tmin, tmax)
    # which timepoints to draw
    timepoints = range(len(axon_dets))[tslice]

    # collect frame-wise artists in list
    if animated:
        anim_frames = []
        animated = plt.subplots(1, figsize=((xmax-xmin)/350, (ymax-ymin)/350))

    # for drawing axon reconstructions (A* paths)
    if draw_axon_reconstructions:
        axon_dets._reconstruct_axons()
        
    # iterate the timepoints
    for t in timepoints:
        # setup the frame parameters / data
        out = setup_frame_drawing(axon_dets, structure_screen, which_dets, t, 
                                  description, dets_kwargs, scnd_dets_kwargs, 
                                  axon_subset, yslice, xslice, ymin, xmin, 
                                  draw_brightened_bg, draw_true_dets, 
                                  draw_scalebar, draw_axon_reconstructions, 
                                  draw_grid, draw_trg_paths)
        frame_fname, frame_lbl, dets, frame_img, scnd_dets = out[:5]
        dets_kwargs, scnd_dets_kwargs, frame_args, dest_dir = out[5:]
        print(dets, )
        print(scnd_dets)
      
        frame_artists = draw_frame(img = frame_img, 
                                   dest_dir = axon_dets.dir,  
                                   dets = dets, 
                                   scnd_dets = scnd_dets, 
                                   fname = frame_fname, 
                                   lbl = frame_lbl, 
                                   show = show, 
                                   animation = animated, 
                                   boxs = axon_dets.axon_box_size,
                                   dets_kwargs = dets_kwargs, 
                                   scnd_dets_kwargs = scnd_dets_kwargs,
                                   first_tpoint = t==timepoints[0],
                                   **frame_args)
        if animated:
            anim_frames.append(frame_artists)

        # also save the single tile images, note that these are not NMS processed
        if save_single_tiles:
            if draw_grid:
                frame_args['gridsize'] = axon_dets.tilesize/axon_dets.Sx
            
            img_tiled, gt_dets_tiled = axon_dets.get_frame_and_truedets(t, unstitched=True)
            # for the current timepoints, iter non-empty tiles
            n_tiles = len(img_tiled)
            for tile_i in range(n_tiles):
                tile_fname = f'{frame_fname}_tile{tile_i:0>2}of{n_tiles:0>2}'
                # draw one tile
                draw_frame(img = img_tiled[tile_i][axon_dets.dataset.get_tcenter_idx()],
                           dets = gt_dets_tiled[tile_i],
                           scnd_dets = gt_dets_tiled[tile_i],
                           fname = tile_fname,
                           boxs = axon_dets.axon_box_size,
                           dest_dir = axon_dets.dir, 
                           show = show,
                           **frame_args,)
        
        print('Done.')

    if animated:
        anim_frames = [anim_frames[0]*2] + anim_frames + [anim_frames[-1]*2]
        ani = ArtistAnimation(animated[0], anim_frames, interval=400, blit=True, 
                              repeat_delay=200)
        if show:
            plt.show()
        print('encoding animation...', flush=True, end='')
        fname = (f'{dest_dir}/{axon_dets.dataset.name}_dets'
                 f'{anim_fname_postfix}.{config.VIDEO_FILETYPE}')
        ani.save(fname, writer=writers[config.VIDEO_ENCODER](fps=fps), dpi=dpi)
        print(f'{axon_dets.dataset.name} animation saved.')
    plt.close('all')
    print(' - Done.', flush=True)

def setup_frame_drawing(axon_dets, structure_screen, which_dets, t, description,
                        dets_kwargs, scnd_dets_kwargs, axon_subset, yslice, 
                        xslice, ymin, xmin, draw_brightened_bg, draw_true_dets, 
                        draw_scalebar, draw_axon_reconstructions, draw_grid, 
                        draw_trg_paths):
    # frame specific label
    frame_fname = f'Dataset{axon_dets.dataset.name}-frame{t:0>3}of{len(axon_dets):0>3}'
    if structure_screen is None:
        if not axon_dets.labelled:
            frame_lbl = f'{description} - {frame_fname}'
        else:
            wd = which_dets if which_dets != 'FP_FN' else 'confident'
            prc, rcl, F1 = axon_dets.get_detection_metrics(wd, t)
            frame_lbl = (f'{description} - Recall: {rcl}, Precision: {prc},'
                            f' F1: {F1} - {frame_fname}')
    else:
        frame_lbl = (f'{description} - {frame_fname} - DIV '
                        f'{structure_screen.get_DIV_point(t)}')
    print(frame_lbl, end='...', flush=True)

    # get the detections to draw
    dets = axon_dets.get_frame_dets(which_dets, t )
    frame_img, gt_dets = axon_dets.get_frame_and_truedets(t)
    frame_img = frame_img[:, yslice, xslice]

    # drawing confident, IDed or all detections
    if which_dets != 'FP_FN':
        scnd_dets = gt_dets if draw_true_dets else None
        dets_kwargs = dets_kwargs if dets_kwargs else config.PREDICTED_BOXES_KWARGS
        scnd_dets_kwargs = scnd_dets_kwargs if scnd_dets_kwargs else config.GROUNDTRUTH_BOXES_KWARGS
    # or only false positives and false negatives
    else:
        dets, scnd_dets = dets
        dets_kwargs = dets_kwargs if dets_kwargs else config.FP_BOXES_KWARGS
        scnd_dets_kwargs = scnd_dets_kwargs if scnd_dets_kwargs else config.FN_BOXES_KWARGS
    
    # adjust the detection coordinates to the image slice
    dets.anchor_y -= ymin
    dets.anchor_x -= xmin
    if scnd_dets is not None:
        scnd_dets.anchor_y -= ymin
        scnd_dets.anchor_x -= xmin
    
    # additional, optional setup for frame drawing
    frame_args = {}
    # draw the reconstructed axons A* paths
    if draw_axon_reconstructions:
        assert which_dets == 'IDed', 'Can only reconstruct IDed detections!'
        axon_reconstr = axon_dets.get_axon_reconstructions(t=t, ymin=ymin, ymax=ymax)
        frame_args['axon_reconstr'] = axon_reconstr
    
    # draw the path to the target location (A* path as well)
    if draw_trg_paths:
        assert structure_screen is not None, 'Pass StructureScreen object!'
        trg_paths = structure_screen.get_trg_path(t=t, ymin=ymin, ymax=ymax)
        frame_args['trg_paths'] = trg_paths
        ygoal, xgoal = structure_screen.structure_outputchannel_coo
        frame_args['target_coo'] = (ygoal-ymin, xgoal-xmin)
    
    # draw tile boundaries
    if draw_grid:
        frame_args['gridsize'] = axon_dets.tilesize
    
    if draw_scalebar:
        assert structure_screen is not None, 'Pass StructureScreen object!'
        frame_args['pixelsize'] = structure_screen.pixelsize
    
    if draw_brightened_bg:
        frame_args['mask'] = axon_dets.dataset.mask[t].todense()[yslice, xslice]

    # draw only a subset of axons 
    if axon_subset is not None:
        dets = dets.drop([ax for ax in dets.index if ax not in axon_subset])
        if axon_reconstr is not None:
            # filter list of paths here
            pass
        if trg_paths is not None:
            # filter list of paths here
            pass
    
    # draw one frame
    dest_dir = axon_dets.dir if structure_screen is None else structure_screen.dir
    return frame_fname, frame_lbl, dets, frame_img, scnd_dets, \
            dets_kwargs, scnd_dets_kwargs, frame_args, dest_dir

def draw_frame(img, dest_dir, dets=None, scnd_dets=None, fname='image', lbl='',
               show=False, animation=None, boxs=70, dets_kwargs=None, 
               scnd_dets_kwargs=None, first_tpoint=False, pixelsize=None, 
               gridsize=None, axon_reconstr=None, trg_paths=None, 
               target_coo=None, mask=None):

    im = torch_data2drawable(img)
    height, width = im.shape[:-1]

    # Create figure and axes, do basic setup if no animation is created
    artists = []
    if not animation:
        fig, ax = plt.subplots(1, figsize=(width/350, height/350))
    else:
        fig, ax = animation
    
    fig.subplots_adjust(wspace=0, hspace=0, top=1, bottom=0, left=0, right=1)
    ax.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
    ax.axis('off')
    ax.set_facecolor('k')
    fig.set_facecolor('k')

    # draw label on top right
    artists.append(ax.text(width-140, 50, lbl, ha='right', va='top',
                        fontsize=height/180, color=config.DARK_GRAY))

    # draw yolo lines, only at t0 if animation 
    if (gridsize and not animation) or (gridsize and animation and first_tpoint):
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)
        at_x = [gridsize*i for i in range(width//gridsize+1)]
        at_y = [gridsize*i for i in range(width//gridsize+1)]
        ax.vlines(at_x, 0, height, color='white', linewidth=1, alpha=.15)
        ax.hlines(at_y, 0, width, color='white', linewidth=1, alpha=.15)

    # draw scalebar in top right
    if pixelsize:
        bar_length = 200/pixelsize #200 um bar
        bar_width, pad = 10, 150
        xstart, xstop = width-pad, width-pad-bar_length
        ystart, ystop = pad, pad
        ax.plot((xstart,xstop), (ystart,ystop), linewidth=bar_width, color=config.DARK_GRAY)
        artists.append(ax.text(width-225, 136, '200 um', ha='right', va='top',
                       fontsize=height/180, color='k'))

    # draw the bounding boxes and potentially reconstructions
    paths_canvas_rgba = np.zeros((height, width, 4))
    artists.extend(draw_detections(dets, dets_kwargs, annotate=True, ax=ax, 
                                   boxs=boxs, axon_reconstr=axon_reconstr, 
                                   trg_paths=trg_paths, 
                                   paths_canvas_rgba=paths_canvas_rgba))
    artists.extend(draw_detections(scnd_dets, scnd_dets_kwargs, ax=ax, boxs=boxs))
    
    # brighten the channel background 
    alpha = np.ones((height, width), float)
    if mask is not None:
        # add white channelsmask below axon image
        artists.append(ax.imshow(mask, vmin=0, vmax=1, cmap='gray'))
        # increase the alpha of the axon channel where there is litle signal
        faint_red_mask = (im[:,:,0] <= .12) & (im[:,:,0] > 0)
        alpha[faint_red_mask] = .1
        # smooth mask
        alpha = gaussian(alpha, 5)
    im = np.concatenate([im, alpha[:,:,np.newaxis]], -1)
    
    # finally draw axon growth image
    artists.append(ax.imshow(im, vmin=0, vmax=1))
    
    if animation:
        return artists
    if show:
        plt.show()
    if dest_dir:
        fig.savefig(f'{dest_dir}/{fname}.{config.FIGURE_FILETYPE}')
        plt.close(fig) 

def draw_detections(dets, kwargs, ax, boxs, annotate=False, axon_reconstr=None, 
                    trg_paths=None, paths_canvas_rgba=None):
    artists = []
    for axon_id, (conf, x, y) in dets.iterrows():
        conf = min(1, conf)
        if kwargs.get('edgecolor') == 'hsv':
            ax_id_col = plt.cm.get_cmap('hsv', 20)(int(axon_id[-3:])%20)
        else:
            ax_id_col = kwargs.get('edgecolor')
        
        if annotate:
            artists.append(ax.text(x-boxs/2, y-boxs/1.5, axon_id.replace('on_',''), 
                                fontsize=5.5, color=ax_id_col))
        
        # draw the A* star paths between associated detections
        if axon_reconstr is not None and axon_id in axon_reconstr.columns.unique(0):
            draw_y = axon_reconstr[axon_id].Y.unstack().dropna().astype(int)
            draw_x = axon_reconstr[axon_id].X.unstack().dropna().astype(int)
            paths_canvas_rgba[draw_y, draw_x] = ax_id_col
        
        # draw the A* star paths to the target
        if trg_paths and axon_id in trg_paths and i==0:
            paths_canvas_rgba[trg_paths[axon_id]] = (.85,.85,.85,1)

        # draw detection box
        draw_kwargs = kwargs.copy()
        draw_kwargs.pop('edgecolor')
        axon_box = Rectangle((x-boxs/2, y-boxs/2), boxs, boxs, edgecolor=ax_id_col, 
                              **draw_kwargs)
        ax.add_patch(axon_box)
        artists.append(axon_box)

    if axon_reconstr is not None and not axon_reconstr.empty or trg_paths:
        # process image without alpha, make line thicker, blur
        selem = np.stack([np.zeros((6,6)),np.ones((6,6)),np.zeros((6,6))],-1)
        axons_img_rgb = dilation(paths_canvas_rgba[:,:,:-1], selem=selem)
        axons_img_rgb = gaussian(axons_img_rgb, 1, preserve_range=True)

        # get the alpha channel back
        alpha = np.maximum(0, (axons_img_rgb.max(-1)))
        paths_canvas_rgba = np.append(axons_img_rgb, alpha[:,:,np.newaxis], -1)
        im2_artist = ax.imshow(paths_canvas_rgba)
        artists.append(im2_artist)
    return artists