import pickle

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
from matplotlib.animation import ArtistAnimation, writers

from skimage.filters import gaussian
from skimage.morphology import dilation
from skimage import filters
from skimage import morphology

from screen_plotting import plot_target_distance_over_time

import config

def draw_all(axon_dets, filename, dest_dir=None, notes='', dt=None, show=False, 
             filter2FP_FN=False, save_single_tiles=False, animated=False, draw_grid=False,
             hide_det1=False, hide_det2=False, use_IDed_dets=False, anim_fname_postfix='',
             draw_axons=None, which_axons=None, t_y_x_slice=[None,None,None], 
             draw_trg_distances=None, insert_distance_plot=None, dpi=160, fps=6,
             **kwargs):

    
    # slicing if passed
    tmin, tmax = t_y_x_slice[0] if t_y_x_slice[0] is not None else (0, axon_dets.dataset.sizet)
    ymin, ymax = t_y_x_slice[1] if t_y_x_slice[1] is not None else (0, axon_dets.dataset.sizey)
    xmin, xmax = t_y_x_slice[2] if t_y_x_slice[2] is not None else (0, axon_dets.dataset.sizex)
    tslice = slice(tmin, tmax)
    yslice = slice(ymin, ymax)
    xslice = slice(xmin, xmax)

    if not dest_dir:
        dest_dir = axon_dets.dir
    
    if animated:
        anim_frames = []
        # animated = plt.subplots(1, figsize=((xmax-xmin)/350, (ymax-ymin)/350))
        
        pad  = ('right', 2.3)
        video_xsize = (xmax-xmin)/350
        video_ysize = (ymax-ymin)/350
        fig = plt.figure(figsize=(video_xsize +pad[1], video_ysize))
        ax = fig.add_axes([0,0, video_xsize/ (video_xsize+pad[1]), 1])
        animated = fig, ax

    if insert_distance_plot:
        ygoal, xgoal = insert_distance_plot.structure_outputchannel_coo
        ygoal, xgoal = (ygoal-ymin, xgoal-xmin)
        
        ax_bottom, ax_left = .13, .66
        # ax_bottom, ax_left = .68, .45
        ax_width, ax_height = .37,.23
        distplot_ax = fig.add_axes([ax_left, ax_bottom, ax_width, ax_height])
    else:
        ygoal, xgoal, distplot_ax = None, None, None
    

    which_det = 'confident' if not use_IDed_dets else 'ID_assigned'
    first_tpoint = True
    # iterate the timepoints
    for t in range(len(axon_dets))[tslice]:
        fname = filename.replace('---', f'{t:0>3}') 
        prc, rcl, F1 = axon_dets.get_detection_metrics(t, which_det=which_det)
        lbl = f'{notes} - drawing frame, {fname} - Recall: {rcl}, Precision: {prc}, F1: {F1}'
        print(lbl, end='...', flush=True)
        
        # label with timepoint instead
        if dt is not None:
            if t == range(len(axon_dets))[tslice][0]:
                time = 0
            else:
                time += dt
            lbl = f'{time//(24*60)} days, {(time//60)%24} hours'

        img_tiled, img, tiled_true_det, image_true_det = axon_dets.get_det_image_and_target(t)
        img = img[:, yslice, xslice]
        det2 = image_true_det
        
        # which detections exactly to draw
        if not use_IDed_dets:
            if not filter2FP_FN:
                det1 = axon_dets.get_confident_det(t)
            
            # drawing FP and FN right now only for non-IDed data (confident set of dets)
            else:
                FP_dets, FN_dets = axon_dets.get_FP_FN_detections(t)
                det1, det2 = FP_dets, FN_dets
                kwargs['det1_default_col'] = '#e34d20'  # red-ish orange for false pos
                kwargs['det2_default_col'] = '#01ff6b'  # green for false neg
                # ensure that id coloring is set to False
                kwargs['color_det2_ids'] = False 
                kwargs['color_det1_ids'] = False 
        else:
            det1 = axon_dets.get_IDed_det(t)
            print(f'n={len(det1)}, lowest conf={det1.iloc[-1:,:].conf}.')
        
        # slice down list of axons to draw
        det1_new = []
        for axon_id, (conf, x, y) in det1.iterrows():
            if not (ymin <= y < ymax) or not (xmin <= x < xmax):
                conf = -1
            y -= ymin
            x -= xmin
            det1_new.append(pd.Series((conf, x, y), name=axon_id))
        det1 = pd.concat(det1_new, axis=1).T
        det1.columns = ['conf', 'anchor_x', 'anchor_y']

        if draw_trg_distances:
            target_paths = axon_dets._compute_dets_path_to_target(cache='from')
            trg_paths = []
            for path in target_paths[t]:
                arr = np.zeros(path.shape, bool)
                arr[path.row[:-35], path.col[:-35]] = True
                trg_paths.append(arr[ymin:ymax, xmin:xmax])
            # trg_paths = [np.ones for path in target_paths[t]]
            # trg_paths = [path.toarray()[ymin:ymax, xmin:xmax] for path in target_paths[t]]
            trg_paths = dict(zip(det1.index, trg_paths))
        else:
            trg_paths = None

        # draw only a subset of axons 
        if which_axons is not None:
            det1 = det1.drop([ax for ax in det1.index if ax not in which_axons])

        # draw the reconstructed axons A* paths
        if use_IDed_dets and draw_axons is not None:
            if t == 0:
                draw_axons = pd.DataFrame([])
            else:
                axons_at_t = axon_dets.get_axon_reconstructions(t)
                # if y min is 0, the coordinates need to be shifted to fit the new img shape
                axons_at_t.loc[:,(slice(None),'Y')] -= ymin
                axons_at_t.loc[:,(slice(None),'X')] -= xmin
                axons_at_t = axons_at_t.loc[:,(det1.index.values, slice(None), slice(None))]
                draw_axons = pd.concat([draw_axons, axons_at_t], axis=1)
        
        if hide_det1:
            det1 = None
        if hide_det2:
            det2 = None
        else:
            # if true label exists
            det2 = det2[(det2.anchor_y>=ymin).values & (det2.anchor_y<ymax).values]
            det2 = det2[(det2.anchor_x>=xmin).values & (det2.anchor_x<xmax).values]
        
        # draw stitched frame
        if draw_grid:
            draw_grid = axon_dets.tilesize
        frame_artists = draw_frame(img, det1, det2, animation=animated, 
                                   dest_dir=dest_dir, draw_grid=draw_grid, 
                                   fname=fname, lbl=lbl, show=show, 
                                   draw_axons=draw_axons, 
                                   trg_paths=trg_paths,
                                   first_tpoint=first_tpoint,
                                   structure_outputchannel_coo=(xgoal, ygoal),
                                   insert_distance_plot=insert_distance_plot,
                                   distplot_ax=distplot_ax,
                                   mask=axon_dets.dataset.mask[t].todense()[yslice, xslice],
                                   **kwargs)
        if animated:
            anim_frames.append(frame_artists)
        first_tpoint = False

        # also save the single tile images, note that these are not NMS processed
        if save_single_tiles:
            if draw_grid:
                draw_grid = axon_dets.tilesize/axon_dets.Sx

            # for the current timepoints, iter non-empty tiles
            n_tiles = len(img_tiled)
            tiled_det = axon_dets.get_confident_det(t, get_tiled_not_image=True)
            for tile_i in range(n_tiles):
                tile = img_tiled[tile_i][axon_dets.dataset.get_tcenter_idx()]
                target_det = tiled_true_det[tile_i]
                draw_frame(tile, None, target_det, 
                           draw_grid=draw_grid, boxs=axon_dets.axon_box_size,
                           dest_dir=dest_dir, show=show, 
                           fname=f'{fname}_tile{tile_i:0>2}|{n_tiles:0>2}')
        print('Done.')

    if animated:
        anim_frames = [anim_frames[0]*2] + anim_frames + [anim_frames[-1]*2]
        ani = ArtistAnimation(animated[0], anim_frames, interval=400, blit=True, 
                              repeat_delay=200)
        if show:
            plt.show()
        print('encoding animation...', flush=True, end='')
        ani.save(f'{dest_dir}/{axon_dets.dataset.name}_assoc_dets{anim_fname_postfix}.{config.VIDEO_FILETYPE}', 
                 writer=writers[config.VIDEO_ENCODER](fps=fps), dpi=dpi)
        print(f'{axon_dets.dataset.name} animation saved.')
    plt.close('all')
    print(' - Done.', flush=True)

def draw_frame(image, det1=None, det2=None, fname='image', dest_dir=None,  lbl='',
               animation=None, boxs=70, draw_grid=None, show=False, color_det1_ids=False, 
               color_det2_ids=True, det1_default_col='w', det2_default_col='w', 
               draw_axons=None, pixelsize=None, draw_scalebar=None, 
               first_tpoint=False, structure_outputchannel_coo=None,
               trg_paths=None, insert_distance_plot=None, distplot_ax=None, 
               mask=None):
    im = np.array(image.detach().cpu())
    if im.shape[0] <= 3:
        im = np.moveaxis(im, 0, 2)
    height, width, cchannels = im.shape
    
    # make everything RGB, fill if necessary
    emptychannel = np.zeros([height,width])
    if cchannels == 2:
        g, b = im[:,:,0], im[:,:,1]
        im = np.stack([emptychannel, g, b], axis=2)
    elif cchannels == 1:
        r = im[:,:,0]
        im = np.stack([r, emptychannel, emptychannel], axis=-1)
    # matplotlib likes float RGB images in range [0,1]
    min_val = im[im>0].min() if im.any().any() else 0
    # im = np.where(im>0, im-min_val, 0)  # shift distribution to 0
    im = np.where(im>1, 1, im)  # ceil to 1

    # Create figure and axes, do basic setup
    if not animation:
        fig, ax = plt.subplots(1, figsize=(width/350, height/350), facecolor='#242424')
        fig.subplots_adjust(wspace=0, hspace=0, top=1, bottom=0, 
                            left=0, right=1)
    else:
        fig, ax = animation
    
    ax.axis('off')
    ax.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
    ax.set_facecolor('k')
    fig.set_facecolor('k')

    # draw yolo lines
    if (draw_grid and not animation) or (draw_grid and animation and first_tpoint):
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)
        gridsize = round(draw_grid)
        at_x = [gridsize*i for i in range(width//gridsize+1)]
        at_y = [gridsize*i for i in range(width//gridsize+1)]
        ax.vlines(at_x, 0, height, color='white', linewidth=1, alpha=.15)
        ax.hlines(at_y, 0, width, color='white', linewidth=1, alpha=.15)

    text_artists = []
    # draw scalebar
    if draw_scalebar and pixelsize:
        # 200 um bar
        bar_length = 200/pixelsize
        bar_width = 10
        pad = 150
        xstart, xstop = width-pad, width-pad-bar_length
        ystart, ystop = pad, pad
        ax.plot((xstart,xstop), (ystart,ystop), linewidth=bar_width, color=config.DARK_GRAY)
        text_artists.append(ax.text(width-225, 136, '200 um', ha='right', va='top',
                        fontsize=height/200, color='k'))
    
    artists = []
    if insert_distance_plot is not None:
        # ax_bottom, ax_left = .13, .7
        # # ax_bottom, ax_left = .68, .45
        # ax_width, ax_height = .4,.23
        # insert_ax = fig.add_axes([ax_left, ax_bottom, ax_width, ax_height])

        ss = insert_distance_plot
        ax.add_patch(Circle(structure_outputchannel_coo, 15, color=config.LIGHT_GRAY))
        ax.add_patch(Circle(structure_outputchannel_coo, 2, color='k'))
        ax.add_patch(Rectangle((300, 1200), 800, 530, color='k'))
        t = int(fname[fname.rfind('of')-3:fname.rfind('of')]) if 'of' in fname else None
        line_artists = plot_target_distance_over_time(ss, draw_until_t=t, draw_until_t_ax=distplot_ax, 
                                       subtr_init_dist=False, count_growth_dir=False, gray_lines=False,
                                       text_scaler=.5)
        artists.extend(line_artists)
    # draw bounding boxes
    rectangles = []
    # draw axons if passed
    axons_img_rgba = np.zeros((height, width, 4))
    for i, det in enumerate((det1, det2)):
        kwargs = {'facecolor':'none'}
        
        if det is not None:
            for axon_id, (conf, x, y) in det.iterrows():
                if conf == -1:
                    continue
                conf = min(1, conf)
                if i == 0:
                    kwargs.update({'alpha':conf, 'linestyle':'dashed', 'linewidth':'1', 'edgecolor':det1_default_col})
                elif i == 1:
                    kwargs.update({'alpha':.75, 'linestyle':'solid', 'linewidth':'1.2', 'edgecolor':det2_default_col})
                
                if (i==0 and color_det1_ids) or (i==1 and color_det2_ids):
                    ax_id_col = plt.cm.get_cmap('hsv', 20)(int(axon_id[-3:])%20)
                    text_artists.append(ax.text(x-boxs/2, y-boxs/1.5, axon_id.replace('on_',''), 
                                        fontsize=5.5, color=ax_id_col))
                    kwargs.update({'edgecolor': ax_id_col})
                
                    # draw the A* star paths between associated detections
                    if draw_axons is not None and axon_id in draw_axons.columns.unique(0) and i==0:
                        draw_y = draw_axons[axon_id].Y.unstack().dropna().astype(int)
                        draw_x = draw_axons[axon_id].X.unstack().dropna().astype(int)
                        axons_img_rgba[draw_y, draw_x] = ax_id_col
                    
                    # draw the A* star paths to the target
                    if trg_paths and axon_id in trg_paths and i==0:
                        axons_img_rgba[trg_paths[axon_id]] = (.85,.85,.85,1)

                # draw detection box
                axon_box = Rectangle((x-boxs/2, y-boxs/2), boxs, boxs, **kwargs)
                rectangles.append(axon_box)
    [ax.add_patch(rec) for rec in rectangles]

    mask_artist = ax.imshow(mask, vmin=0, vmax=1, cmap='gray')
    
    alpha = np.ones((height, width), float)
    faint_red_mask = (im[:,:,0] <= .12) & (im[:,:,0] > 0)
    alpha[faint_red_mask] = .1
    alpha = gaussian(alpha, 5)
    
    im = np.concatenate([im, alpha[:,:,np.newaxis]], -1)
    im_artist = ax.imshow(im, vmin=0, vmax=1)
    text_artists.append(ax.text(width-140, 50, lbl, ha='right', va='top',
                        fontsize=height/200, color=config.DARK_GRAY))
    artists.extend([mask_artist, im_artist,  *text_artists, *rectangles])

    if draw_axons is not None and not draw_axons.empty or trg_paths:
        # process image without alpha, make line thicker, blur
        selem = np.stack([np.zeros((6,6)),np.ones((6,6)),np.zeros((6,6))],-1)
        axons_img_rgb = dilation(axons_img_rgba[:,:,:-1], selem=selem)
        axons_img_rgb = gaussian(axons_img_rgb, 1, preserve_range=True)

        # get the alpha channel back
        alpha = np.maximum(0, (axons_img_rgb.max(-1)))
        axons_img_rgba = np.append(axons_img_rgb, alpha[:,:,np.newaxis], -1)
        im2_artist = ax.imshow(axons_img_rgba)
        artists.append(im2_artist)
        
    if animation:
        return artists
    if show:
        plt.show()
    if dest_dir:
        fig.savefig(f'{dest_dir}/{fname}.{config.FIGURE_FILETYPE}')
        plt.close(fig) 
