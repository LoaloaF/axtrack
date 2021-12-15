import pickle
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.patches import Polygon

from matplotlib.lines import Line2D
from matplotlib.animation import ArtistAnimation, writers
from matplotlib.colors import ListedColormap


from skimage.filters import gaussian
from skimage.morphology import dilation
from skimage.morphology import square

import pandas as pd
import numpy as np

from config import TRAIN_Ps, TEST_Ps, VIDEO_ENCODER
import config

from AxonDetections import AxonDetections

def plot_preprocessed_input_data(preporc_dat_csv, notes='', dest_dir=None, show=False, motion_plots=False, fname_prefix=''):
    data = pd.read_csv(preporc_dat_csv, header=[0,1,2], index_col=0)

    fig = plt.figure(figsize=config.LARGE_FIGSIZE)
    fig.subplots_adjust(hspace=.4, wspace=.05, top=.95, bottom=.1, left= .07, right=.97)
    widths = [1, 1, 0.3, 1, 1]
    heights = [1, 1, 1] if motion_plots else [1, 1]
    spec = fig.add_gridspec(ncols=5, nrows=len(heights), width_ratios=widths,
                            height_ratios=heights)
    axes = [fig.add_subplot(spec[row,col]) for row in range(len(heights)) for col in (0,1,3,4)]
    axes[0].text(.48,.03, 'pixel intensity', transform=fig.transFigure, fontsize=config.FONTS)
    axes[0].text(.02,.5, 'density', transform=fig.transFigure, rotation=90, fontsize=config.FONTS)
    # fig.suptitle(notes)

    which_preproc_steps = data.columns.unique(1)
    
    # when standardized label differs....
    if len(which_preproc_steps) == 7:
        which_preproc_steps = which_preproc_steps[:-1]
    for i, which_preproc in enumerate(which_preproc_steps):
        if 'Motion' in which_preproc and not motion_plots:
            continue
        for j, which_timpoint in enumerate(('t_0', 't_-1')):
            ax = axes[i*2+j]

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(color='#d1d1cf', zorder=0)
            ax.tick_params(axis='both', labelsize=config.SMALL_FONTS)
            which_preproc_title = which_preproc if not which_preproc.startswith('Standardized') else 'Standardized'
            tit = which_timpoint if which_timpoint == 't_0' else 't_N'
            ax.set_title(f'{tit}-data: {which_preproc_title}', fontsize=config.FONTS)
            if j:
                ax.tick_params(labelleft=False)
            
            for dat_name in data.columns.unique(0):
                # when standardized label differs....
                if which_preproc not in data[dat_name].columns.unique(0):
                    which_preproc = data[dat_name].columns.unique(0)[i]
                dat = data[(dat_name, which_preproc, which_timpoint)]
                
                if which_preproc.startswith('Standardized'):
                    args = {'bins': 2**9, }
                    ax.set_ylim((0, .95))
                    ax.set_xlim((-.06, 1.15))
                elif 'Motion' in which_preproc:
                    args = {'bins': 2**10, }
                    ax.set_ylim((0, 0.1))
                    ax.set_xlim((-.1, 14))
                else:
                    args = {'bins': 2**9} # image data is saved in 12bit format (0-4095)
                    ax.set_ylim((0, 65))
                    ax.set_xlim((-.001, .014))
                
                lbl = f'{(dat==0).sum()/len(dat):.2f}'

                if i == 0:
                    name = 'Train' if dat_name == 'train' else 'Infer'
                    lbl = f'{name}, sparsity {lbl}'
                ax.hist(dat, alpha=.5, density=True, zorder=3,
                                label=lbl, **args)
            ax.legend(fontsize=config.SMALL_FONTS, loc='upper right', bbox_to_anchor=(1.03, 1.03))
    if show:
        plt.show()
    if dest_dir is not None:
        fname = f'{dest_dir}/{fname_prefix}preprocessed_data.{config.FIGURE_FILETYPE}'
        fig.savefig(fname)
    plt.close()


def plot_training_process(training_data_files, draw_detailed_loss=False, 
                          draw_best_thresholds=False, show=False, dest_dir=None):
    # create the figure, share epoch (x axis)
    fig, axes = plt.subplots(2,4, figsize=config.LARGE_FIGSIZE, sharex=True)
    fig.subplots_adjust(wspace=.03, left=.05, right=.97, bottom=.08, top=.95)
    axes = axes.flatten()
    
    # iterate over different runs
    legend_elements  = []
    file_name_apdx = '_'
    for exp_run_id, (loss_file, mtrc_file, params) in training_data_files.items():
        # read the respective loss and metrics file
        loss = pd.read_pickle(loss_file)
        loss_x_epochs = loss.columns.unique(0)
        metric = pd.read_pickle(mtrc_file)
        metric_x_epochs = metric.columns.unique(0)
        
        # iterate the different plots/ value groups to plot
        for i, ax in enumerate(axes):
            ax.tick_params(labelsize=config.SMALL_FONTS)
            if i not in [0,4]:
                ax.tick_params(labelleft=False)

            # loss plots: xy anchor, box loss, no-box loss, summed loss
            if i<4:
                x_epochs = loss_x_epochs
                ax.set_ylim(0,10)

                # get one specific loss
                name = loss.index[i]
                trn_loss = loss.loc[name, (slice(None), 'train',)].dropna().droplevel(1)
                tst_loss = loss.loc[name, (slice(None), 'test',)].dropna().droplevel(1)
                # average over batch dim + exponential smoothing 
                trn_data = trn_loss.groupby('epoch').mean().ewm(span=25).mean()
                tst_data = tst_loss.groupby('epoch').mean().ewm(span=25).mean()

            # metrics plots (precision, recall, F1)
            elif i <7:
                x_epochs = metric_x_epochs
                ax.set_ylim(0.25,1.05)
                ax.set_yticks(np.arange(0.4,1.01, .1))
                ax.set_xlabel('Epoch', fontsize=config.FONTS)
                
                # get one specific metrics
                name = metric.index[i-4]
                trn_mtrc = metric.loc[name, (slice(None), 'train')].droplevel((1,2))
                tst_mtrc = metric.loc[name, (slice(None), 'test')].droplevel((1,2))
                # smooth 
                trn_data = trn_mtrc.ewm(span=3).mean()
                tst_data = tst_mtrc.ewm(span=3).mean()

            # last plot (8/8) has no data, legend is place here
            elif i == 7:
                ax.axis('off')
                # save the run's identifying color to make legend in the end
                legend_elements.extend([
                    Line2D([0], [0], label=f'Train', color=trn_col,
                            **dict(TRAIN_Ps, **{'linewidth':3})),
                    Line2D([0], [0], label=f'Test', color=tst_col,
                            **dict(TEST_Ps, **{'linewidth':3}),),])
                # for filename later
                if len(training_data_files) >1:
                    file_name_apdx += exp_run_id[:5]
                continue

            # option to draw the selected best thresholds of F1
            if i == 5 and draw_best_thresholds:
                thrs = metric.loc[:,(slice(None),'test')].columns.get_level_values(2).values
                plt.scatter(x_epochs, thrs, s=.5, color='k', alpha=.3)
            
            # general args for every axis
            ax.grid(True)
            ax.set_title(name, fontsize=config.FONTS)

            # finally draw the train and test data defined above
            ls = ax.plot(x_epochs, trn_data, **TRAIN_Ps)
            trn_col = ls[-1].get_color()
            ls = ax.plot(x_epochs, tst_data, color=trn_col, **TEST_Ps)
            tst_col = ls[-1].get_color()

    # draw legend at last exes
    ax.legend(handles=legend_elements, bbox_to_anchor=(.05, .95), ncol=1,
                    loc='upper left', fontsize=config.FONTS)
    
    if show:
        plt.show()
    if dest_dir is not None:
        fname = f'{dest_dir}/training{file_name_apdx}.{config.FIGURE_FILETYPE}'
        fig.savefig(fname, dpi=450)

def draw_frame(image, det1=None, det2=None, fname='image', dest_dir=None,  lbl='',
               animation=None, boxs=70, draw_grid=None, show=False, color_det1_ids=False, 
               color_det2_ids=True, det1_default_col='w', det2_default_col='w', 
               draw_axons=None, pixelsize=None, draw_scalebar=None):
    # make sure image data has the right format: HxWxC
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
    else:
        fig, ax = animation
    fig.subplots_adjust(wspace=0, hspace=0, top=.99, bottom=.01, 
                        left=.01, right=.99)
    ax.axis('off')
    ax.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
    ax.set_facecolor('#242424')

    # draw yolo lines
    if (draw_grid and not animation) or (draw_grid and animation and int(fname[fname.rfind('of')-3:fname.rfind('of')]) == 0):
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)
        gridsize = round(draw_grid)
        at_x = [gridsize*i for i in range(width//gridsize+1)]
        at_y = [gridsize*i for i in range(width//gridsize+1)]
        ax.vlines(at_x, 0, height, color='white', linewidth=.5, alpha=.3)
        ax.hlines(at_y, 0, width, color='white', linewidth=.5, alpha=.3)

    # draw scalebar
    if draw_scalebar and pixelsize:
        # 200 um bar
        bar_length = 200/pixelsize
        bar_width = 10
        pad = 100
        xstart, xstop = width-pad, width-pad-bar_length
        ystart, ystop = pad, pad
        ax.plot((xstart,xstop), (ystart,ystop), linewidth=bar_width, color=config.DARK_GRAY)

    # draw bounding boxes
    rectangles = []
    text_artists = []
    # draw axons if passed
    axons_img_rgba = np.zeros((height, width, 4))
    for i, det in enumerate((det1, det2)):
        kwargs = {'facecolor':'none'}
        
        if det is not None:
            for axon_id, (conf, x, y) in det.iterrows():
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

                # draw detection box
                axon_box = Rectangle((x-boxs/2, y-boxs/2), boxs, boxs, **kwargs)
                rectangles.append(axon_box)
    [ax.add_patch(rec) for rec in rectangles]

    im_artist = ax.imshow(im, vmin=0, vmax=1)
    text_artists.append(ax.text(.05, .95, lbl, transform=fig.transFigure, 
                        fontsize=height/200, color='w'))
    artists = [im_artist, *text_artists, *rectangles]

    if draw_axons is not None and not draw_axons.empty:
        # process image without alpha, make line thicker, blur
        selem = np.stack([np.zeros((14,14)),np.ones((14,14)),np.zeros((14,14))],-1)
        axons_img_rgb = dilation(axons_img_rgba[:,:,:-1], selem=selem)
        axons_img_rgb = gaussian(axons_img_rgb, 2, preserve_range=True)

        # get the alpha channel back
        alpha = np.maximum(0, (axons_img_rgb.max(-1)))
        axons_img_rgba = np.append(axons_img_rgb, alpha[:,:,np.newaxis], -1)
        im2_artist = ax.imshow(axons_img_rgba)
        artists = [im2_artist, *artists]
        
    if animation:
        return artists
    if show:
        plt.show()
    if dest_dir:
        fig.savefig(f'{dest_dir}/{fname}.{config.FIGURE_FILETYPE}')
        plt.close(fig) 


def plot_prc_rcl(metrics_files, dest_dir=None, show=None):
    fig, axes = plt.subplots(1, 2, figsize=config.MEDIUM_FIGSIZE)
    fig.subplots_adjust(bottom=.01, wspace=.3, right=.95)
    ticks = np.arange(0.5, .91, 0.2)
    lims = (.3, 1.04)

    # setup both axes
    for i, ax in enumerate(axes):
        ax.set_ylim(*lims) 
        ax.set_yticks(ticks)
        ax.set_yticklabels([f'{t:.1f}' for t in ticks], fontsize=config.SMALL_FONTS)

        ax.set_xlim(*lims)
        ax.set_xticks(ticks)
        ax.set_xticklabels([f'{t:.1f}' for t in ticks], fontsize=config.SMALL_FONTS)

        ax.set_aspect('equal')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid()
        if i == 0:
            ax.set_xlabel('Precision', fontsize=config.FONTS)
            ax.set_ylabel('Recall', fontsize=config.FONTS)
        else:
            ax.set_xlabel('Thresholds', fontsize=config.FONTS)
            ax.set_ylabel('F1', fontsize=config.FONTS)

    # iterate different files
    legend_elements, runs = [], ''
    for i, (name, files) in enumerate(metrics_files.items()):
        print(name)
        runs += name[:5]

        # not only the metrics file for the epoch is passed, but also the 15  
        # epochs before and after. the average over these 30 is computed below
        metrics = [pd.read_pickle(file) for file in files]
        # print(metrics)
        metric = np.stack([df.values for df in metrics if not df.empty]).mean(0)
        metric = pd.DataFrame(metric, index=metrics[0].index, 
                              columns=metrics[0].columns.droplevel(0))
        thrs = metric.columns.unique(1).values.astype(float)

        # unpack data
        trn_prc, trn_rcl, trn_F1 = metric.train.values
        tst_prc, tst_rcl, tst_F1  = metric.test.values
        best_thr = np.argsort(tst_F1)[-1]

        # precision recall, train and test
        ls = axes[0].plot(trn_prc, trn_rcl, **TRAIN_Ps)
        col = ls[-1].get_color()
        # col = None
        axes[0].plot(tst_prc, tst_rcl, color=col, **TEST_Ps)
        axes[0].scatter(tst_prc[best_thr], tst_rcl[best_thr], color=col, s=15, zorder=20)
        
        # F1 plot
        axes[1].plot(thrs, trn_F1, color=col, **TRAIN_Ps)
        axes[1].plot(thrs, tst_F1, color=col, **TEST_Ps)
        axes[1].scatter(thrs[best_thr], tst_F1[best_thr], color=col, s=15, zorder=20)

        # legend 
        legend_elements.extend([
            Line2D([0], [0], label=f'Train - {name[12:]}', color=col, **dict(TRAIN_Ps, **{'linewidth':4})),
            Line2D([0], [0], label=f'Test', color=col, **dict(TEST_Ps, **{'linewidth':3}))
            ])
    axes[0].legend(handles=legend_elements, bbox_to_anchor=(-.2, 1.05), ncol=2,
                   loc='lower left', fontsize=config.SMALL_FONTS)
    if show:
        plt.show()
    if dest_dir:
        fig.savefig(f'{dest_dir}/prc_rcl_{runs}.{config.FIGURE_FILETYPE}')

# def draw_axon_reconstruction(reconstructions, img_shape, color='green'):
#     draw_y = reconstructions.loc[:, (slice(None),'Y')].unstack().dropna().astype(int)
#     draw_x = reconstructions.loc[:, (slice(None),'X')].unstack().dropna().astype(int)

#     drawn_axons = np.zeros((img_shape), float)
#     drawn_axons[draw_y, draw_x] = 1
#     drawn_axons = dilation(drawn_axons, square(5))
#     drawn_axons = gaussian(drawn_axons, 3, preserve_range=True)

#     drawn_axons_rgba = np.stack([drawn_axons, *[np.zeros_like(drawn_axons)]*2, 
#                                  drawn_axons], -1)
#     return drawn_axons_rgba

def draw_all(axon_dets, filename, dest_dir=None, notes='', dt=None, show=False, 
             filter2FP_FN=False, save_single_tiles=False, animated=False, 
             hide_det1=False, hide_det2=False, use_IDed_dets=False, anim_fname_postfix='',
             draw_axons=None, which_axons=None, t_y_x_slice=[None,None,None], **kwargs):
    
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
        animated = plt.subplots(1, figsize=(xmax/350, ymax/350), facecolor='#242424')
    which_det = 'confident' if not use_IDed_dets else 'ID_assigned'
    # iterate the timepoints
    for t in range(len(axon_dets))[tslice]:
        fname = filename.replace('---', f'{t:0>3}') 
        prc, rcl, F1 = axon_dets.get_detection_metrics(t, which_det=which_det)
        lbl = f'{notes} - {fname} - Recall: {rcl}, Precision: {prc}, F1: {F1}'
        print(lbl, end='...', flush=True)
        
        # label with timepoint instead
        if dt is not None:
            if t == 0:
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
        det1 = det1[(det1.anchor_y>=ymin).values & (det1.anchor_y<ymax).values]
        det1 = det1[(det1.anchor_x>=xmin).values & (det1.anchor_x<xmax).values]
        # if y min is 0, the coordinates need to be shifted to fit the new img shape
        det1.anchor_y -= ymin
        det1.anchor_x -= xmin
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
        frame_artists = draw_frame(img, det1, det2, animation=animated, 
                                   dest_dir=dest_dir, draw_grid=axon_dets.tilesize, 
                                   fname=fname, lbl=lbl, show=show, 
                                   draw_axons=draw_axons, **kwargs)
        if animated:
            anim_frames.append(frame_artists)

        # also save the single tile images, note that these are not NMS processed
        if save_single_tiles:
            # for the current timepoints, iter non-empty tiles
            n_tiles = len(img_tiled)
            tiled_det = axon_dets.get_confident_det(t, get_tiled_not_image=True)
            for tile_i in range(n_tiles):
                tile = img_tiled[tile_i][axon_dets.dataset.get_tcenter_idx()]
                target_det = tiled_true_det[tile_i]
                draw_frame(tile, None, None, 
                           draw_grid=axon_dets.tilesize/axon_dets.Sx, boxs=axon_dets.axon_box_size,
                           dest_dir=dest_dir, show=show, 
                           fname=f'{fname}_tile{tile_i:0>2}|{n_tiles:0>2}')

    if animated:
        anim_frames = [anim_frames[0]*2] + anim_frames + [anim_frames[-1]*2]
        ani = ArtistAnimation(animated[0], anim_frames, interval=400, blit=True, 
                              repeat_delay=200)
        if show:
            plt.show()
        print('encoding animation...', flush=True, end='')
        # if VIDEO_ENCODER == 'ffmpeg':
            # pickle.dump(ani, open(f'{dest_dir}/{axon_dets.dataset.name}_assoc_dets_unrendered.pkl', 'wb'))
        ani.save(f'{dest_dir}/{axon_dets.dataset.name}_assoc_dets{anim_fname_postfix}.mp4', 
                 writer=writers[VIDEO_ENCODER](fps=4), dpi=160)
        print(f'{axon_dets.dataset.name} animation saved.')
    plt.close('all')
    print(' - Done.', flush=True)


















def draw_mask(structure_screen, show=False):
    plt.subplots(figsize=(16,9))
    plt.title(structure_screen.name + ' red = start, yellow = goal, white = micro channels')
    
    draw =  np.zeros_like(structure_screen.start_mask, float)
    draw[np.array(structure_screen.mask.todense()).astype(bool)] = 1
    draw[structure_screen.start_mask.astype(bool)] = .4
    draw[structure_screen.goal_mask.astype(bool)] = .7
    plt.imshow(draw, cmap='hot')
    
    if show:
        plt.show()
    fname = f'{structure_screen.dir}/masks_{structure_screen.name}.png'
    plt.savefig(fname)
    plt.close()
    return fname

def plot_axon_IDs_lifetime(structure_screen, show=False, sort=True, subtr_t0=True, 
                           draw_verticle_lineat=None):
    # ID lifetime is a bool array with IDs as rows, t as columns
    id_lifetime = structure_screen.get_id_lifetime()
    # option to sort by ID lifetime length and to left-align the data to t0
    if sort:
        id_lifetime = id_lifetime[np.argsort(id_lifetime.sum(1))]
    if subtr_t0:
        id_lifetime_new = np.zeros_like(id_lifetime, bool)
        for i, axon_row in enumerate(id_lifetime):
            id_lives = np.where(axon_row)[0]
            id_lifetime_new[i, id_lives-id_lives[0]] = True
        id_lifetime = id_lifetime_new

    fig, ax = plt.subplots(figsize=config.SMALL_FIGSIZE)
    fig.subplots_adjust(left=.15)
    # colormap that is white for False (ID not present at t) and green for True
    ax.imshow(id_lifetime, cmap=ListedColormap([config.WHITE, config.GREEN]))
    ax.set_title(f'ID lifetime - {structure_screen.name}', fontsize=config.FONTS)
    if draw_verticle_lineat:
        ax.vlines(draw_verticle_lineat, 0, id_lifetime.shape[0]-1, linewidth=.5, color='k')

    # x, y axis setup
    ax.tick_params(labelsize=config.SMALL_FONTS)
    ax.set_xticks(np.arange(0,id_lifetime.shape[1], 40))
    ax.set_xlabel('timepoint', fontsize=config.FONTS)
    ax.set_yticks(np.arange(0,id_lifetime.shape[0], 40))
    ax.set_ylabel('Axon ID', fontsize=config.FONTS)

    if structure_screen.notes:
        ax.text(0.05, 0.95, f'Notes: {structure_screen.notes}', 
                transform=fig.transFigure)

    if show:
        plt.show()
    fname = f'{structure_screen.dir}/ID_lifetime_{structure_screen.name}.{config.FIGURE_FILETYPE}'
    plt.savefig(fname)
    plt.close()
    return fname
        

def plot_target_distance_over_time(structure_screen, show=False, subtr_init_dist=True,
                                   draw_until_t=None, count_growth_dir=True,
                                   fname_postfix=''):
    dists = structure_screen.get_distances()
    if subtr_init_dist:
        dists = structure_screen.subtract_initial_dist(dists)

    if count_growth_dir:
        growth_dir = structure_screen.growth_direction_start2end(dists)
        n_correct = (growth_dir <= -config.MIN_GROWTH_DELTA).sum()
        n_incorrect = (growth_dir >= config.MIN_GROWTH_DELTA).sum()
    
    nIDs, sizet = dists.shape

    fig, ax = plt.subplots(figsize=config.MEDIUM_FIGSIZE)
    fig.subplots_adjust(left=.14, right=.95)
    [ax.spines[which_spine].set_visible(False) for which_spine in ax.spines]
    # title = 'distance to outputchannel'
    
    # for dynamic plotting, eg inserting distance over time into timelapse video
    if draw_until_t:
        ax.set_facecolor('k')
        fig.set_facecolor('k')
        tmax = draw_until_t
        main_col = config.LIGHT_GRAY

    # static, draw all timepoints
    else:
        tmax = sizet
        main_col = config.BLACK

    # x axis setup (time)
    xmin, xmax = 0, sizet
    ax.set_xlim(xmin, xmax)
    days = dists.columns.values.astype(int)
    # get the indices where a new day starts, only plot those
    timepoints_indices = np.unique(days, return_index=True)[1]
    ax.set_xticks(timepoints_indices)
    ax.set_xticklabels([days[t] for t in timepoints_indices])
    ax.set_xlabel('DIV', color=main_col, fontsize=config.FONTS)

    # look at absolute distance to outputchannel
    if not subtr_init_dist:
        min_dist = structure_screen._compute_goalmask2target_maxdist()
        ax.add_patch(Rectangle((0,0), sizet, min_dist, facecolor=config.DARK_GRAY, 
                               edgecolor='none', alpha=.2))
        ymin, ymax = 0, 4000
        yticks = range(ymin, ymax, 1500)
        ylbl = f'distance to outputchannel {config.um}'
        ax.text(xmax-5,ymin+200, 'outputchannel reached', ha='right', va='center', fontsize=config.SMALL_FONTS, color=main_col)
    
    # or let every ID start at distance=0, then see if it increases or decreases
    else:
        ymin, ymax = -2500, 2500
        yticks = range(ymin+500,ymax-499, 1000)
        ylbl = f'{config.delta} distance to outputchannel {config.um}'
        alpha = .06 if not draw_until_t else .12
        ax.add_patch(Rectangle((0,0), xmax, ymax, color=config.RED, alpha=alpha, zorder=1))
        lbl = 'distance increases' if not count_growth_dir else f'distance increases,\nn = {n_incorrect}'
        ax.text(xmax-5,ymax-240, lbl, ha='right', va='center', fontsize=config.SMALL_FONTS, color=main_col)
        
        lbl = 'distance decreases' if not count_growth_dir else f'distance decreases,\nn = {n_correct}'
        ax.add_patch(Rectangle((0,0), xmax, ymin, color=config.GREEN, alpha=alpha, zorder=1))
        ax.text(xmax-5,ymin+240, lbl, ha='right', va='center', fontsize=config.SMALL_FONTS, color=main_col)

    # y axis setup (distance)
    ax.grid(axis='y', which='major', alpha=.4)
    ax.set_ylim(ymin, ymax)
    ax.set_ylabel(ylbl, fontsize=config.FONTS, color=main_col)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticks)
    # ax.set_yticklabels([f'{yt}' for yt in yticks])
    ax.tick_params(labelsize=config.SMALL_FONTS, color=main_col, labelcolor=main_col)

    # title and spines
    title = f'{structure_screen.name.replace("_", "-")}'
    ax.set_title(title, color=main_col, pad=15, fontsize=config.FONTS)
    ax.axline((0, ymin), (0, ymax), color=main_col, linewidth=1.3)
    ax.axline((xmin, 0), (xmax, 0), color=main_col, linewidth=1.3)
    
    # finally draw axon distances over time
    cmap = plt.cm.get_cmap('tab10', 20)
    for axon_id, dists in dists.iterrows():
        dists = dists.iloc[:tmax]
        ax.plot(dists.values, color=cmap(int(axon_id[-3:])%20), linewidth=1, zorder=3)
        
        if count_growth_dir and abs(growth_dir[axon_id]) >config.MIN_GROWTH_DELTA:
            if growth_dir[axon_id] >= config.MIN_GROWTH_DELTA:
                col = config.RED
            if growth_dir[axon_id] <= -config.MIN_GROWTH_DELTA:
                col = config.GREEN
            x_last = np.where(dists.notna())[0].max()
            y_last = dists.iloc[x_last]
            ax.scatter(x_last, y_last, s=50, facecolor='none',  edgecolor=col, zorder=4, alpha=.4)

    fname = f'{structure_screen.dir}/target_distance_{structure_screen.name}{fname_postfix}.{config.FIGURE_FILETYPE}'
    if show:
        plt.show()
    if not draw_until_t:
        plt.savefig(fname)
        plt.close()
    # for draw until t, return the figure objects so that they can be inserted in a video
    else:
        return fig,ax
    return fname

















def plot_target_distance_over_time_allscreens(all_delta_dists, dest_dir, speed, show=False,
                                            fname_postfix=''):
    
    tmin, tmax = all_delta_dists.columns[0],  all_delta_dists.columns[-1]
    all_delta_dists_med = all_delta_dists.groupby(level='design').mean()
    all_delta_dists_std = all_delta_dists.groupby(level='design').std()
    
    fig, ax = plt.subplots(figsize=config.MEDIUM_FIGSIZE)
    fig.subplots_adjust(left=.14, right=.95)
    [ax.spines[which_spine].set_visible(False) for which_spine in ax.spines]

    # x axis setup (time)
    tmin, tmax = 1.33, 6.57
    xticks = range(2,7)
    ax.set_xlim(tmin, tmax)
    ax.set_xticks(xticks)
    ax.set_xlabel('DIV', fontsize=config.FONTS)
    
    if speed:
        ymin, ymax = -1, 4000
        yticks = range(ymin+1, ymax, 1500)
        ylbl = f'axon growth {config.um_d}'
    
    else:
        ymin, ymax = -1500, 1500
        yticks = range(ymin+100,ymax-99, 400)
        ylbl = f'{config.delta} distance to outputchannel {config.um}'
        alpha = .12
        ax.add_patch(Rectangle((0,0), tmax, ymax, color=config.RED, alpha=alpha, edgecolor='none', zorder=1))
        ax.text(tmax, ymax-100, 'distance increases', ha='right', va='center', fontsize=config.SMALL_FONTS)
        ax.add_patch(Rectangle((0,0), tmax, ymin, color=config.GREEN, alpha=alpha, edgecolor='none', zorder=1))
        ax.text(tmax, ymin+100, 'distance decreases', ha='right', va='center', fontsize=config.SMALL_FONTS)

    # y axis setup (distance)
    ax.grid(axis='y', which='major', alpha=.4)
    ax.set_ylim(ymin, ymax)
    ax.set_ylabel(ylbl, fontsize=config.FONTS)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticks)
    # ax.set_yticklabels([f'{yt}' for yt in yticks])
    ax.tick_params(labelsize=config.SMALL_FONTS)

    # title and spines
    # ax.set_title(config.delta , pad=15, fontsize=config.FONTS)
    
    # finally draw axon distances over time
    cmap = plt.cm.get_cmap('tab10', 20)
    for i in range(all_delta_dists_med.shape[0]):
        design_means = all_delta_dists_med.iloc[i,:].ewm(span=24).mean()
        design_stds = all_delta_dists_std.iloc[i,:].ewm(span=24).mean()
        design = all_delta_dists_med.index[i]

        ax.plot(design_means.index, design_means.values, color=config.DESIGN_CMAP(design-1), linewidth=1, zorder=3)
        ax.fill_between(design_means.index, design_means-design_stds, design_means+design_stds, color='gray', alpha=0.3,
                interpolate=True, edgecolor='none')

    speed = '' if not speed else '_growth_speed'
    fname = f'{dest_dir}/target_distance_timeline{speed}{fname_postfix}.{config.FIGURE_FILETYPE}'
    if show:
        plt.show()
    else:
        plt.savefig(fname)
        plt.close()





















def _draw_designcolorbar(colorbar_axes):
    plt.figure()
    designs = np.arange(1,23)
    im = plt.imshow(designs[:,np.newaxis], cmap=config.DESIGN_CMAP)
    plt.close()
    cbar = plt.colorbar(im, ticks=np.linspace(1.5,21.5, 22), cax=colorbar_axes)
    ticklabels = [f'D{d:0>2}' if not i%4 else '' for i, d in enumerate(designs)]
    cbar.set_ticklabels(ticklabels)
    cbar.ax.tick_params(labelsize=config.FONTS)

def _get_barplot(n_bars, draw_colorbar=True, larger_bottom=False):
    # define the size of plot elements in inches
    lr_pad = [1, .6]
    # bars, white space between bars, space on left and right
    ax_width = 0.5*n_bars + 0.1*(n_bars-1) +.4
    width = ax_width + lr_pad[0] + lr_pad[1]
    
    if True:
    # if draw_colorbar:
        # colorbar size
        cbar_width = .4
        width += cbar_width
        lr_pad[1] += cbar_width

    fig, ax = plt.subplots(figsize=(width, config.BARPLOT_HEIGHT))
    # calculate proportional valuesby dividing through overall figure width
    lr_pad_prop = lr_pad[0]/width, 1-lr_pad[1]/width
    
    # make plot
    btm = .2 if not larger_bottom else .35
    fig.subplots_adjust(left=lr_pad_prop[0], right=lr_pad_prop[1], top=.9, bottom=btm)

    if draw_colorbar:
        bottom, left = .2, (lr_pad[0]+ax_width+cbar_width*2/3) /width
        width, height = (cbar_width*3/5) /width, .7
        cax = fig.add_axes([left, bottom, width, height])
        # draw colorbar
        _draw_designcolorbar(cax)
    return fig, ax

def plot_n_axons(n_axons, dest_dir, DIV_range, rank=False, show=False, fname_postfix=''):
    # average over designs
    if rank:
        m = n_axons.unstack(level=('timelapse', 'CLSM_area')).mean(1)
        n_axons = n_axons.reindex(m.sort_values(ascending=False).index, level=1)
    n_axons_design_avg = n_axons.unstack(level=('timelapse', 'CLSM_area')).mean(1)
    n_axons.sort_index(level=1, inplace=True)
    
    # x locations of bars
    xlocs = np.arange(1,n_axons_design_avg.shape[0]+1)
    
    designs_idx = n_axons.index.get_level_values(1)
    x, xlocs_single = 1, []
    for i in range(len(designs_idx)):
        xlocs_single.append(x)
        if i+1<len(designs_idx) and designs_idx[i] != designs_idx[i+1]:
            x += 1
    # x_design_avg = n_axons.index.unique(1)
    # x_design = n_axons.index.get_level_values('design')
    
    # get the figure
    fig, ax = _get_barplot(n_bars=n_axons_design_avg.shape[0])
    ax.tick_params(labelsize=config.SMALL_FONTS)

    # set ticks
    ax.set_xlim((xlocs[0]-.75, xlocs[-1]+.75))
    ax.set_xticks(xlocs)
    ax.set_xticklabels([f'D{x:0>2}' for x in n_axons_design_avg.index], fontsize=config.FONTS, weight='bold')
    lbl = 'Axons identified'
    if DIV_range:
        lbl += f' DIV{DIV_range[0]}-{DIV_range[1]}'
    ax.set_ylabel(lbl, fontsize=config.FONTS, weight='bold')
    
    # finally draw the number of axons
    ax.bar(xlocs, n_axons_design_avg, color=config.DESIGN_CMAP(n_axons_design_avg.index-1), zorder=1)
    ax.scatter(xlocs_single, n_axons, marker='1', s=80, color='k', zorder=2)
    
    return _save_plot(show, dest_dir, 'naxons', DIV_range, fname_postfix)
    
def plot_axon_growthspeed(growth_speeds, dest_dir, DIV_range, rank=False,
                          show=False, split_by = 'design', fname_postfix=''):
    # flatten which dataset/ timelapse dim and design replicate dim

    if split_by not in growth_speeds.index.names:
        print(f'Invalid feature to split by: {split_by}. \n Valid are {growth_speeds.index.names}')
        exit(1)
    
    average_over = [dim for dim in growth_speeds.index.names if dim != split_by]
    
    growth_speeds = growth_speeds.unstack(level=tuple(average_over))
    growth_speeds.drop('NA', errors='ignore', inplace=True)

    if rank:
        growth_speeds = growth_speeds.reindex(growth_speeds.median(1).sort_values(ascending=False).index)
    # growth_speeds_medians = growth_speeds.median(1)
    # x llocations of violin plots
    xlocs = np.arange(1,growth_speeds.shape[0]+1)

    # get the figure
    fig, ax = _get_barplot(n_bars=growth_speeds.shape[0], draw_colorbar=split_by=='design')
    ax.tick_params(labelsize=config.SMALL_FONTS)

    # set y axis up
    ymin, ymax = 0, 4000
    ax.set_ylim(ymin, ymax)
    lbl = f'Axon growth {config.um_d}'
    ax.set_ylabel(lbl, fontsize=config.FONTS, weight='bold')
    
    # setup x axis
    xmin, xmax = xlocs.min()-.75, xlocs.max()+.75
    ax.set_xlim(xmin, xmax)
    ax.set_title(split_by, fontsize=config.FONTS, weight='bold')
    xtl = [f'D{i:0>2}' for i in growth_speeds.index] if split_by == 'design' else growth_speeds.index
    
    # finally draw the number of axons
    for i, x in enumerate(xlocs):
        Y = growth_speeds.iloc[i,:].dropna()
        vp = ax.boxplot(Y, widths=.6, positions=(x,), patch_artist=True, 
                        whis=(5,95), showfliers=False)
        vp['medians'][0].set_color('k')
        vp['medians'][0].set_zorder(6)
        vp['boxes'][0].set_zorder(5)
        vp['boxes'][0].set_alpha(1)
        if split_by == 'design':
            vp['boxes'][0].set_color(config.DESIGN_CMAP(growth_speeds.index[i]))
        else:
            vp['boxes'][0].set_color(config.DEFAULT_COLORS[i])

    ax.set_xticks(xlocs)
    ax.set_xticklabels(xtl, fontsize=config.FONTS, weight='bold')
    return _save_plot(show, dest_dir, 'growth_direction', None, fname_postfix)










# sorry for this monster...
def plot_axon_destinations(destinations, dest_dir, neg_comp_metric, show=False, fname_postfix='', 
                           order=None, design_subset=None, split_by='design',
                           draw_bar_singles=False, draw_naxons_counts=True,
                           which_bars='metric',
                           shift_bar_singles=True):

    # check that a valid column name was passed
    if split_by not in destinations.index.names:
        print(f'Invalid feature to split by: {split_by}. \n Valid are {destinations.index.names}')
        exit(1)

    # get rid of samples that don't have vlid values for that feature
    notna = destinations.index.get_level_values(split_by) != 'NA'
    destinations = destinations[notna]
    # limit plot to a subset of designs
    if design_subset:
        destinations = destinations.loc[(slice(None), design_subset), :]
    destinations.sort_index(level=split_by, inplace=True)

    # take the mean over the passed dimension
    destinations_mean = destinations[which_bars].groupby(level=split_by).mean()
    mean = destinations_mean
    
    # define the order in which the bars are drawn, below ranks accornding to mean
    if order=='rank':
        order = mean.sort_values(ascending= not(which_bars!='neg_metric')).index
    # an order of designs passed
    elif order:
        order = pd.Index(order)
    # or just the alphabetical order
    else:
        order = destinations.index.unique(split_by)
    
    # use the order above the reindex
    mean = mean.reindex(order)
    # doesn't work for some reason???
    # destinations = destinations.reindex(designs_order, level=split_by, axis=0)
    indexer = tuple([slice(None) if level != split_by else order for level in destinations.index.names])
    destinations = destinations.loc[indexer]

    # x locations of bars
    xlocs = np.arange(1,mean.shape[0]+1)
    
    # get the c index of the single datapoints 
    idx = destinations.index.get_level_values(split_by)
    x, xlocs_single = 1, []
    for i in range(len(idx)):
        xlocs_single.append(x)
        # if the next index is different than the previous increase x by 1
        if i+1<len(idx) and idx[i] != idx[i+1]:
            x += 1

    # get the figure
    fig, ax = _get_barplot(n_bars=len(mean), draw_colorbar=False)
    ax.tick_params(labelsize=config.SMALL_FONTS)

    # setup title
    split_by_lbl = split_by
    if split_by_lbl == 'final lane design' or split_by_lbl == '2-joint design':
        split_by_lbl += f', opening diameter {config.um}'
    if split_by_lbl == 'channel width':
        split_by_lbl += f' {config.um}'
    ax.set_title(split_by_lbl, fontsize=config.FONTS, weight='bold')

    # setup x axis
    xmin, xmax = (xlocs[0]-.75, xlocs[-1]+.75)
    ax.set_xlim(xmin, xmax)
    ax.set_xticks(xlocs)
    if split_by == 'design':
        xtls = [f'D{i:0>2}' for i in mean.index]
    else:
        xtls = [i for i in mean.index]
    ax.set_xticklabels(xtls, fontsize=config.FONTS, weight='bold', rotation=45, ha='right', va='top')
    
    # setup y axis
    if which_bars in ['pos_metric', 'neg_metric']:
        if neg_comp_metric == 'cross_grown':
            ymin, ymax = 0, 15
            yticks = np.arange(0, ymax, 4)
        elif neg_comp_metric == 'directions':
            ymin, ymax = 0, 120
            yticks = np.arange(0,120,30)
    elif draw_bar_singles:
        if neg_comp_metric == 'cross_grown':
            ymin, ymax = 0, 12
            yticks = [0,1,4,8,]
        elif neg_comp_metric == 'directions':
            ymin, ymax = 0, 30
            yticks = [0,1,10,20,]

    elif which_bars == 'metric':
        if neg_comp_metric == 'cross_grown':
            ymin, ymax = 0, 3.5
            yticks = np.arange(ymin, ymax, 1)
        elif neg_comp_metric == 'directions':
            ymin, ymax = 0, 18
            yticks = np.arange(ymin, ymax, 3)


    # y ticks left
    ax.set_ylim(ymin, ymax)
    ax.set_yticks(yticks)
    ax.grid(axis='y', alpha=.6, linewidth=.7)

    # y axis mess..... draw patches 
    if which_bars == 'metric':
        lbl = 'directionality ratio'
        ax.set_ylabel(lbl, fontsize=config.FONTS, weight='bold')
        ax.hlines([1.66, 1.33, .33, .66], xmin, xmax, linewidth=.5, alpha=.5, zorder=3)

        ax.add_patch(Rectangle((0,1), xmax, ymax, color=config.GREEN, alpha=.1, zorder=1))
        ax.add_patch(Rectangle((0,1), xmax, -1, color=config.RED, alpha=.1, zorder=1))
    else:
        ax.tick_params(axis='y', left=False, labelleft=False, right=True, labelright=True, 
                       labelcolor='gray', labelsize=config.SMALL_FONTS)

        if which_bars =='pos_metric':
            text = 'reached\ntarget'
            col = config.GREEN
        else:
            text = 'reached\nneighbour'
            col = config.RED
        ax.text(xmax+.2, 2, text, ha='left', va='center', fontsize=config.SMALL_FONTS, color='gray')
        ax.add_patch(Rectangle((0,ymin), xmax, ymax-ymin, color=col, alpha=.1, zorder=1))

    # get colors for bars, some bars have more than one color
    main_cols = []
    cols = []
    split_by_values = mean.index
    if split_by == 'design':
        main_cols = config.DESIGN_CMAP(split_by_values)
    else:
        # more than one color
        for val in split_by_values:
            designs = destinations.reindex([val], level=split_by).index.unique('design')
            cols.append(config.DESIGN_CMAP(designs))
            main_cols.append(cols[-1][0])

    
    # draw bars
    bw = .6
    ax.bar(xlocs, mean, width=bw, zorder=4, color=main_cols)
    
    # and points forming bar
    if draw_bar_singles:
        # shift by abit to see overlapping points
        if shift_bar_singles:
            shift = (np.random.uniform(low=0, high=bw*(2/3), size=(len(xlocs_single)))) -bw*(2/6)
        else:
            shift = 0
        
        if which_bars == 'metric':
            ax.scatter(xlocs_single + shift, destinations[which_bars], marker='.', 
                       s=100, edgecolor='k', facecolor='w', zorder=200, linewidth=1.2,
                       alpha=.7, clip_on=False)
        
        else:
            # params for 3D plus and minus
            yrange = abs(ymin)+abs(ymax)
            x_diam = .18
            thickness = .09
            lw = .9

            if which_bars == 'pos_metric':
                y_diam = yrange*.03
                patches = [_plus3D((x,y), x_diam, y_diam, lw, thickness) for x,y in zip(xlocs_single + shift, destinations[which_bars])]

            elif which_bars == 'neg_metric':
                y_diam = yrange*.007
                patches = [_minus3D((x,y), x_diam, y_diam, lw, thickness) for x,y in zip(xlocs_single + shift, destinations[which_bars])]
            [ax.add_patch(p) for p in patches]

    # for feature split, bars have more than one color, draw ploygons here
    if not split_by == 'design':
        angle = .1
        # iter bars
        for i in range(len(mean)):
            # get basic bar coordinates
            y, x = mean.iloc[i], xlocs[i]
            n_sections =  len(cols[i])
            if n_sections == 1:
                continue
            # for bars of more than one color, get the evenly spaced bar segment centers
            y_centers = np.linspace(0,y, n_sections+1)[1:-1]

            # for each color draw a polygon that becomes higher but has a lower zorder
            for j in range(n_sections-1):
                bl = [x-bw/2, 0]
                br = [x+bw/2, 0]
                tl = [x-bw/2, y_centers[j]-angle]
                tr = [x+bw/2, y_centers[j]+angle]
                ax.add_patch(Polygon([bl,tl,tr,br], color=cols[i][j+1], zorder=100-j))
    
    # y axis on the right
    if draw_naxons_counts:
        ax2 = ax.twinx()
        ax2.tick_params(axis='y', labelcolor='gray', labelsize=config.SMALL_FONTS)

        if neg_comp_metric == 'cross_grown':
            ymin2 = -15
            ymax2 = ymin2* -2.5
            yticks = np.arange(ymin2,ymax2, 15, dtype=int) 
            y_text = (-7.5, 7.5)
        elif neg_comp_metric == 'directions':
            ymin2 = -40
            ymax2 = 120
            yticks = np.arange(ymin2,ymax2, 40, dtype=int) 
            y_text = (-20, 20)
            
            ax2.hlines(0, xmin, xmax, edgecolor='k')
        
        ax2.set_ylim(ymin2, ymax2)
        ax2.set_yticks(yticks)
        ax2.set_yticklabels(np.abs(yticks))
    
        ax2.text(xmax+.2, y_text[0], 'reached\nneighbour', ha='left', va='center', fontsize=config.SMALL_FONTS, color='gray')
        ax2.text(xmax+.2, y_text[1], 'reached\ntarget', ha='left', va='center', fontsize=config.SMALL_FONTS, color='gray')

        ax2.scatter(xlocs_single, destinations.pos_metric, marker='+', s=90, color='k',  alpha=.75, zorder=5, linewidth=1.4)
        ax2.scatter(xlocs_single, -destinations.neg_metric, marker='_', s=90, color='k', alpha=.75, zorder=5, linewidth=1.4)
    return _save_plot(show, dest_dir, 'destinations', DIV_range=None, fname_postfix=fname_postfix)
    










# sorry for this monster...
def plot_axon_distribution(axon_data, dest_dir, show=False, fname_postfix='', 
                           order=None, design_subset=None, split_by='design',
                           draw_bar_singles=False, 
                           which_metric='metric',
                           shift_bar_singles=True):

    print(axon_data)
    # check that a valid column name was passed
    if split_by not in axon_data.index.names:
        print(f'Invalid feature to split by: {split_by}. \n Valid are {axon_data.index.names}')
        exit(1)

    def select_feature(data, split_by, design_subset):
        # get rid of samples that don't have vlid values for that feature
        notna = data.index.get_level_values(split_by) != 'NA'
        data = data[notna]
        # limit plot to a subset of designs
        if design_subset:
            data = data.loc[(slice(None), design_subset), :]
        return data.sort_index(level=split_by)
    
    axon_data = select_feature(axon_data, split_by, design_subset)

    # take the mean over the passed dimension
    
    medians = axon_data.groupby(level=split_by).apply(lambda dat: dat.stack().median())
    print(medians)

    # define the order in which the bars are drawn, below ranks accornding to mean
    if order == 'rank':
        order = medians.sort_values(ascending= not(which_metric.startswith('neg_metric'))).index
    # an order of feature levels passed
    elif order:
        order = pd.Index(order)
    # or just the alphabetical order of feature level
    else:
        order = axon_data.index.unique(split_by)
    
    # use the order above the reindex
    medians = medians.reindex(order)
    # doesn't work for some reason???
    # destinations = destinations.reindex(designs_order, level=split_by, axis=0)
    indexer = tuple([slice(None) if level != split_by else order for level in axon_data.index.names])
    axon_data = axon_data.loc[indexer]

    # x locations of bars
    xlocs = np.arange(1,medians.shape[0]+1)
    
    # get the c index of the single datapoints 
    idx = axon_data.index.get_level_values(split_by)
    x, xlocs_single = 1, []
    for i in range(len(idx)):
        xlocs_single.append(x)
        # if the next index is different than the previous increase x by 1
        if i+1<len(idx) and idx[i] != idx[i+1]:
            x += 1

    # get the figure
    fig, ax = _get_barplot(n_bars=len(medians), draw_colorbar=False)
    ax.tick_params(labelsize=config.SMALL_FONTS)

    # setup title
    split_by_lbl = split_by
    if split_by_lbl == 'final lane design' or split_by_lbl == '2-joint design':
        split_by_lbl += f', opening diameter {config.um}'
    if split_by_lbl == 'channel width':
        split_by_lbl += f' {config.um}'
    ax.set_title(split_by_lbl, fontsize=config.FONTS, weight='bold')

    # setup x axis
    xmin, xmax = (xlocs[0]-.75, xlocs[-1]+.75)
    ax.set_xlim(xmin, xmax)
    ax.set_xticks(xlocs)
    if split_by == 'design':
        xtls = [f'D{i:0>2}' for i in medians.index]
    else:
        xtls = [i for i in medians.index]
    ax.set_xticklabels(xtls, fontsize=config.FONTS, weight='bold', rotation=45, ha='right', va='top')
    
    # setup y axis
    if which_metric in ['speed']:
        ymin, ymax = 0, 4000
        yticks = [0,1500,3000,]
        ax.set_ylim(ymin, ymax)
        
    elif draw_bar_singles:
        ymin, ymax = 0, 12
        yticks = [0,1,4,8,]
    elif which_metric == 'metric':
        ymin, ymax = 0, 3.5
        yticks = np.arange(ymin, ymax, 1)

    # y ticks left
    ax.set_ylim(ymin, ymax)
    ax.set_yticks(yticks)
    ax.grid(axis='y', alpha=.6, linewidth=.7)

    # y axis mess..... draw patches 
    if which_metric == 'speed':
        lbl = f'Axon growth {config.um_d}'
        ax.set_ylabel(lbl, fontsize=config.FONTS, weight='bold')

        # ax.add_patch(Rectangle((0,1), xmax, ymax, color=config.GREEN, alpha=.1, zorder=1))
    else:
        pass
        # ax.tick_params(axis='y', left=False, labelleft=False, right=True, labelright=True, 
        #                labelcolor='gray', labelsize=config.SMALL_FONTS)

        # if which_metric =='pos_metric':
        #     text = 'reached\ntarget'
        #     col = config.GREEN
        # else:
        #     text = 'reached\nneighbour'
        #     col = config.RED
        # ax.text(xmax+.2, 2, text, ha='left', va='center', fontsize=config.SMALL_FONTS, color='gray')
        # ax.add_patch(Rectangle((0,ymin), xmax, ymax-ymin, color=col, alpha=.1, zorder=1))

    # get colors for bars, some bars have more than one color
    main_cols = []
    cols = []
    split_by_values = medians.index
    if split_by == 'design':
        main_cols = config.DESIGN_CMAP(split_by_values)
    else:
        # more than one color
        for val in split_by_values:
            designs = axon_data.reindex([val], level=split_by).index.unique('design')
            cols.append(config.DESIGN_CMAP(designs))
            main_cols.append(cols[-1][0])

    # draw bars
    bw = .6


    # print(xtls)
    # print(medians)
    # print(axon_data)
    # finally draw the number of axons
    for i, x in enumerate(xlocs):
        mask = axon_data.index.get_level_values(split_by) == medians.index[i]
        Y = axon_data[mask].stack().dropna().values
        # print(xtl[i], Y.shape, '\n')
        vp = ax.boxplot(Y, widths=.6, positions=(x,), patch_artist=True, 
                        whis=(5,95), showfliers=False)
        vp['medians'][0].set_color('k')
        vp['medians'][0].set_zorder(6)
        vp['boxes'][0].set_zorder(5)
        vp['boxes'][0].set_alpha(1)
        if split_by == 'design':
            vp['boxes'][0].set_color(config.DESIGN_CMAP(medians.index[i]))
        else:
            vp['boxes'][0].set_color(main_cols[i])

        print(vp['boxes'][0].get_path())

    ax.set_xticks(xlocs)
    ax.set_xticklabels(xtls, fontsize=config.FONTS, weight='bold')


    
    # # and points forming bar
    # if draw_bar_singles:
    #     # shift by abit to see overlapping points
    #     if shift_bar_singles:
    #         shift = (np.random.uniform(low=0, high=bw*(2/3), size=(len(xlocs_single)))) -bw*(2/6)
    #     else:
    #         shift = 0
        
    #     if which_metric == 'metric':
    #         ax.scatter(xlocs_single + shift, axon_data[which_metric], marker='.', 
    #                    s=100, edgecolor='k', facecolor='w', zorder=200, linewidth=1.2,
    #                    alpha=.7, clip_on=False)
        
    #     else:
    #         # params for 3D plus and minus
    #         yrange = abs(ymin)+abs(ymax)
    #         x_diam = .18
    #         thickness = .09
    #         lw = .9

    #         if which_metric == 'pos_metric':
    #             y_diam = yrange*.03
    #             patches = [_plus3D((x,y), x_diam, y_diam, lw, thickness) for x,y in zip(xlocs_single + shift, axon_data[which_metric])]

    #         elif which_metric == 'neg_metric':
    #             y_diam = yrange*.007
    #             patches = [_minus3D((x,y), x_diam, y_diam, lw, thickness) for x,y in zip(xlocs_single + shift, axon_data[which_metric])]
    #         [ax.add_patch(p) for p in patches]

    # for feature split, bars have more than one color, draw ploygons here
    if not split_by == 'design':
        angle = .1
        # iter bars
        for i in range(len(medians)):
            # get basic bar coordinates
            y, x = medians.iloc[i], xlocs[i]
            n_sections =  len(cols[i])
            if n_sections == 1:
                continue
            # for bars of more than one color, get the evenly spaced bar segment centers
            y_centers = np.linspace(0,y, n_sections+1)[1:-1]

            # for each color draw a polygon that becomes higher but has a lower zorder
            for j in range(n_sections-1):
                bl = [x-bw/2, 0]
                br = [x+bw/2, 0]
                tl = [x-bw/2, y_centers[j]-angle]
                tr = [x+bw/2, y_centers[j]+angle]
                ax.add_patch(Polygon([bl,tl,tr,br], color=cols[i][j+1], zorder=100-j))
    
  
    return _save_plot(show, dest_dir, 'destinations', DIV_range=None, fname_postfix=fname_postfix)
    






# def plot_growth_directions(directions, dest_dir, show=False, rank=False, 
#                            draw_min_dist_lines=True, fname_postfix='',
#                            split_by='design'):
#     # flatten which dataset/ timelapse dim and design replicate dim

#     average_over = [dim for dim in directions.index.names if dim != split_by]
#     directions = directions.unstack(level=tuple(average_over))
#     directions.sort_index(level=1, inplace=True)
#     directions.drop('NA', errors='ignore', inplace=True)
#     if directions.empty:
#         return None

#     # directions = directions.unstack(level=('timelapse', 'CLSM_area'))
#     if rank:
#         directions = directions.reindex(directions.median(1).sort_values().index)
#     directions_medians = directions.median(1)
#     # xlocs
#     xlocs = np.arange(1,directions.shape[0]+1)

#     # get the figure
#     fig, ax = _get_barplot(n_bars=directions.shape[0])
#     ax.tick_params(labelsize=config.SMALL_FONTS)

#     # set y axis up
#     ymin, ymax = -2500, 2500
#     ax.set_ylim(ymin, ymax)
#     lbl = f'Axons {config.delta} distance to target {config.um}'
#     ax.set_ylabel(lbl, fontsize=config.FONTS, weight='bold')
#     ax.set_title(split_by, fontsize=config.FONTS, weight='bold')
    
#     # setup x axis
#     xmin, xmax = xlocs.min()-.75, xlocs.max()+.75
#     ax.set_xlim(xmin, xmax)
#     ax.set_xticks(xlocs)

#     xtl = [f'D{i:0>2}' for i in directions.index] if split_by == 'design' else directions.index
#     ax.set_xticklabels(xtl, fontsize=config.FONTS, weight='bold', ha='right', va='top', rotation=25)

#     ax.hlines(0, xlocs[0]-.75, xlocs[-1]+.75, color='k', linewidth=1, zorder=2)
#     if draw_min_dist_lines:
#         ax.hlines(100, xlocs[0]-.75, xlocs[-1]+.75, color='k', linewidth=.5, linestyle='dashed', zorder=1)
#         ax.hlines(-100, xlocs[0]-.75, xlocs[-1]+.75, color='k', linewidth=.5, linestyle='dashed', zorder=1)
    
#     ax.add_patch(Rectangle((0,0), xmax, ymax, color=config.RED, alpha=.1, zorder=1))
#     ax.text(xmax-.1,ymax-300, 'incorrect direction', ha='right', va='center', fontsize=config.SMALL_FONTS, zorder=8)
#     ax.add_patch(Rectangle((0,0), xmax, ymin, color=config.GREEN, alpha=.1, zorder=1))
#     ax.text(xmax-.1,ymin+300, 'correct direction', ha='right', va='center', fontsize=config.SMALL_FONTS, zorder=8)

#     # finally draw the number of axons
#     for i, x in enumerate(xlocs):
#         Y = directions.iloc[i,:].dropna()
#         vp = ax.violinplot(Y, widths=.8, positions=(x,), showextrema=False, 
#                            showmedians=True)  
#         vp['cmedians'].set_color('k')
#         vp['cmedians'].set_zorder(6)
#         vp['bodies'][0].set_zorder(5)
#         vp['bodies'][0].set_alpha(1)
#         if split_by == 'design':
#             vp['bodies'][0].set_color(config.DESIGN_CMAP(directions.index[i]))
#     return _save_plot(show, dest_dir, 'growth_direction', None, fname_postfix)












# def plot_counted_growth_directions(directions, dest_dir, show=False, rank=False, 
#                                    fname_postfix=''):
#     # flatten which dataset/ timelapse dim and design replicate dim
#     # directions[directions.abs()<300] = np.nan
#     n_axons = directions.notna().sum(1)
#     n_axons_grown = {'n_wrong_direc': -(directions>100).sum(1)/n_axons.values, 
#                      'n_right_direc': (directions<-100).sum(1)/n_axons.values} 
#     directions = pd.DataFrame(n_axons_grown, index=directions.index)
#     directions.sort_index(level=1, inplace=True)
    
#     directions['metric'] = directions.n_right_direc + directions.n_wrong_direc
#     directions_sum = directions.metric.groupby(level='design').mean()

#     if rank:
#         designs_order = directions_sum.sort_values(ascending=False).index
#         # directions = directions.reindex(designs_order, level=1)
#         directions = directions.loc[(slice(None), designs_order.values), :]
#         directions_sum = directions_sum.reindex(designs_order.values)
    
#     xlocs = np.arange(1,directions_sum.shape[0]+1)
    
#     tls = directions.index.get_level_values(0)
#     designs_idx = directions.index.get_level_values(1)
#     x, xlocs_single = 1, []
#     for i in range(len(designs_idx)):
#         xlocs_single.append(x-.05 if tls[i] == 'tl13' else x+.05)
#         if i+1<len(designs_idx) and designs_idx[i] != designs_idx[i+1]:
#             x += 1
        
#     # get the figure
#     fig, ax = _get_barplot(n_bars=len(directions_sum))
#     ax.tick_params(labelsize=config.SMALL_FONTS)

#     # setup y axis
#     # ymin,ymax = -51, 171
#     # yticks = np.arange(ymin+1,ymax,25)
#     # ax.set_ylim(ymin, ymax)
#     lbl = 'n axons growth direction'
#     ax.set_ylabel(lbl, fontsize=config.FONTS, weight='bold')
#     # ax.set_yticks(yticks)
#     # ax.set_yticklabels(abs(yticks))
    
#     # setup x axis
#     xmin, xmax = (xlocs[0]-.75, xlocs[-1]+.75)
#     ax.set_xlim(xmin, xmax)
#     ax.set_xticks(xlocs)
#     ax.set_xticklabels([f'D{i:0>2}' for i in directions_sum.index], fontsize=config.FONTS, weight='bold')
#     ax.hlines(0, xlocs[0]-.75, xlocs[-1]+.75, color='k', linewidth=1, zorder=2)
    
#     # ax.add_patch(Rectangle((0,0), xmax, ymax, color=config.GREEN, alpha=.1, zorder=1))
#     # ax.text(xmax-.1,ymax-10, 'correct direction', ha='right', va='center', fontsize=config.SMALL_FONTS)
#     # ax.add_patch(Rectangle((0,0), xmax, ymin, color=config.RED, alpha=.1, zorder=1))
#     # ax.text(xmax-.1,ymin+10, 'incorrect direction', ha='right', va='center', fontsize=config.SMALL_FONTS)

    
#     # finally draw the number of axons
#     ax.bar(xlocs, directions_sum, color=config.DESIGN_CMAP(directions_sum.index-1), zorder=3)
#     ax.scatter(xlocs_single, directions.n_right_direc, marker='+', s=50, color='k', zorder=4, linewidth=1)
#     ax.scatter(xlocs_single, directions.n_wrong_direc, marker='_', s=50, color='k', zorder=4, linewidth=1)
    
#     return _save_plot(show, dest_dir, 'growth_direction_counted', DIV_range=None, fname_postfix=fname_postfix)
    
def _save_plot(show, dest_dir, base_fname, DIV_range, fname_postfix):
    if show:
        plt.show()
    else:
        fname = f'{dest_dir}/{base_fname}'
        if DIV_range:
            fname += f'_DIV{DIV_range[0]}-{DIV_range[1]}'
        fname = f'{fname}{fname_postfix}.{config.FIGURE_FILETYPE}'
        print('Saving: ', fname)
        plt.savefig(fname, dpi=300)
        plt.close()
        return fname

def _plus3D(xy_center, x_diameter, y_diameter, linewidth, thickness, facecolor='w', edgecolor='k', zorder=200):
    top, halftop, halfbottom, bottom = (-y_diameter/2, -y_diameter*thickness, y_diameter*thickness, y_diameter/2)
    left, halfleft, halfright, right = (-x_diameter/2, -x_diameter*thickness, x_diameter*thickness, x_diameter/2)
    x = (left, halfleft, halfleft, halfright, halfright, right, right, halfright, halfright, halfleft, halfleft, left)
    y = (halftop, halftop, top, top, halftop, halftop, halfbottom, halfbottom, bottom, bottom, halfbottom, halfbottom)

    coordinates = np.array((x,y)).T + xy_center
    plus = Polygon(coordinates, linewidth=linewidth, facecolor=facecolor, 
                edgecolor=edgecolor, zorder=zorder, clip_on=False, alpha=.8)
    return plus

def _minus3D(xy_center, x_diameter, y_diameter, linewidth, thickness, facecolor='w', edgecolor='k', zorder=200):
    top, halftop, halfbottom, bottom = (-y_diameter/2, -y_diameter*thickness, y_diameter*thickness, y_diameter/2)
    left, halfleft, halfright, right = (-x_diameter/2, -x_diameter*thickness, x_diameter*thickness, x_diameter/2)
    x = (left, right, right, left)
    y = (top, top, bottom, bottom)

    coordinates = np.array((x,y)).T + xy_center
    minus = Polygon(coordinates, linewidth=linewidth, facecolor=facecolor, 
                edgecolor=edgecolor, zorder=zorder, clip_on=False, alpha=.8)
    return minus