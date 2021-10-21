import pickle
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
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


def plot_preprocessed_input_data(preporc_dat_csv, notes='', dest_dir=None, show=False, motion_plots=False):
    data = pd.read_csv(preporc_dat_csv, header=[0,1,2], index_col=0)

    fig = plt.figure(figsize=(18,9))
    fig.subplots_adjust(hspace=.4, wspace=.05, top=.95, bottom=.05, left= .03, right=.97)
    widths = [1, 1, 0.3, 1, 1]
    heights = [1, 1, 1] if motion_plots else [1, 1]
    spec = fig.add_gridspec(ncols=5, nrows=len(heights), width_ratios=widths,
                            height_ratios=heights)
    axes = [fig.add_subplot(spec[row,col]) for row in range(len(heights)) for col in (0,1,3,4)]
    fig.suptitle(notes)

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
            ax.tick_params(axis='both', labelsize=8)
            ax.set_title(f'{which_timpoint}-data: {which_preproc}', fontsize=9)
            if j:
                ax.tick_params(labelleft=False, labelright=True)
            
            for dat_name in data.columns.unique(0):
                # when standardized label differs....
                if which_preproc not in data[dat_name].columns.unique(0):
                    which_preproc = data[dat_name].columns.unique(0)[i]
                dat = data[(dat_name, which_preproc, which_timpoint)]
                
                if which_preproc.startswith('Standardized'):
                    args = {'bins': 2**14, 'range':(0, 60)}
                    ax.set_ylim((0, 2))
                    ax.set_xlim((-.5, 1.5))
                    # ax.set_xlim((-.5, 1.5))
                elif 'Motion' in which_preproc:
                    args = {'bins': 2**10, }
                    ax.set_ylim((0, 0.1))
                    ax.set_xlim((-.5, 14))
                else:
                    args = {'bins': 2**10, 'range':(0, 1/2**4)} # image data is saved in 12bit format (0-4095)
                    ax.set_ylim((0, 50))
                    ax.set_xlim((-.01, .09))
                
                dat_sprsty = f', sparsity: {(dat==0).sum()/len(dat):.3f}'
                ax.hist(dat, alpha=.5, density=True, zorder=3,
                                label=dat_name+dat_sprsty, **args)
            ax.legend(fontsize=8)
    if show:
        plt.show()
    if dest_dir is not None:
        fname = f'{dest_dir}/preprocessed_data.{config.FIGURE_FILETYPE}'
        fig.savefig(fname)
    plt.close()


def plot_training_process(training_data_files, draw_detailed_loss=False, 
                          draw_best_thresholds=False, show=False, dest_dir=None):
    # create the figure, share epoch (x axis)
    fig, axes = plt.subplots(2,4, figsize=(19,9), sharex=True)
    fig.subplots_adjust(wspace=.03, left=.03, right=.97, bottom=.05, top=.95)
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
            if i not in [0,4]:
                ax.tick_params(labelleft=False)

            # loss plots: xy anchor, box loss, no-box loss, summed loss
            if i<4:
                x_epochs = loss_x_epochs
                ax.set_ylim(0,15)

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
                ax.set_ylim(0.45,1.05)
                ax.set_yticks(np.arange(0.4,1.01, .1))
                ax.set_xlabel('Epoch')
                
                # get one specific metrics
                name = metric.index[i-4]
                trn_mtrc = metric.loc[name, (slice(None), 'train')].droplevel((1,2))
                tst_mtrc = metric.loc[name, (slice(None), 'test')].droplevel((1,2))
                # smooth 
                trn_data = trn_mtrc.ewm(span=10).mean()
                tst_data = tst_mtrc.ewm(span=10).mean()

            # last plot (8/8) has no data, legend is place here
            elif i == 7:
                ax.axis('off')
                # save the run's identifying color to make legend in the end
                legend_elements.extend([
                    Line2D([0], [0], label=f'Train - {exp_run_id}', color=trn_col,
                            **dict(TRAIN_Ps, **{'linewidth':4})),
                    Line2D([0], [0], label=f'Test', color=tst_col,
                            **dict(TEST_Ps, **{'linewidth':4}),),])
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
            ax.set_title(name)

            # finally draw the train and test data defined above
            ls = ax.plot(x_epochs, trn_data, **TRAIN_Ps)
            trn_col = ls[-1].get_color()
            ls = ax.plot(x_epochs, tst_data, **TEST_Ps)
            tst_col = ls[-1].get_color()

    # draw legend at last exes
    ax.legend(handles=legend_elements, bbox_to_anchor=(.05, .95), ncol=1,
                    loc='upper left', fontsize=11)
    
    if show:
        plt.show()
    if dest_dir is not None:
        fname = f'{dest_dir}/training{file_name_apdx}.png'
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
    im = np.where(im>0, im-min_val, 0)  # shift distribution to 0
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
    if draw_grid and int(fname[fname.rfind(':')+1:fname.rfind('of')]) == 0:
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)
        gridsize = round(draw_grid)
        at_x = [gridsize*i for i in range(width//gridsize+1)]
        at_y = [gridsize*i for i in range(width//gridsize+1)]
        ax.vlines(at_x, 0, height, color='white', linewidth=.5, alpha=.1)
        ax.hlines(at_y, 0, width, color='white', linewidth=.5, alpha=.1)

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

    im_artist = ax.imshow(im)
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
        fig.savefig(f'{dest_dir}/{fname}.png')
        plt.close(fig) 


def plot_prc_rcl(metrics_files, dest_dir=None, show=None):
    fig, axes = plt.subplots(1, 2)
    fig.subplots_adjust(bottom=.001)
    ticks = np.arange(0.2, 1.1, 0.1)
    lims = (.25, 1.04)

    # setup both axes
    for i, ax in enumerate(axes):
        ax.set_ylim(*lims) 
        ax.set_yticks(ticks)
        ax.set_yticklabels([f'{t:.1f}' for t in ticks], fontsize=8)

        ax.set_xlim(*lims)
        ax.set_xticks(ticks)
        ax.set_xticklabels([f'{t:.1f}' for t in ticks], fontsize=8)

        ax.set_aspect('equal')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid()
        if i == 0:
            ax.set_xlabel('Precision', fontsize=10)
            ax.set_ylabel('Recall', fontsize=10)
        else:
            ax.set_xlabel('Thresholds', fontsize=10)
            ax.set_ylabel('F1', fontsize=10)

    # iterate different files
    legend_elements, runs = [], ''
    for name, files in metrics_files.items():
        runs += name[:5]

        # not only the metrics file for the epoch is passed, but also the 15  
        # epochs before and after. the average over these 30 is computed below
        metrics = [pd.read_pickle(file) for file in files]
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
        axes[0].plot(tst_prc, tst_rcl, color=col, **TEST_Ps)
        axes[0].scatter(tst_prc[best_thr], tst_rcl[best_thr], color=col, s=15, zorder=20)
        
        # F1 plot
        axes[1].plot(thrs, trn_F1, color=col, **TRAIN_Ps)
        axes[1].plot(thrs, tst_F1, color=col, **TEST_Ps)
        axes[1].scatter(thrs[best_thr], tst_F1[best_thr], color=col, s=15, zorder=20)

        # legend 
        legend_elements.extend([
            Line2D([0], [0], label=f'Train - {name}', color=col, **dict(TRAIN_Ps, **{'linewidth':4})),
            Line2D([0], [0], label=f'Test', color=col, **dict(TEST_Ps, **{'linewidth':3}))
            ])
    axes[0].legend(handles=legend_elements, bbox_to_anchor=(-.2, 1.05), ncol=2,
                   loc='lower left', fontsize=8,)
    if show:
        plt.show()
    if dest_dir:
        fig.savefig(f'{dest_dir}/prc_rcl_{runs}.png', dpi=450)

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
                draw_frame(tile, tiled_det[tile_i], target_det, 
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
        plt.close()
        print(f'{axon_dets.dataset.name} animation saved.')
    print(' - Done.', flush=True)
























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
                                   draw_until_t=None):
    dists = structure_screen.get_distances()
    if subtr_init_dist:
        dists = structure_screen.subtract_initial_dist(dists)
    nIDs, sizet = dists.shape

    fig, ax = plt.subplots(figsize=config.MEDIUM_FIGSIZE)
    fig.subplots_adjust(left=.15)
    [ax.spines[which_spine].set_visible(False) for which_spine in ax.spines]
    title = 'distance to outputchannel'
    
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

    # look at absolute distance to outputchannel
    if not subtr_init_dist:
        min_dist = structure_screen.goalmask2target_maxdist
        ax.add_patch(Rectangle((0,0), sizet, min_dist, facecolor=config.DARK_GRAY, 
                               edgecolor='none', alpha=.2))
        ymin, ymax = 0, 4000
        yticks = range(ymin, ymax, 1500)
        ax.text(.14, 0.13, 'outputchanel\nreached', fontsize=config.FONTS, 
                color=main_col, transform=fig.transFigure, ha='right') 
    
    # or let every ID start at distance=0, then see if it increases or decreases
    else:
        title = 'relative ' + title
        ymin, ymax = -2500, 2500
        yticks = range(ymin+500,ymax-499, 1000)
        ax.text(.1, 0.5, r'decreases \hspace{4cm} increases', rotation=90,
                fontsize=config.FONTS, color=main_col, transform=fig.transFigure,
                va='center', ha='center')

    # y axis setup (distance)
    ax.grid(axis='y', which='major', alpha=.4)
    ax.set_ylim(ymin, ymax)
    ax.set_yticks(yticks)
    ax.set_yticklabels([str(yt)+r' $\upmu$m' for yt in yticks])
    ax.tick_params(labelsize=config.SMALL_FONTS, color=main_col, labelleft=False,
                   left=False, labelright=True, labelcolor=main_col)

    # x axis setup (time)
    xmin, xmax = 0, sizet
    ax.set_xlim(xmin, xmax)
    days = np.array([t//(24*60) for t in dists.columns])
    # get the indices where a new day starts, only plot those
    timepoints_indices = np.unique(days, return_index=True)[1]
    ax.set_xticks(timepoints_indices)
    ax.set_xticklabels([days[t] for t in timepoints_indices])
    ax.set_xlabel('DIV', color=main_col, fontsize=config.FONTS)

    # title and spines
    title = f'{structure_screen.name.replace("_", "-")}: {title}'
    ax.set_title(title, color=main_col, pad=15, fontsize=config.FONTS)
    ax.axline((0, ymin), (0, ymax), color=main_col, linewidth=1.3)
    ax.axline((xmin, 0), (xmax, 0), color=main_col, linewidth=1.3)
    
    # finally draw axon distances over time
    cmap = plt.cm.get_cmap('tab10', 20)
    for axon_id, dists in dists.iterrows():
        dists = dists.iloc[:tmax]
        ax.plot(dists.values, color=cmap(int(axon_id[-3:])%20), linewidth=1)

    if show:
        plt.show()
    if not draw_until_t:
        fname = f'{structure_screen.dir}/target_distance_{structure_screen.name}.{config.FIGURE_FILETYPE}'
        plt.savefig(fname)
        plt.close()
        return fname
    # for draw until t, return the figure objects so that they can be inserted 
    # in a video
    else:
        return fig,ax












def _draw_designcolorbar(colorbar_axes):
    plt.figure()
    designs = np.arange(1,23)
    im = plt.imshow(designs[:,np.newaxis], cmap=config.DESIGN_CMAP)
    plt.close()
    cbar = plt.colorbar(im, ticks=np.linspace(1.5,21.5, 22), cax=colorbar_axes)
    ticklabels = [f'D{d:0>2}' if not i%3 else '' for i, d in enumerate(designs)]
    cbar.set_ticklabels(ticklabels)
    cbar.ax.tick_params(labelsize=config.SMALL_FONTS)

def _get_barplot(n_bars, draw_colorbar=True):
    # define the size of plot elements in inches
    lr_pad = [.9, .4]
    # bars, white space between bars, space on left and right
    ax_width = 0.5*n_bars + 0.1*(n_bars-1) +.4
    width = ax_width + lr_pad[0] + lr_pad[1]
    if draw_colorbar:
        # colorbar size
        cbar_width = .4
        width += cbar_width
        lr_pad[1] += cbar_width

    fig, ax = plt.subplots(figsize=(width, config.BARPLOT_HEIGHT))
    # calculate proportional valuesby dividing through overall figure width
    lr_pad_prop = lr_pad[0]/width, 1-lr_pad[1]/width
    # make plot
    fig.subplots_adjust(left=lr_pad_prop[0], right=lr_pad_prop[1], top=.9, bottom=.1)

    if draw_colorbar:
        bottom, left = .1, (lr_pad[0]+ax_width+cbar_width*2/3) /width
        width, height = (cbar_width*1/3) /width, .8
        cax = fig.add_axes([left, bottom, width, height])
        # draw colorbar
        _draw_designcolorbar(cax)
    return fig, ax

def plot_n_axons(n_axons, dest_dir, DIV_range, show=False, fname_postfix=''):
    # average over designs
    n_axons_design_avg = n_axons.groupby(level='design').mean()
    # xlocs
    x_design_avg = n_axons.index.unique(1)
    x_design = n_axons.index.get_level_values('design')
    
    # get the figure
    fig, ax = _get_barplot(n_bars=len(n_axons_design_avg))

    # set ticks
    ax.set_xlim((x_design_avg[0]-.75, x_design_avg[-1]+.75))
    ax.set_xticks(x_design_avg)
    ax.set_xticklabels([f'D{i:0>2}' for i in x_design_avg], fontsize=config.FONTS)
    lbl = 'Axons identified'
    if DIV_range:
        lbl += f' DIV{DIV_range[0]}-{DIV_range[1]}'
    ax.set_ylabel(lbl, fontsize=config.FONTS)
    ax.tick_params(labelsize=config.SMALL_FONTS)
    
    # finally draw the number of axons
    ax.bar(x_design_avg, n_axons_design_avg, color=config.DESIGN_CMAP(x_design_avg-1), zorder=1)
    ax.scatter(x_design, n_axons, marker='1', s=80, color='k', zorder=2)
    
    return _save_plot(show, dest_dir, 'naxons', DIV_range, fname_postfix)
    
def plot_axon_growthspeed(growth_speeds, dest_dir, DIV_range,
                          show=False, fname_postfix=''):
    # flatten which dataset/ timelapse dim and design replicate dim
    growth_speeds = growth_speeds.unstack(level=('timelapse', 'CLSM_area'))
    
    # # rank    
    # order = growth_speeds_avg.sort_values(ascending=True).index
    # growth_speeds_avg = growth_speeds_avg.reindex(order)
    # growth_speeds = growth_speeds.reindex(order)

    # xlocs
    x_design_avg = growth_speeds.index.unique()
    
    # get the figure
    fig, ax = _get_barplot(n_bars=growth_speeds.shape[0])

    # set ticks
    ax.set_xlim((x_design_avg.min()-.75, x_design_avg.max()+.75))
    ax.set_xticks(x_design_avg)
    ax.set_xticklabels([f'D{i:0>2}' for i in x_design_avg], fontsize=config.FONTS)
    lbl = 'Axon outgrowth speed ' + config.um_d
    if DIV_range:
        lbl += f' DIV{DIV_range[0]}-{DIV_range[1]}'
    ax.set_ylabel(lbl, fontsize=config.FONTS)
    ax.tick_params(labelsize=config.SMALL_FONTS)

    # finally draw the number of axons
    for x in x_design_avg:
        Y = growth_speeds.loc[x,:].dropna().sort_values()
        ax.boxplot(Y, widths=.8, patch_artist=True, positions=(x,),
                   flierprops={'markerfacecolor':'k', 'marker':'.', 'markersize':3}, 
                   medianprops={'color':'k'},
                   boxprops={'facecolor': config.DESIGN_CMAP(x-1)},
        )
    
    print(fname_postfix)
    return _save_plot(show, dest_dir, 'growth_speed', DIV_range, fname_postfix)


def plot_axon_destinations(destinations, dest_dir, show=False, fname_postfix=''):
    # average over designs
    print(destinations)
    destinations['metric'] = destinations.target - destinations.cross_growth
    print(destinations)


    destinations_design_avg = destinations.groupby(level='design').mean()
    print(destinations_design_avg)
    # xlocs
    x_design_avg = destinations.index.unique('design')
    x_design_left = destinations.index.get_level_values('design')-.15
    x_design_right = destinations.index.get_level_values('design')+.15
    
    # get the figure
    fig, ax = _get_barplot(n_bars=len(destinations_design_avg))

    # set ticks
    ax.set_xlim((x_design_avg[0]-.75, x_design_avg[-1]+.75))
    ax.set_xticks(x_design_avg)
    ax.set_xticklabels([f'D{i:0>2}' for i in x_design_avg], fontsize=config.FONTS)
    lbl = 'n axons targetgrown - cross-grown'
    ax.set_ylabel(lbl, fontsize=config.FONTS)
    ax.tick_params(labelsize=config.SMALL_FONTS)
    
    ax.hlines(0, x_design_avg[0]-.75, x_design_avg[-1]+.75, color='k', linewidth=1)
    ax.vlines([*x_design_left,*x_design_right], -.8, .8, color='k', linewidth=.5)
    
    # finally draw the number of axons
    ax.bar(x_design_avg, destinations_design_avg.metric, color=config.DESIGN_CMAP(x_design_avg-1), zorder=1)
    # ax.scatter(x_design, destinations.metric, marker='1', s=80, color='k', zorder=2)
    ax.scatter(x_design_left, destinations.target, marker='+', s=50, color='k', zorder=2, linewidth=1)
    ax.scatter(x_design_right, destinations.cross_growth, marker='_', s=50, color='k', zorder=2, linewidth=1)
    
    return _save_plot(show, dest_dir, 'destinations', DIV_range=None, fname_postfix=fname_postfix)
    

def plot_growth_directions(directions, dest_dir, show=False, fname_postfix=''):
    # average over designs
    print(directions)
    # directions['metric'] = directions.target - directions.cross_growth
    # print(directions)

    min_dist = 500
    # directions[directions.abs()<500] = np.nan
    # print((directions>min_dist))
    # print((directions>min_dist).sum())
    naxons = directions.notna().sum(1)
    print(naxons)
    directions.fillna(0, inplace=True)
    # print(directions)
    # print(directions< -min_dist)
    # directions = (directions< -min_dist).sum(1)

    # directions = pd.DataFrame(((directions< -min_dist).sum(1), (directions>min_dist).sum(1)), 
    directions = pd.DataFrame(((directions< -min_dist).sum(1) /naxons.values, (directions>min_dist).sum(1)/naxons.values), 
                               columns=directions.index, index=['n_correct_dir','n_incorrect_dir']).T
    print(directions)
    # directions = directions/naxons.values
    directions['metric'] = directions['n_correct_dir'] - directions['n_incorrect_dir']
    print(directions)


    directions_design_avg = directions.groupby(level='design').mean()
    print(directions_design_avg)
    # xlocs
    x_design_avg = directions.index.unique('design')
    x_design_left = directions.index.get_level_values('design')-.15
    x_design_right = directions.index.get_level_values('design')+.15
    
    # get the figure
    fig, ax = _get_barplot(n_bars=len(directions_design_avg))

    # set ticks
    ax.set_xlim((x_design_avg[0]-.75, x_design_avg[-1]+.75))
    ax.set_xticks(x_design_avg)
    ax.set_xticklabels([f'D{i:0>2}' for i in x_design_avg], fontsize=config.FONTS)
    lbl = 'n axons targetgrown - cross-grown'
    ax.set_ylabel(lbl, fontsize=config.FONTS)
    ax.tick_params(labelsize=config.SMALL_FONTS)
    
    ax.hlines(0, x_design_avg[0]-.75, x_design_avg[-1]+.75, color='k', linewidth=1)
    # ax.vlines([*x_design_left,*x_design_right], -.8, .8, color='k', linewidth=.5)
    
    # finally draw the number of axons
    ax.bar(x_design_avg, directions_design_avg.metric, color=config.DESIGN_CMAP(x_design_avg-1), zorder=1)
    # ax.scatter(x_design, destinations.metric, marker='1', s=80, color='k', zorder=2)
    ax.scatter(x_design_left, directions.n_correct_dir, marker='+', s=50, color='k', zorder=2, linewidth=1)
    ax.scatter(x_design_right, directions.n_incorrect_dir, marker='_', s=50, color='k', zorder=2, linewidth=1)
    
    return _save_plot(show, dest_dir, 'growth_direction', DIV_range=None, fname_postfix=fname_postfix)
    








    # flatten which dataset/ timelapse dim and design replicate dim
    directions = directions.unstack(level=('timelapse', 'CLSM_area'))
    print(directions)
    
    # # rank    
    # order = growth_speeds_avg.sort_values(ascending=True).index
    # growth_speeds_avg = growth_speeds_avg.reindex(order)
    # growth_speeds = growth_speeds.reindex(order)

    # xlocs
    x_design_avg = directions.index.unique()
    x_design_left = directions.index.get_level_values('design')-.15
    x_design_right = directions.index.get_level_values('design')+.15
    
    # get the figure
    fig, ax = _get_barplot(n_bars=directions.shape[0])

    # set ticks
    ax.set_xlim((x_design_avg.min()-.75, x_design_avg.max()+.75))
    ax.set_xticks(x_design_avg)
    ax.set_xticklabels([f'D{i:0>2}' for i in x_design_avg], fontsize=config.FONTS)
    lbl = 'Axon delta distance to target' + config.um_d
    # if DIV_range:
    #     lbl += f' DIV{DIV_range[0]}-{DIV_range[1]}'
    ax.set_ylabel(lbl, fontsize=config.FONTS)
    ax.tick_params(labelsize=config.SMALL_FONTS)

    # finally draw the number of axons
    for x in x_design_avg:
        Y = directions.loc[x,:].dropna().sort_values()
        print(Y)
        Y = Y[Y.abs()>500]
        print(Y)
        n_good_ones = (Y<0).sum() *100
        n_bad_ones = (Y>0).sum() *100
        ax.scatter(x-.15, n_good_ones, marker='+', color='k', zorder=2, s=50, linewidth=1)
        ax.scatter(x+.15, n_bad_ones, marker='_', color='k', zorder=2, s=50, linewidth=1)
        ax.boxplot(Y, widths=.8, patch_artist=True, positions=(x,),
                   flierprops={'markerfacecolor':'k', 'marker':'.', 'markersize':3}, 
                   medianprops={'color':'k'},
                   boxprops={'facecolor': config.DESIGN_CMAP(x-1)},
        )   
    
    return _save_plot(show, dest_dir, 'growth_direction', None, fname_postfix)


def _save_plot(show, dest_dir, base_fname, DIV_range, fname_postfix):
    if show:
        plt.show()
    else:
        fname = f'{dest_dir}/{base_fname}'
        if DIV_range:
            fname += f'_DIV{DIV_range[0]}-{DIV_range[1]}'
        plt.savefig(f'{fname}{fname_postfix}.{config.FIGURE_FILETYPE}')
        plt.close()
        return fname