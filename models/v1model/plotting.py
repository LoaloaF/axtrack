import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from matplotlib.animation import ArtistAnimation, writers

from skimage.filters import gaussian
from skimage.morphology import dilation
from skimage.morphology import square

import pandas as pd
import numpy as np

from config import TRAIN_Ps, TEST_Ps, VIDEO_ENCODER

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

    for i, which_preproc in enumerate(data.columns.unique(1)):
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
                dat = data[(dat_name, which_preproc, which_timpoint)]
                
                if which_preproc.startswith('Standardized'):
                    args = {'bins': 2**10, 'range':(0, 60)}
                    ax.set_ylim((0, 0.1))
                    ax.set_xlim((-.5, 20))
                elif 'Motion' in which_preproc:
                    args = {'bins': 2**10, }
                    ax.set_ylim((0, 0.1))
                    ax.set_xlim((-.5, 14))
                else:
                    args = {'bins': 2**10, 'range':(0, 1/2**4)} # image data is saved in 12bit format (0-4095)
                    ax.set_ylim((0, 60))
                    ax.set_xlim((-.01, .09))
                
                dat_sprsty = f', sparsity: {(dat==0).sum()/len(dat):.3f}'
                ax.hist(dat, alpha=.5, density=True, zorder=3,
                                label=dat_name+dat_sprsty, **args)
            ax.legend(fontsize=8)
    if show:
        plt.show()
    if dest_dir is not None:
        fname = f'{dest_dir}/preprocessed_data.png'
        fig.savefig(fname, dpi=600)
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
               draw_axons=None):
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
        g = im[:,:,0]
        im = np.stack([emptychannel, g, emptychannel], axis=-1)
        
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
    if draw_grid:
        gridsize = round(draw_grid)
        at_x = [gridsize*i for i in range(width//gridsize+1)]
        at_y = [gridsize*i for i in range(width//gridsize+1)]
        ax.vlines(at_x, 0, height, color='white', linewidth=.5, alpha=.2)
        ax.hlines(at_y, 0, width, color='white', linewidth=.5, alpha=.2)
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)

    # draw bounding boxes
    rectangles = []
    text_artists = []
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
                        if i == 0:
                            axons_img_rgba = np.zeros((height, width, 4))
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
        alpha = np.maximum(0, (axons_img_rgb.max(-1)-.2))
        axons_img_rgba = np.append(axons_img_rgb, alpha[:,:,np.newaxis], -1)
        im2_artist = ax.imshow(axons_img_rgba)
        artists = [im2_artist, *artists]
    if animation:
        return artists
    if show:
        plt.show()
    if dest_dir:
        fig.savefig(f'{dest_dir}/{fname}.jpg', dpi=450)
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

def draw_axon_reconstruction(reconstructions, img_shape, color='green'):
    draw_y = reconstructions.loc[:, (slice(None),'Y')].unstack().dropna().astype(int)
    draw_x = reconstructions.loc[:, (slice(None),'X')].unstack().dropna().astype(int)

    drawn_axons = np.zeros((img_shape), float)
    drawn_axons[draw_y, draw_x] = 1
    drawn_axons = dilation(drawn_axons, square(5))
    drawn_axons = gaussian(drawn_axons, 3, preserve_range=True)

    drawn_axons_rgba = np.stack([drawn_axons, *[np.zeros_like(drawn_axons)]*2, 
                                 drawn_axons], -1)
    return drawn_axons_rgba

def draw_all(axon_dets, filename, dest_dir=None, notes='', show=False, 
             filter2FP_FN=False, save_single_tiles=False, animated=False, 
             hide_det1=False, hide_det2=False, use_IDed_dets=False, 
             draw_axons=None, which_axons=None, **kwargs):
    if not dest_dir:
        dest_dir = axon_dets.dir
    if animated:
        anim_frames = []
        animated = plt.subplots(1, figsize=(axon_dets.dataset.sizex/350, 
                                axon_dets.dataset.sizey/350), facecolor='#242424')
    which_det = 'confident' if not use_IDed_dets else 'ID_assigned'
    
    # iterate the timepoints
    for t in range(len(axon_dets)):
        fname = filename.replace('---', f'{t:0>3}') 
        prc, rcl, F1 = axon_dets.get_detection_metrics(t, which_det=which_det)
        lbl = f'{notes} - {fname} - Recall: {rcl}, Precision: {prc}, F1: {F1}'
        print(lbl, end='...', flush=True)

        img_tiled, img, tiled_true_det, image_true_det = axon_dets.get_det_image_and_target(t)
        det1, det2 = axon_dets.get_confident_det(t), image_true_det

        # which detections exactly to draw
        if use_IDed_dets:
            det1 = axon_dets.get_IDed_det(t)
            if draw_axons is not None:
                if t == 0:
                    ax_recstrs = pd.DataFrame([])
                else:
                    ax_recstrs = pd.concat([ax_recstrs, axon_dets.get_axon_reconstructions(t)], axis=1)
        elif filter2FP_FN:
            FP_dets, FN_dets = axon_dets.get_FP_FN_detections(t)
            det1, det2 = FP_dets, FN_dets
            kwargs['det1_default_col'] = '#e34d20'  # red-ish orange for false pos
            kwargs['det2_default_col'] = '#0cb31d'  # green for false neg
            # ensure that id coloring is set to False
            kwargs['color_det2_ids'] = False 
            kwargs['color_det1_ids'] = False 
        if hide_det1:
            det1 = None
        if hide_det2:
            det2 = None
        if which_axons is not None:
            det1 = det1.drop([ax for ax in det1.index if ax not in which_axons])
        

        # draw stitched frame
        frame_artists = draw_frame(img, det1, det2, animation=animated, 
                                   dest_dir=dest_dir, draw_grid=axon_dets.tilesize, 
                                   fname=fname, lbl=lbl, show=show, 
                                   draw_axons=ax_recstrs, **kwargs)
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
        ani = ArtistAnimation(animated[0], anim_frames, interval=1000, blit=True, 
                              repeat_delay=200)
        if show:
            plt.show()
        print('encoding animation...')
        ani.save(f'{dest_dir}/{axon_dets.dataset.name}_assoc_dets.mp4', 
                 writer=writers[VIDEO_ENCODER](fps=1))
        print(f'{axon_dets.dataset.name} animation saved.')
    print(' - Done.')
























def plot_axon_IDs_lifetime(axon_dets, dest_dir=None, show=False):
    id_lifetime = axon_dets.IDed_detections.Width.astype(bool).unstack()
    id_lifetime.fillna(False, inplace=True)

    plt.figure(figsize=(13,7))
    plt.imshow(id_lifetime, cmap=plt.get_cmap('cividis'))
    ax = plt.gca()

    ax.set_yticks(np.arange(len(axon_dets)))
    ax.set_yticklabels(['' if t%4 else t for t in range(len(axon_dets))])
    ax.set_ylabel('timepoint', fontsize=12)

    ax.set_xticks(np.arange(id_lifetime.shape[1]))
    ax.set_xticklabels(['' if t%10 else t for t in range(id_lifetime.shape[1])])
    ax.set_xlabel('Axon ID', fontsize=12)

    lbl = f'{axon_dets.dataset.name}'
    ax.set_title(lbl,  fontsize=12)

    if show:
        plt.show()
    if dest_dir:
        plt.savefig(f'{dest_dir}/{axon_dets.dataset.name}_id_lifetime.png')
        

def plot_dist_to_target(screen, dest_dir, show=False):
    dets = screen.get_dists_to_target()

    fig, ax = plt.subplots(figsize=(7, 4), facecolor='k',)
    ax.set_facecolor('k')

    ymin, ymax = dets.min().min()-200, dets.max().max()+200
    xmin, xmax = -1, len(screen)+1
    ax.set_ylim(ymin, ymax)
    ax.set_xlim(xmin, xmax)

    ax.axline((0, ymin), (0, ymax), color='grey', linewidth=1)
    ax.axline((xmin, 0), (xmax, 0), color='grey', linewidth=1)
    ax.vlines(range(1,len(screen)), ymax+10, ymax+20, color='grey', linewidth=1)
    ax.set_ylabel('distance to outputchannel', color='white', fontsize=12)
    ax.text(-1, 0, 'increases                  decreases', ha='center', va='center', color='white', fontsize=10, rotation=90)
    
    ax.plot(dets.T)
    if show:
        plt.show()
