import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from matplotlib.animation import ArtistAnimation, writers

import pandas as pd
import numpy as np

from config import TRAIN_Ps, TEST_Ps

from AxonDetections import AxonDetections


def plot_preprocessed_input_data(preporc_dat_csv, dest_dir=None, show=False):
    data = pd.read_csv(preporc_dat_csv, header=[0,1,2], index_col=0)

    fig = plt.figure(figsize=(18,9))
    fig.subplots_adjust(hspace=.4, wspace=.05, top=.95, bottom=.05, left= .03, right=.97)
    widths = [1, 1, 0.3, 1, 1]
    heights = [1, 1, 1]
    spec = fig.add_gridspec(ncols=5, nrows=3, width_ratios=widths,
                            height_ratios=heights)
    axes = [fig.add_subplot(spec[row,col]) for row in (0,1,2) for col in (0,1,3,4)]

    for i, name in enumerate(data.columns.unique(1)):
        for j, which_timpoint in enumerate(('t_0', 't_-1')):
            ax = axes[i*2+j]

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(color='#d1d1cf', zorder=0)
            ax.tick_params(axis='both', labelsize=8)
            ax.set_title(f'{which_timpoint}-data: {name}', fontsize=9)
            if j:
                ax.tick_params(labelleft=False, labelright=True)
            
            train = data[('train', name, which_timpoint)]
            test = data[('test', name, which_timpoint)]
            if name.startswith('Standardized'):
                args = {'bins': 140, }
                ax.set_ylim((0, 0.12))
                ax.set_xlim((-.5, 14))
            elif 'Motion' in name:
                args = {'bins': 100, }
                ax.set_ylim((0, 0.02))
                ax.set_xlim((-.5, 14))
            else:
                args = {'bins': 140,}
                ax.set_ylim((0, 12))
                ax.set_xlim((-.01, .09))
            
            train_sprsty = f', sparsity: {(train==0).sum()/len(train):.3f}'
            ax.hist(train, color='blue', alpha=.5, density=True, zorder=3,
                            label='train'+train_sprsty, **args)
            test_sprsty = f', sparsity: {(test==0).sum()/len(test):.3f}'
            ax.hist(test, color='red', alpha=.5, density=True, zorder=3,
                            label='test'+test_sprsty, **args)
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
        metric = pd.read_pickle(mtrc_file)
        x_epochs = metric.columns.unique(0)

        # iterate the different plots/ value groups to plot
        for i, ax in enumerate(axes):
            if i not in [0,4]:
                ax.tick_params(labelleft=False)

            # loss plots: xy anchor, box loss, no-box loss, summed loss
            if i<4:
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
                ax.set_ylim(0.45,.95)
                ax.set_yticks(np.arange(0.4,.96, .1))
                ax.set_xlabel('Epoch')
                
                # get one specific metrics
                name = metric.index[i-4]
                trn_mtrc = metric.loc[name, (slice(None), 'train')].droplevel((1,2))
                tst_mtrc = metric.loc[name, (slice(None), 'test')].droplevel((1,2))
                # smooth 
                trn_data = trn_mtrc.ewm(span=25).mean()
                tst_data = tst_mtrc.ewm(span=25).mean()

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
                    loc='upper left', fontsize=14)
    
    if show:
        plt.show()
    if dest_dir is not None:
        fname = f'{dest_dir}/training{file_name_apdx}.png'
        fig.savefig(fname, dpi=450)

def draw_frame(image, det1=None, det2=None, fname='image', dest_dir=None,  lbl='',
               animation=None, boxs=70, draw_grid=None, show=False, color_det1_ids=False, 
               color_det2_ids=True, det1_default_col='w', det2_default_col='w', 
               draw_astar_paths=None):
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
        im = np.stack([r, emptychannel, emptychannel], axis=2)
        
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

    if draw_astar_paths == 'whattttt?:)':
        pass
    
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

                axon_box = Rectangle((x-boxs/2, y-boxs/2), boxs, boxs, **kwargs)
                rectangles.append(axon_box)
    [ax.add_patch(rec) for rec in rectangles]

    im_artist = ax.imshow(im)
    text_artists.append(ax.text(.05, .95, lbl, transform=fig.transFigure, 
                        fontsize=height/200, color='w'))
    if animation:
        return [im_artist, *text_artists, *rectangles]
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
        metric = np.stack([df.values for df in metrics]).mean(0)
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


def draw_all(axon_dets, filename, dest_dir=None, notes='', show=False, filter2FP_FN=False,
             save_single_tiles=False, animated=False, hide_det1=False, 
             hide_det2=False, **kwargs):
    if not dest_dir:
        dest_dir = axon_dets.dir
    if animated:
        anim_frames = []
        animated = plt.subplots(1, figsize=(axon_dets.dataset.sizex/350, 
                                axon_dets.dataset.sizey/350), facecolor='#242424')

    # iterate the timepoints
    for t in range(len(axon_dets)):
        fname = filename.replace('---', f'{t:0>3}') 
        prc, rcl, F1 = axon_dets.get_detection_metrics(t)
        lbl = f'{notes} - {fname} - Recall: {rcl}, Precision: {prc}, F1: {F1}'
        print(lbl, end='...', flush=True)

        img_tiled, img, tiled_true_det, image_true_det = axon_dets.get_det_image_and_target(t)
        det1, det2 = axon_dets.detections[t], image_true_det

        if filter2FP_FN:
            FP_dets, FN_dets = axon_dets.get_FP_FN_detections(t)
            det1, det2 = FP_dets, FN_dets
            kwargs['det1_default_col'] = '#e34d20'  # red-ish orange for false pos
            kwargs['det2_default_col'] = '#0cb31d'  # green for false neg
            # ensure that id colording is set to False
            kwargs['color_det2_ids'] = False 
            kwargs['color_det1_ids'] = False 
        if hide_det1:
            det1 = None
        if hide_det2:
            det2 = None
        # draw stitched frame
        frame_artists = draw_frame(img, det1, det2, animation=animated, 
                                   dest_dir=dest_dir, draw_grid=axon_dets.tilesize, 
                                   fname=fname, lbl=lbl, show=show, **kwargs)
        if animated:
            anim_frames.append(frame_artists)

        # also save the single tile images
        if save_single_tiles:
            # for the current timepoints, iter non-empty tiles
            n_tiles = len(img_tiled)
            for tile_i in range(n_tiles):
                img, target = img_tiled[tile_i], tiled_true_det[tile_i],
                draw_frame(img, axon_dets.pandas_tiled_dets[t][tile_i], target, 
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
                 writer=writers['imagemagick'](fps=1))
        print(f'{axon_dets.dataset.name} animation saved.')
    print(' - Done.')
























def plot_axon_IDs(axon_dets, dest_dir, show=False):
    id_lifetime = axon_dets.all_detections.loc[:, (slice(None),'conf')].droplevel(1,1)
    id_lifetime = id_lifetime.fillna(0).astype(bool).T

    plt.figure(figsize=(13,7))
    plt.imshow(id_lifetime, cmap=plt.get_cmap('cividis'))
    ax = plt.gca()

    ax.set_yticks(np.arange(len(axon_dets)))
    ax.set_yticklabels(['' if t%4 else t for t in range(len(axon_dets))])
    ax.set_ylabel('timepoint', fontsize=12)

    ax.set_xticks(np.arange(id_lifetime.shape[1]))
    ax.set_xticklabels(['' if t%10 else t for t in range(id_lifetime.shape[1])])
    ax.set_xlabel('Axon ID', fontsize=12)

    lbl = f'{axon_dets.dataset.name} - alpha:{axon_dets.alpha}, beta:{axon_dets.beta}'
    ax.set_title(lbl,  fontsize=12)

    if show:
        plt.show()
    if dest_dir:
        plt.savefig(f'{dest_dir}/{axon_dets.dataset.name}_id_lifetime_a{axon_dets.alpha}_b{axon_dets.beta}.png')
        

def plot_dist_to_target(axon_dets, dest_dir, show=False):
    dets = axon_dets.get_dists_to_target(window_size=1)
    print(dets)

    fig, ax = plt.subplots(figsize=(7, 4), facecolor='k',)
    ax.set_facecolor('k')

    ymin, ymax = dets.min().min()-200, dets.max().max()+200
    xmin, xmax = -1, len(axon_dets)+1
    ax.set_ylim(ymin, ymax)
    ax.set_xlim(xmin, xmax)

    ax.axline((0, ymin), (0, ymax), color='grey', linewidth=1)
    ax.axline((xmin, 0), (xmax, 0), color='grey', linewidth=1)
    ax.vlines(range(1,len(axon_dets)), ymax+10, ymax+20, color='grey', linewidth=1)
    ax.set_ylabel('distance to outputchannel', color='white', fontsize=12)
    ax.text(-1, 0, 'increases                  decreases', ha='center', va='center', color='white', fontsize=10, rotation=90)
    
    
    ax.plot(dets.T, colo)
    # for t in range(len(axon_dets)):
    #     dets[t]

    plt.show()


