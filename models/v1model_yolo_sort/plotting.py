import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import pandas as pd
import numpy as np

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
        # for j, which_data in enumerate(('train', 'test')):
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


def plot_training_process(training_files, parameter_list, dest_dir=None, 
                          split_test_train=True, show=False): #boxplot=False):
    fig, axes = plt.subplots(2,3, figsize=(19,9), sharex=True)
    fig.subplots_adjust(wspace=.03, left=.03, right=.97)
    axes = list(axes.flatten())

    if not split_test_train:
        # get size  of 2 bottom right plots
        gs = axes[-2].get_gridspec()
        # remove them
        [axes.pop(-1).remove() for _ in range(2)]
        # add one big one
        axes.append(fig.add_subplot(gs[-2:]))

    [ax.axis('off') for ax in axes]
    x_epochs = pd.read_pickle(training_files[0]).columns.unique(0)

    main_title = ''
    for training_file, parameters in zip(training_files, parameter_list):
        main_title = main_title+'  -   '+parameters['NOTES']
        fig.suptitle(main_title)
        data = pd.read_pickle(training_file).loc[:,x_epochs]
        x_epochs = data.columns.unique(0)

        # for i, name in enumerate(data.index):
        for i in range(6):
            if i<data.shape[0]:
                name = data.index[i]
            ax = axes[i]
            ax.axis('on')
            ax.set_facecolor('#eeeeee')
            ax.grid()
            ax.set_title(name)

            # collapsing batch dim by averaging
            train_dat = data.loc[name, (slice(None), 'train')].droplevel(1).unstack().mean(1)
            train_tpr = data.loc['total_pos_labels_rate', (slice(None), 'train')].droplevel(1).unstack().mean(1) 
            train_dat_tpr_adjust = (train_dat/train_tpr) *(train_tpr).mean()
            test_dat = data.loc[name, (slice(None), 'test',)].droplevel(1).unstack().mean(1)
            test_tpr = data.loc['total_pos_labels_rate', (slice(None), 'test')].droplevel(1).unstack().mean(1) 
            test_dat_tpr_adjust = (test_dat/test_tpr) *(test_tpr).mean()

            if i in (0,1,2):
                ax.set_ylim(0,15)
                if i:
                    ax.tick_params(left=False, labelleft=False)
            else:
                pass
                # ax.set_ylim (0,40)

            if i in (4, 5):
                if split_test_train:
                    ax = axes[i]
                ax.tick_params(labelleft=False, labelright=True)
                ax.grid(False)
                ax.set_title('variance in training')

                # get the sum and make the batches dim columns
                train_loss = data.loc['total_summed_loss', (slice(None), 'train')]
                train_loss = train_loss.droplevel(1).unstack().sort_index(axis=1)
                train_nbatches = train_loss.notna().sum(1)

                test_loss = data.loc['total_summed_loss', (slice(None), 'test')]
                test_loss = test_loss.droplevel(1).unstack().sort_index(axis=1)
                test_nbatches = test_loss.notna().sum(1)

                losses, nbatchess, colors = (train_loss, test_loss), (train_nbatches, test_nbatches), (train_col, test_col)
               
               
                # for data_i in range(2):
                #     loss, nbatches, color = losses[data_i], nbatchess[data_i], colors[data_i]
                #     if split_test_train and data_i == 0 and i == 5:
                #         continue

                #     loss_allsteps = loss.stack()
                #     # flat index but still in range 0-max_epoch
                #     loss_allsteps.index = [E+ b/(nbatches[E]+1) for E,b in loss_allsteps.index]
                    
                #     # plot indicate where new epoch starts
                #     ax.vlines(x_epochs, 0, 40, linewidth=.8, alpha=.4, color='k')
                #     # plot every single batch loss
                #     ax.plot(loss_allsteps.index, loss_allsteps, 'o-', alpha=.6,
                #             markersize=.6, linewidth=.5, color=color, )
                #     # plot how many batches the epoch had
                #     for epoch_idx, nb in nbatches.iteritems():
                #         col = 'purple' if split_test_train else color
                #         ax.plot((epoch_idx, epoch_idx+1), (nb,nb), 
                #                 color=col, linewidth=2, alpha=.7)

                #     if split_test_train and data_i == 0 and i == 4:
                #         break
                
                if split_test_train and i == 4:
                    precision = data.loc['total_precision_0.80', (slice(None), 'train')]
                    precision = precision.droplevel(1).unstack().sort_index(axis=1).mean(1)
                    precision = precision.ewm(span=25).mean()
                    recall = data.loc['total_recall_0.80', (slice(None), 'train')]
                    recall = recall.droplevel(1).unstack().sort_index(axis=1).mean(1)
                    recall = recall.ewm(span=25).mean()
                    ax.plot(recall, color=train_col, alpha=.5, label='train recall')
                    ax.plot(precision, color=train_col, alpha=.5, linestyle='dashed', label='train precision')
                    
                    precision = data.loc['total_precision_0.80', (slice(None), 'test')]
                    precision = precision.droplevel(1).unstack().sort_index(axis=1).mean(1)
                    precision = precision.ewm(span=25).mean()
                    recall = data.loc['total_recall_0.80', (slice(None), 'test')]
                    recall = recall.droplevel(1).unstack().sort_index(axis=1).mean(1)
                    recall = recall.ewm(span=25).mean()
                    ax.plot(recall, color=test_col, alpha=.5, label='test recall')
                    ax.plot(precision, color=test_col, alpha=.5, linestyle='dashed', label='test recall')
                    ax.legend()

            else:
                train_dat = train_dat.ewm(span=18).mean()
                test_dat = test_dat.ewm(span=18).mean()
                train_dat_tpr_adjust = train_dat_tpr_adjust.ewm(span=18).mean()
                test_dat_tpr_adjust = test_dat_tpr_adjust.ewm(span=18).mean()
                

                
                # plot train
                lines = ax.plot(x_epochs, train_dat, linestyle='-.',  
                                label='train', linewidth=1, alpha=.8)
                train_col = lines[-1].get_color()
                ax.plot(x_epochs, train_dat_tpr_adjust, label='train, TPR normed.', 
                        linewidth=1.4, color=train_col, alpha=.8)
                
                # plot test
                lines = ax.plot(x_epochs, test_dat, linestyle='-.',  
                                label='test', linewidth=1, alpha=.8)
                test_col = lines[-1].get_color()
                ax.plot(x_epochs, test_dat_tpr_adjust, label='test, TPR normed.', 
                        linewidth=1.4, color=test_col, alpha=.8)
                ax.legend()
                
            ax.set_xticks(np.arange(0,len(x_epochs)+1,10))
            ax.set_xticklabels(np.arange(0,len(x_epochs)+1,10), rotation=37, 
                               ha='center', va='center', fontsize=8, y=-.02)
            ax.set_xlabel('Epoch')
            ax.set_xlim(-1, len(x_epochs)+1)
            
    if show:
        plt.show()
    if dest_dir is not None:
        fname = f'{dest_dir}/loss_over_time.png'
        if len(training_files) == 2:
            fname = f'{dest_dir}/loss_over_time_compare.png'
        fig.savefig(fname, dpi=600)

def draw_frame(image, target_anchors=None, pred_anchors=None, animation=None, 
               dest_dir=None, fname='image', draw_YOLO_grid=None, show=False,
               color_true_ids=False, color_pred_ids=True):
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
    if draw_YOLO_grid is not None:
        xlines = np.linspace(0, width, draw_YOLO_grid[0]+1)
        ylines = np.linspace(0, height, draw_YOLO_grid[1]+1)
        ax.hlines(ylines, 0, width, color='white', linewidth=.5, alpha=.2)
        ax.vlines(xlines, 0, height, color='white', linewidth=.5, alpha=.2)
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)
    
    # draw bounding boxes
    rectangles = []
    text_artists = []
    boxs = 70
    for i, anchors in enumerate((target_anchors, pred_anchors)):
        kwargs = {'edgecolor':'w', 'facecolor':'none', 'linewidth': 1, 
                'linestyle':'solid'}
        
        if anchors is not None:
            for axon_id, (conf, x, y) in anchors.iterrows():
                if i == 1:
                    kwargs.update({'alpha':conf, 'linestyle':'dashed'})
                if (i==0 and color_true_ids) or (i==1 and color_pred_ids):
                    ax_id_col = plt.cm.get_cmap('hsv', 20)(int(axon_id[-3:])%20)
                    kwargs.update({'edgecolor': ax_id_col})
                    text_artists.append(ax.text(x-boxs/2, y-boxs/1.5, axon_id.replace('on_',''), 
                                        fontsize=5.5, color=kwargs['edgecolor']))
                axon_box = Rectangle((x-boxs/2, y-boxs/2), boxs, boxs, **kwargs)
                rectangles.append(axon_box)
    [ax.add_patch(rec) for rec in rectangles]

    im_artist = ax.imshow(im)
    text_artists.append(ax.text(.05, .95, fname, transform=fig.transFigure, 
                        fontsize=height/200, color='w'))
    if animation:
        return [im_artist, *text_artists, *rectangles]
    if show:
        plt.show()
    if dest_dir:
        fig.savefig(f'{dest_dir}/{fname}.jpg', dpi=450)
        plt.close(fig) 