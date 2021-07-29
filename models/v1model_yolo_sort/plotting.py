import matplotlib.pyplot as plt
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
        for j, which_data in enumerate(('train', 'test')):
            ax = axes[i*2+j]

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(color='#d1d1cf', zorder=0)
            ax.tick_params(axis='both', labelsize=8)
            ax.set_title(f'{which_data}-data: {name}', fontsize=9)
            if j:
                ax.tick_params(labelleft=False, labelright=True)
            
            t0 = data[(which_data, name, 't_0')]
            tn1 = data[(which_data, name, 't_-1')]
 
            if name.startswith('Standardized'):
                args = {'bins': 140, }
                ax.set_ylim((0, 0.12))
            elif 'Motion' in name:
                args = {'bins': 100, }
                ax.set_ylim((0, 0.02))
                ax.set_xlim((-.5, 14))
            else:
                args = {'bins': 140,}
                ax.set_ylim((0, 12))
                ax.set_xlim((-.01, .09))
            
            ax.hist(t0, color='blue', alpha=.5, density=True, zorder=3,
                            label='t_0', **args)
            ax.hist(tn1, color='red', alpha=.5, density=True, zorder=3,
                            label='t_-1', **args)
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
        print(training_file)
        data = pd.read_pickle(training_file).loc[:,x_epochs]
        
        # datafile2 =  '/home/loaloa/gdrive/projects/biohybrid MEA/tl140_outputdata//runs/v1Model_exp3_newcnn//run12_28.07.2021_21.47.22/metrics//all_epochs.pkl'
        # data2 = pd.read_pickle(datafile2).loc[:,x_epochs]
        # data2 = data2.stack(level=(-1,-2))
        # data2.columns = data2.columns.values+301
        # data2 = data2.unstack(level=(-1,-2))
        # data = pd.concat([data, data2], axis=1)
            
        x_epochs = data.columns.unique(0)

        # for i, name in enumerate(data.index):
        for i in range(data.shape[0]+1):
            if i<data.shape[0]:
                name = data.index[i]
            ax = axes[i]
            ax.axis('on')
            ax.set_facecolor('#eeeeee')
            ax.grid()
            ax.set_title(name)

            # collapsing batch dim by averaging
            train_dat = data.loc[name, (slice(None), 'train')].droplevel(1).unstack().mean(1)
            test_dat = data.loc[name, (slice(None), 'test',)].droplevel(1).unstack().mean(1)

            if i in (0,1,2):
                ax.set_ylim(0,15)
                if i:
                    ax.tick_params(left=False, labelleft=False)
            else:
                ax.set_ylim (0,40)

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
                for data_i in range(2):
                    loss, nbatches, color = losses[data_i], nbatchess[data_i], colors[data_i]
                    if split_test_train and data_i == 0 and i == 5:
                        continue

                    loss_allsteps = loss.stack()
                    # flat index but still in range 0-max_epoch
                    loss_allsteps.index = [E+ b/(nbatches[E]+1) for E,b in loss_allsteps.index]
                    
                    # plot indicate where new epoch starts
                    ax.vlines(x_epochs, 0, 40, linewidth=.8, alpha=.4, color='k')
                    # plot every single batch loss
                    ax.plot(loss_allsteps.index, loss_allsteps, 'o-', alpha=.6,
                            markersize=.6, linewidth=.5, color=color, )
                    # plot how many batches the epoch had
                    for epoch_idx, nb in nbatches.iteritems():
                        col = 'purple' if split_test_train else color
                        ax.plot((epoch_idx, epoch_idx+1), (nb,nb), 
                                color=col, linewidth=2, alpha=.7)

                    if split_test_train and data_i == 0 and i == 4:
                        break
                # if not split_test_train and i == 4:
                #     continue
                    # else:
                    #     ax.boxplot(train_loss.dropna(1).T, boxprops={'linewidth':.3}, 
                    #             whiskerprops={'linewidth':.3}, showfliers=False,
                    #             medianprops={'color':train_col})
                    #     ax.scatter(x_epochs+1, nbatches, label='test', s=8, marker='o', 
                    #                     linewidth=1, alpha=.5, color='purple')

            else:
                # smoothing curve
                train_dat_smoo = train_dat.ewm(span=20).mean()
                test_dat_smoo = test_dat.ewm(span=20).mean()
                
                # plot unsmoothed
                lines = ax.plot(x_epochs, train_dat,  label='train', linewidth=1, alpha=.3)
                train_col = lines[-1].get_color()
                ax.plot(x_epochs, train_dat_smoo, label='train', linewidth=1, color=train_col)
                
                # plot smoothed
                lines = ax.plot(x_epochs, test_dat,  label='test', linewidth=1, alpha=.3)
                test_col = lines[-1].get_color()
                ax.plot(x_epochs, test_dat_smoo, label='test', linewidth=1, color=test_col)
                ax.legend()

            ax.set_xticks(np.arange(0,len(x_epochs)+1,10))
            ax.set_xticklabels(np.arange(0,len(x_epochs)+1,10), rotation=37, ha='center', va='center', fontsize=8, y=-.02)
            ax.set_xlabel('Epoch')
            ax.set_xlim(-1, len(x_epochs)+1)
            
    if show:
        plt.show()
    if dest_dir is not None:
        fname = f'{dest_dir}/loss_over_time.png'
        if len(training_files) == 2:
            fname = f'{dest_dir}/loss_over_time_compare.png'
        fig.savefig(fname, dpi=600)


def draw_tile(tile, target_anchors=None, pred_anchors=None, tile_axis=None, 
              dest_dir=None, fname='tile', draw_YOLO_grid=None, show=False):
    im = np.array(tile)
    height, width, cchannels = im.shape
    emptychannel = np.zeros([height,width])
    if cchannels == 2:
        g, b = im[:,:,0], im[:,:,1]
        im = np.stack([emptychannel, g, b], axis=2)
    elif cchannels == 1:
        r = im[:,:,0]
        im = np.stack([emptychannel, r, emptychannel], axis=2)
        
    # matplotlib likes float images in range [0,1]
    min_val = im[im>0].min() if im.any().any() else 0
    im = np.where(im>0, im-min_val, 0)  # shift distribution to 0
    im = np.where(im>1, 1, im)  # ceil to 1
    
    if dest_dir or tile_axis is None:
        small_fig, ax = plt.subplots(1, figsize=(5,5), facecolor='#242424')
        small_fig.subplots_adjust(wspace=0, hspace=0, top=.99, bottom=.01, 
                                  left=.01, right=.99)
        if tile_axis is None:
            axes = [ax, None]
        if dest_dir:
            axes = [ax, tile_axis]
    else:
        axes = [None, tile_axis]

    for ax_i, ax in enumerate(axes):
        if ax is None:
            continue

        # Create figure and axes
        ax.axis('off')
        ax.imshow(im)

        if draw_YOLO_grid is not None: 
            xlines = np.linspace(0, width, draw_YOLO_grid[0])
            ylines = np.linspace(0, height, draw_YOLO_grid[1])
            ax.hlines(ylines, 0, width, color='white', linewidth=.5, alpha=.2)
            ax.vlines(xlines, 0, height, color='white', linewidth=.5, alpha=.2)  
            ax.set_xlim(0, width)
            ax.set_ylim(height, 0)
        
        if target_anchors is not None and not target_anchors.empty:
            s = 1600 if ax_i==0 else 140
            lw = 3 if ax_i==0 else 1
            kwargs = {"color":'white', "linewidth":lw, "linestyle":'solid',
                    's':s, 'facecolor':'none'}
            ax.scatter(target_anchors.anchor_x, target_anchors.anchor_y, **kwargs)
        
        if pred_anchors is not None and not pred_anchors.empty:
            s = 2300 if ax_i==0 else 180
            kwargs = {"color":'white', "linestyle":'dashed', 's':s, 
                    'facecolor':'none'}
            for r, box in pred_anchors.iterrows():
                kwargs['alpha'] = box.conf if box.conf <1 else 1
                kwargs['linewidth'] = 3*box.conf if ax_i==0 else 1.3*box.conf
                ax.scatter(box.anchor_x, box.anchor_y, **kwargs)

        if ax_i == 0 and show:
            plt.show()
        if ax_i == 0 and dest_dir:
            small_fig.savefig(f'{dest_dir}/{fname}.jpg')
            plt.close(small_fig) 
    return axes[-1]