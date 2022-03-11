import pickle

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import config
from config import TRAIN_Ps, TEST_Ps

def plot_preprocessed_input_data(data, notes='', dest_dir=None, show=False, 
                                 fname_prefix=''):
    # data: each column represents one framem, one preprocessing step, 
    # 1 mio pixel intensity samples

    # create a large figure canvas
    fig = plt.figure(figsize=config.LARGE_FIGSIZE)
    fig.subplots_adjust(hspace=.4, wspace=.05, top=.95, left= .07, right=.97)
    # add subplots using a specified grid
    widths, heights = [1, 1, 0.3, 1, 1], [1, 1]
    spec = fig.add_gridspec(ncols=5, nrows=len(heights), width_ratios=widths,
                            height_ratios=heights)
    axes = [fig.add_subplot(spec[row,col]) for row in range(len(heights)) 
            for col in (0,1,3,4)]
    axes[0].text(.48,.03, 'Pixel intensity', transform=fig.transFigure, 
                 fontsize=config.FONTS)
    axes[0].text(.02,.5, 'Density', transform=fig.transFigure, rotation=90, 
                 fontsize=config.FONTS)

    which_preproc_steps = data.columns.unique(1)
    # when standardized label differs....
    if len(which_preproc_steps) == 7:
        which_preproc_steps = which_preproc_steps[:-1]
    for i, which_preproc in enumerate(which_preproc_steps):
        if 'Motion' in which_preproc:
            continue
        # iterate over first and last frames
        for j, which_timpoint in enumerate(('t_0', 't_-1')):
            ax = axes[i*2+j]

            # basics
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(alpha=.5, zorder=0)
            ax.tick_params(axis='both', labelsize=config.SMALL_FONTS)
            if j:
                ax.tick_params(labelleft=False)
            
            # title
            which_preproc_title = which_preproc if not which_preproc.startswith('Standardized') else 'Standardized'
            tit = which_timpoint if which_timpoint == 't_0' else 't_N'
            ax.set_title(f'{tit}-data: {which_preproc_title}', fontsize=config.FONTS)
            
            # setup yaxis
            ax.set_yscale('log')
            ax.set_ylim((1e-1, 8e6))
            yticks = ax.get_yticks().astype(int)
            ax.set_yticklabels([yt if yt else '' for yt in yticks])
            
            # iterate train and inference dataset
            for dat_name in data.columns.unique(0):
                # when standardized label differs....
                if which_preproc not in data[dat_name].columns.unique(0):
                    which_preproc = data[dat_name].columns.unique(0)[i]
                dat = data[(dat_name, which_preproc, which_timpoint)]
                
                if which_preproc.startswith('Standardized'):
                    ax.set_xlim((1e-2, 5e1))
                    xticks = [1e-2, 1e-1, 1e0, 5]

                else:
                    ax.set_xlim((1e-4, 1e0))
                    xticks = [1e-4, 1e-3, 1e-2, 1e-1, 1e0]
                
                # setup xaxis
                ax.set_xscale('log')
                ax.set_xticks(xticks)
                ax.set_xticklabels([xt if i else '' for i, xt in enumerate(xticks)])
                
                # draw the histogram
                lbl = f'{(dat==0).sum()/len(dat):.2f}'
                ax.hist(dat, alpha=.5, zorder=3, bins=2**8,
                        label='Train' if dat_name == 'train' else 'Infer')
        
        if i == 0:
            ax.legend(fontsize=config.SMALL_FONTS, loc='upper right', 
                      bbox_to_anchor=(1.03, 1.03))
    if show:
        plt.show()
    if dest_dir is not None:
        fname = f'{dest_dir}/{fname_prefix}preprocessed_data.{config.FIGURE_FILETYPE}'
        fig.savefig(fname)
    plt.close()

def plot_training_process(data, draw_best_thresholds=False, show=False, dest_dir=None):
    # create the figure, share epoch (x axis)
    fig, axes = plt.subplots(2,4, figsize=config.LARGE_FIGSIZE, sharex=True)
    fig.subplots_adjust(wspace=.12, hspace=.18, left=.05, right=.97, bottom=.1, 
                        top=.94)
    axes = axes.flatten()
    
    # iterate over different runs
    legend_elements  = []
    file_name_apdx = '_'
    for exp_run_id, data in data.items():
        # which metrics to plot
        toplot = data.columns.unique(0).drop('total_pos_labels_rate').tolist()
        toplot[2], toplot[3] = toplot[3], toplot[2]     # fix order

        # iterate the different plots/ value groups to plot
        for ax_i, which_plot in enumerate(toplot):
            ax = axes[ax_i]
            
            # general args for every axis
            ax.tick_params(labelsize=config.SMALL_FONTS, labelleft=False, 
                           left=False)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(True)
            tit = which_plot.replace('total_','').replace('_','-').capitalize()
            ax.set_title(tit, fontsize=config.FONTS)
            ax.set_xticklabels(ax.get_xticks().astype(int))
            
            # only plots on the left get yticklabels
            if ax_i in (0,4):
                ax.tick_params(labelleft=True, left=False)
            # loss plots: no-box loss, box loss, xy anchor, summed loss
            if 'loss' in which_plot:
                ylim = (0,11)
                span = 25
            # metrics plots (precision, recall, F1)
            elif which_plot in ( 'precision', 'recall', 'F1'):
                ax.set_xlabel('Epoch', fontsize=config.FONTS)
                ylim = (0.25,1.05)
                span = 25
            
            # yaxis
            ax.set_ylim(ylim)
            ax.set_yticklabels(ax.get_yticks().round(2))

            # get one specific loss + exponential smoothing 
            trn_plotdata = data.loc[:, (which_plot, 'train',)].ewm(span=span).mean()
            tst_plotdata = data.loc[:, (which_plot, 'test',)].ewm(span=span).mean()

            # finally draw the train and test data defined above
            ls = ax.plot(data.index, trn_plotdata, **TRAIN_Ps)
            trn_col = ls[-1].get_color()
            ls = ax.plot(data.index, tst_plotdata, color=trn_col, **TEST_Ps)
            tst_col = ls[-1].get_color()

        # last plot (8/8) has no data, legend is placed here
        axes[-1].axis('off')
        # save the run's identifying color to make legend in the end
        legend_elements.extend([
            Line2D([0], [0], label=f'Train', color=trn_col, **TRAIN_Ps),
            Line2D([0], [0], label=f'Test', color=tst_col, **TEST_Ps)])
        # for filename later
        if len(data) >1:
            file_name_apdx += exp_run_id[:5]

    # draw legend at last exes
    axes[-1].legend(handles=legend_elements, bbox_to_anchor=(0, 1), ncol=1,
                    loc='upper left', fontsize=config.FONTS)
    if show:
        plt.show()
    if dest_dir is not None:
        fname = f'{dest_dir}/training{file_name_apdx}.{config.FIGURE_FILETYPE}'
        fig.savefig(fname)

def plot_prc_rcl(data, dest_dir=None, show=None):
    fig, axes = plt.subplots(1, 2, figsize=config.MEDIUM_FIGSIZE)
    fig.subplots_adjust(bottom=.01, wspace=.3, right=.95)
    ticks = np.arange(0.5, .91, 0.2)
    lims = (.3, 1.04)

    # setup both axes
    for i, ax in enumerate(axes):
        # general setup
        ax.set_aspect('equal')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid()

        # yaxis setup
        ax.set_ylim(*lims) 
        ax.set_yticks(ticks)
        ax.set_yticklabels([f'{t:.1f}' for t in ticks], fontsize=config.SMALL_FONTS)

        # xaxis setup
        ax.set_xlim(*lims)
        ax.set_xticks(ticks)
        ax.set_xticklabels([f'{t:.1f}' for t in ticks], fontsize=config.SMALL_FONTS)

        if i == 0:
            ax.set_xlabel('Precision', fontsize=config.FONTS)
            ax.set_ylabel('Recall', fontsize=config.FONTS)
        else:
            ax.set_xlabel('Thresholds', fontsize=config.FONTS)
            ax.set_ylabel('F1', fontsize=config.FONTS)

    # iterate different run metrices
    legend_elements, runs_lbl = [], ''
    for i, (name, metrics) in enumerate(data.items()):
        runs_lbl += name

        # unpack data
        prc, rcl, F1 = metrics.precision, metrics.recall, metrics.F1
        trn_prc, trn_rcl, trn_F1 = prc.train, rcl.train, F1.train
        tst_prc, tst_rcl, tst_F1  = prc.test, rcl.test, F1.test
        
        thrs = metrics.index.unique(2).values.astype(float)
        best_thr = np.argsort(tst_F1).iloc[-1]

        # precision recall plotting
        ls = axes[0].plot(trn_prc, trn_rcl, **TRAIN_Ps)
        col = ls[-1].get_color()
        axes[0].plot(tst_prc, tst_rcl, color=col, **TEST_Ps)
        # best treshold based on F1
        axes[0].scatter(tst_prc.iloc[best_thr], tst_rcl.iloc[best_thr], color=col, 
                        s=15, zorder=20)
        
        # F1 plot
        axes[1].plot(thrs, trn_F1, color=col, **TRAIN_Ps)
        axes[1].plot(thrs, tst_F1, color=col, **TEST_Ps)
        axes[1].scatter(thrs[best_thr], tst_F1.iloc[best_thr], color=col, s=15, 
                        zorder=20)

        # legend 
        legend_elements.extend([
            Line2D([0], [0], label=f'Train - {name[12:]}', color=col, **TRAIN_Ps),
            Line2D([0], [0], label=f'Test', color=col, **TEST_Ps),
            ])
    axes[0].legend(handles=legend_elements, bbox_to_anchor=(-.2, 1.05), ncol=2,
                   loc='lower left', fontsize=config.SMALL_FONTS)
    if show:
        plt.show()
    if dest_dir:
        fig.savefig(f'{dest_dir}/prc_rcl_{runs_lbl}.{config.FIGURE_FILETYPE}')

    
    
    
    # if below ran once, viz results
    # if show_results or dest_dir:
    #     ec = self.MCF_edge_cost_thr
    #     eec = self.MCF_entry_exit_cost
    #     mr = self.MCF_miss_rate
    #     vsw = self.MCF_vis_sim_weight
    #     ccm = self.MCF_conf_capping_method
    #     print(f'Current params: ec:{ec} eec:{eec} mr:{mr} vsw:{vsw} ccm:{ccm}')

    #     metrics = pd.read_csv(f'{self.dir}/MCF_param_search.csv', index_col=0)
    #     # print(metrics.iloc[:,[1169, 1029, 813, 914, 780, 483, 316, 252, 32, 1, 0, 18]].T.to_string())
    #     metrics.sort_values(['mota', 'idf1'], axis=1, inplace=True, ascending=False)
    #     prc = metrics.loc['idp']
    #     rcl = metrics.loc['idr']
    #     mota = metrics.loc['mota']
    #     f1 = metrics.loc['idf1']

    #     n = metrics.loc['num_unique_objects'][0]
    #     mostly_tr = metrics.loc['mostly_tracked'][0] / n
    #     part_tr = metrics.loc['partially_tracked'][0] / n
    #     mostly_lost = metrics.loc['mostly_lost'][0] / n

    #     print()
    #     print(metrics.iloc[:,:10].columns)
    #     print(metrics.iloc[:,:10].T.reset_index(drop=True).T)
    #     print()


    #     fig, axes = plt.subplots(1, 2, figsize=config.MEDIUM_FIGSIZE)
    #     fig.subplots_adjust(bottom=.01, left=.1, wspace=.3, right=.95)
    #     ticks = np.arange(0.5, .91, 0.2)
    #     lims = (.3, .94)

    #     # setup both axes
    #     for i, ax in enumerate(axes):
    #         ax.set_ylim(*lims) 
    #         ax.set_yticks(ticks)
    #         ax.set_yticklabels([f'{t:.1f}' for t in ticks], fontsize=config.SMALL_FONTS)

    #         ax.set_xlim(*lims)
    #         ax.set_xticks(ticks)
    #         ax.set_xticklabels([f'{t:.1f}' for t in ticks], fontsize=config.SMALL_FONTS)

    #         ax.set_aspect('equal')
    #         ax.spines['top'].set_visible(False)
    #         ax.spines['right'].set_visible(False)
    #         ax.grid()
    #         if i == 0:
    #             ax.set_xlabel('Identity precision', fontsize=config.FONTS)
    #             ax.set_ylabel('Identity recall', fontsize=config.FONTS)
    #         else:
    #             ax.set_xlabel('MOT accuracy', fontsize=config.FONTS)
    #             ax.set_ylabel('Identity F1', fontsize=config.FONTS)

    #     colors = []
    #     sizes = []
    #     for i in range(len(mota)):
    #         if i == 0:
    #             colors.append(config.DEFAULT_COLORS[0])
    #             sizes.append(250)
    #         else:
    #             colors.append('k')
    #             sizes.append(3)
    #     axes[0].scatter(prc, rcl, alpha=1, edgecolor='k', color=colors, s=sizes)
    #     axes[1].scatter(mota, f1, alpha=1, edgecolor='k', color=colors, s=sizes)


    #     bottom, left = .85, .1
    #     width, height = .85, .1
    #     ax = fig.add_axes([left, bottom, width, height])
    #     ax.set_ylim(.2,.8)
    #     ax.set_xlim(0,1)
    #     ax.set_xticks(np.arange(0,1.01,.1))
    #     ax.tick_params(left=False, labelleft=False, labelsize=config.SMALL_FONTS)

    #     ax.text(mostly_tr/2, .5, 'mostly tracked', clip_on=False, 
    #             ha='center', va='center', fontsize=config.SMALL_FONTS)
    #     ax.text(mostly_tr + part_tr/2, .5, 'partially tracked', clip_on=False, 
    #             ha='center', va='center', fontsize=config.SMALL_FONTS)
    #     ax.text(mostly_tr + part_tr + mostly_lost/2, .5, 'mostly lost', clip_on=False, 
    #             ha='center', va='center', fontsize=config.SMALL_FONTS)
    #     ax.set_title('Proportion of axons', fontsize=config.FONTS)

    #     ax.barh(0.5, mostly_tr, color=config.DEFAULT_COLORS[0], alpha=.8)
    #     ax.barh(0.5, part_tr, left=mostly_tr, color=config.DEFAULT_COLORS[2], alpha=.8)
    #     ax.barh(0.5, mostly_lost, left=mostly_tr+part_tr, color=config.DEFAULT_COLORS[1], alpha=.8)
        
    #     if show_results:
    #         plt.show()
    #     elif dest_dir:
    #         plt.savefig(f'{dest_dir}/MCF_param_search_results.{config.FIGURE_FILETYPE}')
