import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import config
import pprint
import os

from scipy.stats import ttest_ind
from scipy.stats import t
from scipy.stats import mannwhitneyu
from scipy.stats import kruskal

def draw_plot(groups, col, ylabel, levels, factor, dest_dir, show, lbl):
    fig, ax = plt.subplots(figsize=(len(groups)*2, 4))
    fig.subplots_adjust(left=.2, right=.95)
    fig.suptitle(lbl, y=.96)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    xlocs = [i for i in range(len(groups)) for sample in range(len(groups[i]))]
    y = [smpl for g in groups for smpl in g]
    y_medians = [np.median(g) for g in groups]
    ax.scatter(xlocs, y, alpha=.7, s=40, color=col)
    ax.scatter(np.unique(xlocs), y_medians, alpha=.7, s=200, color='k', marker='_', linewidth=4)

    
    xticks = np.unique(xlocs)
    ax.set_xticks(xticks)
    ax.set_xlim(xticks[0]-1, xticks[-1]+1)
    ax.set_xticklabels(levels)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0,20)
    if show:
        plt.show()
    elif dest_dir:
        fname = f'{ylabel}_{factor}'
        plt.savefig(f'{dest_dir}/{fname}.{config.FIGURE_FILETYPE}')
    plt.close()

def test_all(data, dest_dir=None, show=True, subset=None):
    factors = ['design', 'timelapse'] + list(config.DESIGN_FEATURE_NAMES)
    factor_levels = {factor: list(data.index.unique(factor).dropna()) 
                    for factor in factors}
    pprint.pprint(factor_levels)

    for metric in data.columns:  #['metric']:
        if metric == 'metric':
            ylabel = 'directionality_ratio'
            col = 'blue'
        elif metric == 'pos_metric':
            ylabel = 'n_reached_target'
            col = 'green'
        elif metric == 'neg_metric':
            ylabel = 'n_reached_neighbour'
            col = 'red'

        print(config.SPACER)
        print(ylabel)
        print(config.SPACER)
        for factor, levels in factor_levels.items():
            factor_data = data.copy()
            if subset and factor in subset:
                factor_data = data.loc[(slice(None), subset[factor]), :]
            
            groups = []
            for level in levels:
                group = factor_data[factor_data.index.get_level_values(factor) == level]
                group = group[metric]
                groups.append(group)

            if len(groups) == 2:
                t_stat, p = mannwhitneyu(*groups)
                which_test = 'Mann-Whitnay U'
            else:
                t_stat, p = kruskal(*groups)
                which_test = 'Krskal-Wallis'
            
            sig = '***' if p <.05 else 'N.S.'
            lbl = f'{factor}: {levels}:\n {which_test} t={t_stat:.3f}, p={p:.3f}\n --> {sig}\n\n'
            print(lbl)

            draw_plot(groups, col, ylabel, levels, factor, dest_dir, show, lbl)


if __name__ == '__main__':
    show = False
    path = '/home/loaloa/ETZ_drive/biohybrid-signal-p/PDMS_structure_screen_v2/all_results_v4/results.csv'
    dest_dir = '/home/loaloa/ETZ_drive/biohybrid-signal-p/PDMS_structure_screen_v2/all_results_v5/significance/'
    subset = None
    
    
    dest_dir = '/home/loaloa/ETZ_drive/biohybrid-signal-p/PDMS_structure_screen_v2/all_results_v5/significance_designsubset/'
    subset = {
        'channel width': (2,3,4),
        'n 2-joints': (9,10,11,12,5,8,6,7),
        'n rescue loops': (2,9,10,11,12),
        '2-joint placement': (5,6,7,8),
        'rescue loop design': (12, 17),
        'use spiky tracks': (12, 20),
        'final lane design': (1, 12,18,19),
        '2-joint design': (12,13,14,15,16),
    }

    os.makedirs(dest_dir, exist_ok=True)

    data = pd.read_csv(path, index_col=list(range(11)))
    test_all(data, dest_dir=dest_dir, show=show, subset=subset)