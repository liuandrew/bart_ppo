import matplotlib.pyplot as plt
import proplot as pplt
from typing import Union
import itertools
import numpy as np
import pandas as pd

pplt.rc.update({'font.size': 10})

color_to_idx = {"red": 0, "orange": 1, "yellow": 2,
                        "gray": 4, "purple": 4}
idx_to_color = {0: "red", 1: "orange", 2: "yellow",
                            3: "gray", 4: "purple"}
bart_plot_colors = {0: 'deep red', 
                    1: 'orange', 
                    2: 'goldenrod'}

def plot_bart_behaviors(df, metrics=['size', 'popped'], ax=None,
                        alpha=1.0):
    colors = [0, 1, 2]
    bart_metrics = {
        'size': {'df_metric': 'bart/size', 'label': 'Inflation size'},
        'popped': {'df_metric': 'bart/popped', 'label': 'Popped rate'},
        'rt': {'df_metric': 'bart/inflate_delay', 'label': 'Reaction time'}
    }

    if ax is None:
        fig, ax = pplt.subplots(nrows=len(metrics), ncols=len(colors), 
                                figwidth=6)

    metric_df = df.set_index(['metric', 'step'])
    color_df = metric_df.loc['bart/color']
    leftlabels = []
    for i, metric in enumerate(metrics):
        df_metric = bart_metrics[metric]['df_metric']
        leftlabels.append(bart_metrics[metric]['label'])
        for j, color in enumerate(colors):
            plot_metric = metric_df.loc[df_metric][color_df['value'] == color]
            ax[i, j].plot(plot_metric.index, 
                        plot_metric.value.ewm(alpha=0.01).mean(),
                        c=bart_plot_colors[j], alpha=alpha)
    ax.format(leftlabels=leftlabels, ylabel='')
    return ax

    
def get_bart_res_eps(
    res, 
    color : Union[list[str], list[int]] = None, 
    min_rt : int = None,
    popped : bool = None,
    ):
    '''
    Select episodes from a res based on the episode results or
    setting
    
    color: color of balloon
    min_rt: minimum reaction_time
    popped: whether balloon popped
    '''
    num_eps = len(res['obs'])
    idxs = np.arange(num_eps)
    
    collected_filters = []
    
    if color is not None:
        color_filter = np.full(idxs.shape, False)
        ep_colors = np.array(res['data']['color'])
        for c in color:
            if type(c) == str:
                c = color_to_idx[c]
        color_filter = (ep_colors == c) | (color_filter)
        collected_filters.append(color_filter)
    
    if min_rt is not None:
        collected_filters.append(np.array(res['data']['inflate_delay']) >= min_rt)
    
    if popped is not None:
        collected_filters.append(np.array(res['data']['popped']) == popped)
    
    collected_filter = np.all(collected_filters, axis=0)
    idxs = idxs[collected_filter]
    
    filtered_res = {}
    for k, v in res.items():
        if k != 'data':
            filtered_res[k] = list(itertools.compress(v, collected_filter))
    
    filtered_res['data'] = {}
    for k, v in res['data'].items():
        filtered_res['data'][k] = list(itertools.compress(v, collected_filter))
    
    return filtered_res

    
def plot_3color_it_rt(res, metrics=['size', 'rt'], ax=None, ep_num=None):
    '''
    Plot main characterics of a single evaluation of bart trials
    Available metrics to pass in for plotting:
        'size'/'rt'/'popped'
    '''
    colors = np.array(res['data']['color'])
    end_size = np.array(res['data']['end_size'])
    popped = np.array(res['data']['popped'])
    reaction_times = np.array(res['data']['inflate_delay'])

    metric_to_label = {
        'size': 'Inflation Times',
        'rt': 'Reaction Times',
        'popped': 'Popped Count'
    }

    if ax is None:
        fig, ax = pplt.subplots(nrows=len(metrics), ncols=3, 
                                figwidth=6, sharex=False)
        ax.format(leftlabels=[metric_to_label[metric] for metric in metrics])
    for i, metric in enumerate(metrics):
        for j in range(3):
            if metric == 'size':
                ax[i, j].hist(end_size[colors == j], c=bart_plot_colors[j])
            elif metric == 'rt':
                ax[i, j].hist(reaction_times[colors == j], c=bart_plot_colors[j])
            elif metric == 'popped':
                p = np.array(popped).sum()
                not_p = (~popped).sum()
                ax[i, j].bar([0, 1], [p, not_p], width=0.5, c=bart_plot_colors[j])
                ax[i, :].format(xformatter=['Popped', 'Not Popped'], xlocator=range(2))
                
    return ax


def plot_3color_meta_ep(res, metrics=['size', 'rt', 'popped'], ax=None, ep_num=0):
    '''
    Plot main characterics of a single evaluation of bart trials
    Available metrics to pass in for plotting:
        'size'/'rt'/'popped'

    ep_num: which episode to plot from a set of meta bart trials
    '''
    colors = np.array(res['data']['current_color'][ep_num])
    end_size = np.array(res['data']['last_size'][ep_num])
    popped = np.array(res['data']['popped'][ep_num])
    reaction_times = np.array(res['data']['inflate_delay'][ep_num])

    metric_to_label = {
        'size': 'Inflation Times',
        'rt': 'Reaction Times',
        'popped': 'Popped Count'
    }

    if ax is None:
        fig, ax = pplt.subplots(nrows=len(metrics), ncols=3, 
                                figwidth=6, sharex=False)
        ax.format(leftlabels=[metric_to_label[metric] for metric in metrics])
    
    for i, metric in enumerate(metrics):
        max_height = 0
        
        for j in range(3):
            if metric == 'size':
                ax[i, :].format(xlim=[0, 1])
                counts, _, _ = ax[i, j].hist(end_size[colors == j], c=bart_plot_colors[j], bins=40,
                                             range=[0, 1])
                max_height = max(max_height, np.max(counts))
            elif metric == 'rt':
                ax[i, j].hist(reaction_times[colors == j], c=bart_plot_colors[j])
            elif metric == 'popped':
                p = popped[colors == j].sum()
                not_p = (~popped[colors == j]).sum()
                ax[i, j].bar([0, 1], [p, not_p], width=0.5, c=bart_plot_colors[j])
                ax[i, :].format(xformatter=['Popped', 'Not Popped'], xlocator=range(2))
        if metric == 'size':
            for j in range(3):
                balloon_mean = res['data']['balloon_means'][ep_num][j]
                # print([balloon_mean, balloon_mean])
                # print([0, max_height])
                ax[i, j].plot([balloon_mean, balloon_mean], [0, max_height])
    return ax


def plot_meta_it_progression(res, metrics=['size', 'rt', 'popped'], ax=None, ep_num=0):
    '''
    Plot main characterics of a single evaluation of bart trials
    Available metrics to pass in for plotting:
        'size'/'rt'/'popped'

    ep_num: which episode to plot from a set of meta bart trials
    '''
    colors = np.array(res['data']['current_color'][ep_num])
    end_size = np.array(res['data']['last_size'][ep_num])
    popped = np.array(res['data']['popped'][ep_num])
    reaction_times = np.array(res['data']['inflate_delay'][ep_num])

    metric_to_label = {
        'size': 'Inflation Times',
        'rt': 'Reaction Times',
        'popped': 'Popped Count'
    }

    if ax is None:
        fig, ax = pplt.subplots(nrows=len(metrics), ncols=3, 
                                figwidth=6, sharex=False)
        ax.format(leftlabels=[metric_to_label[metric] for metric in metrics])
    
    for i, metric in enumerate(metrics):
        for j in range(3):
            if metric == 'size':
                ax[i, :].format(ylim=[0, 1])
                its = end_size[colors == j]
                smoothed_its = pd.Series(its).ewm(alpha=0.1).mean()
                ax[i, j].scatter(range(len(its)), its, c=bart_plot_colors[j], alpha=0.2)
                ax[i, j].plot(range(len(its)), smoothed_its, c=bart_plot_colors[j], linewidth=2)
                balloon_mean = res['data']['balloon_means'][ep_num][j]
                ax[i, j].plot([0, len(its)], [balloon_mean, balloon_mean])
            elif metric == 'rt':
                rts = reaction_times[colors == j]
                ax[i, j].scatter(range(len(rts)), rts, c=bart_plot_colors[j])
            elif metric == 'popped':
                p = popped[colors==j]*1
                ax[i, j].scatter(range(len(p)), p, c=bart_plot_colors[j])
                ax[i, :].format(yformatter=['Not Popped', 'Popped'], ylocator=range(2))
                


def get_meta_mean_diffs(res):
    num_eps = len(res['data']['current_color'])
    all_diffs = []

    for ep_num in range(num_eps):
        ep_diffs = []
        
        colors = np.array(res['data']['current_color'][ep_num])
        end_size = np.array(res['data']['last_size'][ep_num])
        popped = np.array(res['data']['popped'][ep_num])
        
        unpopped_its = end_size[~popped]
        unpopped_colors = colors[~popped]
        
        balloon_means = res['data']['balloon_means'][ep_num]
        
        for j in range(3):
            it_mean = unpopped_its[unpopped_colors == j].mean()
            mean_diff = abs(balloon_means[j] - it_mean)
            ep_diffs.append(mean_diff)
        all_diffs.append(ep_diffs)
    
    return all_diffs



def get_meta_fixed_mean_diff(res):
    num_eps = len(res['data']['current_color'])
    all_means = []

    for ep_num in range(num_eps):
        ep_means = []
        
        colors = np.array(res['data']['current_color'][ep_num])
        end_size = np.array(res['data']['last_size'][ep_num])
        popped = np.array(res['data']['popped'][ep_num])
        
        unpopped_its = end_size[~popped]
        unpopped_colors = colors[~popped]
        
        balloon_means = res['data']['balloon_means'][ep_num]
        
        for j in range(3):
            it_mean = unpopped_its[unpopped_colors == j].mean()
            ep_means.append(it_mean)
        all_means.append(ep_means)
    
    all_means = np.array(all_means)
    mean_diff = np.abs(all_means - all_means.mean(axis=0))
    
    return mean_diff, all_means
