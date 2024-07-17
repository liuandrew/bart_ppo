import matplotlib.pyplot as plt
import proplot as pplt
from typing import Union
import itertools
import numpy as np

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

    
def plot_3color_it_rt(res, metrics=['size', 'rt'], ax=None):
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
                p = np.array(res['data']['popped']).sum()
                not_p = (~np.array(res['data']['popped'])).sum()
                ax[i, j].bar([0, 1], [p, not_p], width=0.5, c=bart_plot_colors[j])
                ax[i, :].format(xformatter=['Popped', 'Not Popped'], xlocator=range(2))
                
    return ax