import matplotlib.pyplot as plt
import proplot as pplt
from typing import Union
import itertools
import numpy as np
import pandas as pd
from plotting_utils import rgb_colors
# Might want to move this to a separate utils file
from bart_representation_analysis import linear_best_fit

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
                

def plot_ncolor_meta_ep(res, metrics=['size', 'rt', 'popped'], ax=None, ep_num=0,
                        num_colors=1):
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

    if num_colors == 1:
        color_it = [1]
    else:
        color_it = range(num_colors)

    if ax is None:
        if num_colors == 1:
            fig, axs = pplt.subplots(nrows=1, ncols=len(metrics), 
                                     figwidth=6, sharey=False)
            axs.format(toplabels=[metric_to_label[metric] for metric in metrics])
        else: 
            fig, axs = pplt.subplots(nrows=len(metrics), ncols=num_colors, 
                                    figwidth=6, sharex=False)
            axs.format(leftlabels=[metric_to_label[metric] for metric in metrics])
    
    for i, metric in enumerate(metrics):
        max_height = 0
        
        for n, j in enumerate(color_it):
            
            ax = axs[i+n*num_colors]
            if metric == 'size':
                ax.format(xlim=[0, 1])
                counts, _, _ = ax.hist(end_size[colors == j], c=bart_plot_colors[j], bins=40,
                                             range=[0, 1])
                max_height = max(max_height, np.max(counts))
            elif metric == 'rt':
                ax.hist(reaction_times[colors == j], c=bart_plot_colors[j])
            elif metric == 'popped':
                p = popped[colors == j].sum()
                not_p = (~popped[colors == j]).sum()
                ax.bar([0, 1], [p, not_p], width=0.5, c=bart_plot_colors[j])
                ax.format(xformatter=['Popped', 'Not Popped'], xlocator=range(2))
        if metric == 'size':
            for n, j in enumerate(color_it):
                ax = axs[i+n*num_colors]
                balloon_mean = res['data']['balloon_means'][ep_num][j]
                ax.plot([balloon_mean, balloon_mean], [0, max_height])
    return axs



def plot_ncolor_meta_progression(res, metrics=['size', 'rt', 'popped'],
                                 ax=None, ep_num=0, num_colors=3, include_non_pops=True):
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

    if num_colors == 1:
        color_it = [1]
    else:
        color_it = range(num_colors)

    if ax is None:
        if num_colors == 1:
            fig, axs = pplt.subplots(nrows=1, ncols=len(metrics), 
                                     figwidth=6, sharey=False)
            axs.format(toplabels=[metric_to_label[metric] for metric in metrics])
        else: 
            fig, axs = pplt.subplots(nrows=len(metrics), ncols=num_colors, 
                                    figwidth=6, sharex=False)
            axs.format(leftlabels=[metric_to_label[metric] for metric in metrics])
    
    for i, metric in enumerate(metrics):
        for n, j in enumerate(color_it):
            ax = axs[i+n*num_colors]
            
            if metric == 'size':
                ax.format(ylim=[0, 1])
                its = end_size[colors == j]
                p = popped[colors == j]
                smoothed_its = pd.Series(its).ewm(alpha=0.1).mean()
                ax.scatter(range(len(its)), its, c=bart_plot_colors[j], alpha=0.2)
                ax.plot(range(len(its)), smoothed_its, c=bart_plot_colors[j], linewidth=2)
                balloon_mean = res['data']['balloon_means'][ep_num][j]
                ax.plot([0, len(its)], [balloon_mean, balloon_mean])

                if include_non_pops:
                    smoothed_its = pd.Series(its[~p]).ewm(alpha=0.1).mean()
                    ax.plot(np.argwhere(~p).flatten(), smoothed_its, c=rgb_colors[0])
            elif metric == 'rt':
                rts = reaction_times[colors == j]
                ax.scatter(range(len(rts)), rts, c=bart_plot_colors[j])
            elif metric == 'popped':
                p = popped[colors==j]*1
                ax.scatter(range(len(p)), p, c=bart_plot_colors[j])
                ax.format(yformatter=['Not Popped', 'Popped'], ylocator=range(2))


def plot_1color5fsize(res):
    '''
    Specific plot for seeing the inflation time meta progression
    over fixed trial of 5 sizes and their pop rates
    '''
    fig, ax = pplt.subplots(nrows=2, ncols=5, 
                            sharey=True, sharex=False,
                            figwidth=7)
    for ep_num in range(5):
        
        colors = np.array(res['data']['current_color'][ep_num])
        end_size = np.array(res['data']['last_size'][ep_num])
        popped = np.array(res['data']['popped'][ep_num])
        # reaction_times = np.array(res['data']['inflate_delay'][ep_num])
    
        its = end_size[colors == 1]
        p = popped[colors == 1]
        smoothed_its = pd.Series(its).ewm(alpha=0.1).mean()
        non_pop_its = pd.Series(its[~p]).ewm(alpha=0.1).mean().tolist()
        non_pop_x = np.argwhere(~p).flatten().tolist()
        balloon_mean = res['data']['balloon_means'][ep_num][1]
        # print(non_pop_x, non_pop_its)

        ax[0, ep_num].scatter(range(len(smoothed_its)), its, c=bart_plot_colors[1], alpha=0.2)
        ax[0, ep_num].plot(range(len(smoothed_its)), smoothed_its, c=bart_plot_colors[1], linewidth=2)
        ax[0, ep_num].plot(non_pop_x, non_pop_its, c=rgb_colors[0])
        ax[0, ep_num].plot([0, len(smoothed_its)], [balloon_mean, balloon_mean])

        ax[1, ep_num].barh([(p*1).sum(), (~p*1).sum()], c=bart_plot_colors[1])

    ax[0, :].format(ylim=[0, 1])
    ax[1, :].format(ylocator=range(2), yformatter=['Popped', 'Not Popped'])
    ax.format(leftlabels=['Inflation Times', 'Pop Counts'],
              toplabels=[rf'$\mu$={mu}' for mu in [0.2, 0.4, 0.6, 0.8, 1.0]])

    

def plot_1colornfsize(res, eps=None, diff_colors=False, plot_pops=True,
                      plot_both_means=True):
    '''
    Specific plot for seeing the inflation time meta progression
    over fixed trial of n sizes and their pop rates
    eps: can past a list to plot specific episodes
    diff_colors: whether to use different colors for lines and scatters
    '''
    if eps is None:
        num_eps = len(res['data']['current_color'])
        rows = int(np.ceil(num_eps / 5))
        eps = range(num_eps)
        ncols = 5
    else:
        if len(eps) < 5:
            ncols = len(eps)
            rows = 1
        else:
            ncols = 5
            rows = int(np.ceil(len(eps) / 5))

    fig, ax = pplt.subplots(nrows=rows, ncols=ncols, 
                            sharey=True, sharex=True,
                            figwidth=7)
    
    if plot_pops:
        pax = ax.panel_axes('t', width='3em', space=0)
        pax.format(yticks=[], xticks=[])
    
    num_steps = len(res['data']['current_color'][0])
    bar_width = num_steps // 3

    mus = []
    for i, ep in enumerate(eps):
        colors = np.array(res['data']['current_color'][ep])
        end_size = np.array(res['data']['last_size'][ep])
        popped = np.array(res['data']['popped'][ep])
        # reaction_times = np.array(res['data']['inflate_delay'][ep])
    
        its = end_size[colors == 1]
        p = popped[colors == 1]
        smoothed_its = pd.Series(its).ewm(alpha=0.1).mean()
        non_pop_its = pd.Series(its[~p]).ewm(alpha=0.1).mean().tolist()
        non_pop_x = np.argwhere(~p).flatten().tolist()
        balloon_mean = res['data']['balloon_means'][ep][1]
        mus.append(balloon_mean)
        # print(non_pop_x, non_pop_its)

        if diff_colors:
            ax[i].scatter(range(len(smoothed_its)), its, c=rgb_colors[i], alpha=0.2)
            ax[i].plot(range(len(smoothed_its)), smoothed_its, c=rgb_colors[i], linewidth=2)
            if plot_both_means:
                ax[i].plot(non_pop_x, non_pop_its, c=rgb_colors[i], ls='--')
            ax[i].plot([0, len(smoothed_its)], [balloon_mean, balloon_mean])
        else:
            ax[i].scatter(range(len(smoothed_its)), its, c=bart_plot_colors[1], alpha=0.2)
            ax[i].plot(range(len(smoothed_its)), smoothed_its, c=bart_plot_colors[1], linewidth=2)
            if plot_both_means:
                ax[i].plot(non_pop_x, non_pop_its, c=rgb_colors[0])
            ax[i].plot([0, len(smoothed_its)], [balloon_mean, balloon_mean])

        if plot_pops:
            pax[i].bar([bar_width], [(~p*1).sum()], c='kelly green', width=bar_width)
            pax[i].bar([bar_width*2], [(p*1).sum()], c='tomato', width=bar_width)

    ax.format(ylim=[0, 1], xlim=[0, num_steps])
    if plot_pops:
        pax.format(yticks=[])
    for a, mu in zip(ax, mus):
        a.format(title=rf'$\mu$={mu:.2f}')

    return fig, ax


def get_meta_mean_diffs(res, colors_used=3):
    num_eps = len(res['data']['current_color'])
    all_diffs = []

    if colors_used <= 1:
        color_it = [1]
    else:
        color_it = range(colors_used)
    for ep_num in range(num_eps):
        ep_diffs = []
        
        colors = np.array(res['data']['current_color'][ep_num])
        end_size = np.array(res['data']['last_size'][ep_num])
        popped = np.array(res['data']['popped'][ep_num])
        
        unpopped_its = end_size[~popped]
        unpopped_colors = colors[~popped]
        
        balloon_means = res['data']['balloon_means'][ep_num]
        
        for j in color_it:
            it_mean = unpopped_its[unpopped_colors == j].mean()
            mean_diff = abs(balloon_means[j] - it_mean)
            ep_diffs.append(mean_diff)
        all_diffs.append(ep_diffs)
    
    return all_diffs

    


def get_meta_self_mean_diff(res, colors_used=3):
    num_eps = len(res['data']['current_color'])
    all_means = []

    if colors_used <= 1:
        color_it = [1]
    else:
        color_it = range(colors_used)

    for ep_num in range(num_eps):
        ep_means = []
        
        colors = np.array(res['data']['current_color'][ep_num])
        end_size = np.array(res['data']['last_size'][ep_num])
        popped = np.array(res['data']['popped'][ep_num])
        
        unpopped_its = end_size[~popped]
        unpopped_colors = colors[~popped]
        
        for j in color_it:
            it_mean = unpopped_its[unpopped_colors == j].mean()
            ep_means.append(it_mean)
        all_means.append(ep_means)
    
    all_means = np.array(all_means)
    mean_diff = np.abs(all_means - all_means.mean(axis=0))
    
    return mean_diff, all_means


def get_pop_rates(res, colors_used=3):
    num_eps = len(res['data']['current_color'])
    all_means = []

    if colors_used <= 1:
        color_it = [1]
    else:
        color_it = range(colors_used)

    for ep_num in range(num_eps):
        ep_means = []
        
        colors = np.array(res['data']['current_color'][ep_num])
        end_size = np.array(res['data']['last_size'][ep_num])
        popped = np.array(res['data']['popped'][ep_num])
        
        for j in color_it:
            it_mean = popped[colors == j].mean()
            ep_means.append(it_mean)
        all_means.append(ep_means)
    
    return all_means

'''
Behavior analysis and classification
'''
def pop_behavior(pop_rates, ret_plot=True):
    '''
    Get the popping behavior of the agent
    * Determine if there is a peak in the popping
    * Remove the peak if it exists
    * Find the pop rate slope from start to around when the agent reaches
      no pop rate
      
    Slope and peak determine behavior
    
    ret_plot: return more info
        False: return slope, local_peak
        True: return x, y, local_peak
    '''
    mus = np.arange(0.2, 1.01, 0.05)
    p = pop_rates
    local_peak = False
    for i in range(3, 13):
        p1, p2, p3, p4, p5 = p[i-2], p[i-1], p[i], p[i+1], p[i+2]
        # if (p3 > 1.05 * p2) and (p3 > 1.05 * p4) and \
        if ((p3 > 0.1+p2) and (p3 > 0.02+p4) and (p4 > 0.02+p5)) or \
            ((p3 > 0.02+p2) and (p3 > 0.1+p4) and (p2 > 0.02+p1)) or \
            ((p3 > 0.02+p2) and (p3 > 0.02+p4) and (p4 > 0.02+p5) and (p2 > 0.02+p1)):
            local_peak = True
            
        if ((p3 > 0.02+p2) and (p3 > 0.02+p4) and (p4 > 0.02+p5) and (p2 > 0.02+p1)):
            removed_peak = np.concatenate([p[:i-2], p[i+2:]])
            removed_mus = np.concatenate([mus[:i-2], mus[i+2:]])
        elif ((p3 > 0.1+p2) and (p3 > 0.02+p4) and (p4 > 0.02+p5)):
            removed_peak = np.concatenate([p[:i-1], p[i+2:]])
            removed_mus = np.concatenate([mus[:i-1], mus[i+2:]])
        elif ((p3 > 0.02+p2) and (p3 > 0.1+p4) and (p2 > 0.02+p1)):
            removed_peak = np.concatenate([p[:i-2], p[i+1:]])
            removed_mus = np.concatenate([mus[:i-2], mus[i+1:]])
        else:
            removed_peak = p
            removed_mus = mus

        if local_peak:
            break

    first_nonpop = np.argmax(removed_peak[3:] < 0.03) + 4
    removed_peak = removed_peak[:first_nonpop]
    removed_mus = removed_mus[:first_nonpop]
    (m, b), r2 = linear_best_fit(removed_mus, removed_peak)
    
    if ret_plot:
        return removed_mus, removed_mus*m+b, local_peak
    else:
        return m, local_peak