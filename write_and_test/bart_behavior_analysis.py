import matplotlib.pyplot as plt
import proplot as pplt
pplt.rc.update({'font.size': 10})

color_to_idx = {"red": 0, "yellow": 1, "orange": 2,
                        "gray": 4, "purple": 4}
idx_to_color = {0: "red", 1: "yellow", 2: "orange",
                            3: "gray", 4: "purple"}
bart_plot_colors = {0: 'deep red', 
                    1: 'orange', 
                    2: 'goldenrod'}

def plot_bart_behaviors(df, metrics=['size', 'popped']):
    colors = [0, 1, 2]
    bart_metrics = {
        'size': {'df_metric': 'bart/size', 'label': 'Inflation size'},
        'popped': {'df_metric': 'bart/popped', 'label': 'Popped rate'}
    }

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
                        c=bart_plot_colors[j])
    ax.format(leftlabels=leftlabels, ylabel='')