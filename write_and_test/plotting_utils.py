import proplot as pplt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from matplotlib.widgets import LassoSelector
from matplotlib.path import Path
import matplotlib.patches as mpatches
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.stats import ttest_ind
from scipy.stats import t

'''
File for helping with some plotting functions
'''

bart_plot_colors = {0: 'deep red', 
                    1: 'orange', 
                    2: 'goldenrod',
                    3: 'gray',
                    4: 'pink'}

def set_rc():
    '''Set rc params to be ready for paper polished plots'''
    pplt.rc.reset()
    pplt.rc.update({
        'font.size': 10,
        'font.family': 'Arial',
        'hatch.linewidth': 1,
        'pdf.fonttype': 42 # Used to embed fonts into pdf
    })
    
    
def add_abc_to_subaxes(ax, text='A.', left=0, top=1.05):
    '''
    Add ABC type labels to specific subaxes similar to those
    from proplot
    
    Note that if creating a single ax as in fig, ax = pplt.subplots()
    the ax to pass in is add_abc_to_subaxes(ax[0]) since it works
    with subaxes specifically
    '''
    abc_kw = pplt.rc.fill(
        {
            'size': 'abc.size',
            'weight': 'abc.weight',
            'color': 'abc.color',
            'family': 'font.family',
        },
        context=True
    )
    border_kw = pplt.rc.fill(
        {
            'border': 'abc.border',
            'borderwidth': 'abc.borderwidth',
            'bbox': 'abc.bbox',
            'bboxpad': 'abc.bboxpad',
            'bboxcolor': 'abc.bboxcolor',
            'bboxstyle': 'abc.bboxstyle',
            'bboxalpha': 'abc.bboxalpha',
        },
        context=True,
    )
    kw = {'zorder': 3.5, 'transform': ax.transAxes}
    
    ax.text(left, top, text, **abc_kw, **border_kw, **kw)
    
    
def linear_bestfit(x, y):
    '''
    Given a set of x, y points, calculate linear line of best fit
    return:
        xs, ys: values to plot the line onto a plot with
        r2: R2 score
    '''
    if type(x) == list:
        x = np.array(x)
    if type(y) == list:
        y = np.array(y)
    
    if len(x.shape) == 1:
        x = x.reshape(-1, 1)
    
    lr = LinearRegression().fit(x, y)
    y_pred = lr.predict(x)
    
    x_min, x_max = x.min(), x.max()
    r2 = r2_score(y, y_pred)
    
    xs = np.array([x_min, x_max]).reshape(-1, 1)
    ys = lr.predict(xs)
    
    return xs, ys, r2


def get_ci(data, err_bars=True, ci=0.95):
    '''
    Compute confidence intervals for a 1d data set
    err_bars: if True, return just the error bars for plotting
              if False, return the actual (ci_lower, ci_upper) confidence intervals
    '''
    confidence_level = ci

    mean = np.mean(data)
    std_dev = np.std(data, ddof=1)  # Use ddof=1 for sample standard deviation
    n = len(data)

    sem = std_dev / np.sqrt(n)
    alpha = 1 - confidence_level
    critical_value = t.ppf(1 - alpha / 2, df=n - 1)  # Two-tailed t critical value
    margin_of_error = critical_value * sem

    if err_bars:
        return margin_of_error
    
    ci_lower = mean - margin_of_error
    ci_upper = mean + margin_of_error
    return (ci_lower, ci_upper)
    

'''
Color Functions
'''

# Colors to assign to auxiliary tasks (they will be assigned in order)
colors = pplt.Cycle('default').by_key()['color']
hex_to_rgb = lambda h: tuple(int(h.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
rgb_colors = np.array([hex_to_rgb(color) for color in colors])/255
rgb_colors = [list(color) for color in rgb_colors]

def rgb_to_hex(rgb, scaleup=True):
    if scaleup:
        r, g, b = int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255)
    else:
        r, g, b = int(rgb[0]), int(rgb[1]), int(rgb[2])
    return '#{:02x}{:02x}{:02x}'.format(r, g, b)

hex_colors = [rgb_to_hex(color) for color in rgb_colors]


def get_color_from_colormap(value, a, b, cmap_name='viridis', to_hex=True):
    """
    Function to return a color in RGB or hex from a colormap for a given value in range [a, b].
    
    Parameters:
    - value: float, the input number in the range [a, b].
    - a: float, start of the range.
    - b: float, end of the range.
    - cmap_name: str, the name of the colormap to use (default is 'viridis').
    - to_hex: bool, if True returns color in hex format, else in RGB.
    
    Returns:
    - color: tuple or str, the color in RGB (tuple) or hex (str) format.
    """
    norm_value = (value - a) / (b - a)
    cmap = pplt.Colormap(cmap_name)
    rgb_color = cmap(norm_value)
    if to_hex:
        return pplt.to_hex(rgb_color)
    else:
        return rgb_color


def create_color_map(vmin, vmax, cmap_name='viridis'):
    """
    Creates a ScalarMappable object to use with ax.colorbar() for a defined range.
    
    Parameters:
    - vmin (float): Minimum value for the color map.
    - vmax (float): Maximum value for the color map.
    - cmap_name (str): Name of the matplotlib colormap to use. Default is 'viridis'.
    
    Returns:
    - mappable (ScalarMappable): An object to use with ax.colorbar(mappable).
    """
    cmap = plt.get_cmap(cmap_name)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    
    return mappable

def color_boxplot(boxplot, c, ax):
    for patch in boxplot['boxes']:
        patch = mpatches.PathPatch(patch.get_path(), color=c)
        ax.add_artist(patch)
'''
Significance Utils
Utility functions for adding p-value annotations to plots
'''
def agent_type_significance_test(ys, ax, boxplot=True, bar_dy_ratio=None,
                                 lim_dy_ratio=None, max_stars=3, 
                                 manual_y_start=None, manual_height=None,
                                 ci_plot=False):
    '''
    Perform significance test with boxplot, also automatically find y heights
    and calibrate ylim using boxplot ranges
    
    boxplot: if True, calculate y heights based on boxplot ranges
        otherwise, calculate based on mean heights assuming bar plot
    bar_dy_ratio: how much higher [0,2] bar should be than [0,1], [1,2]
        default 1.5 for boxplot, 2 for bar
    lim_dy_ratio: how much higher ylim should be than top bar
        default 3.5 for boxplot, 4 for bar
        
    manual_y_start, mannual_heights: give fixed values for where significance bars
        should start and how tall they should be
    max_stars how many stars we show up to
    '''
    if manual_y_start is not None and manual_height is not None:
        xidxs = [(0, 1), (1, 2), (0, 2)]
        xvals = [(0, 0.95), (1.05, 2), (0, 2)]
        dy = manual_height
        high_q = manual_y_start
        for j in range(3):
            j1, j2 = xidxs[j]
            x1, x2 = xvals[j]
            if j1 == 0 and j2 == 2:
                high_q = high_q + 1.5*dy
            p = ttest_ind(ys[j1], ys[j2]).pvalue
            significance_bars(x1, x2, p, high_q, ax, dy=dy, max_stars=max_stars)
            
    elif ci_plot:
        xidxs = [(0, 1), (1, 2), (0, 2)]
        xvals = [(0, 0.95), (1.05, 2), (0, 2)]

        cis = [get_ci(y) for y in ys]
        y_means = [np.mean(y) for y in ys]
        y_highs = [y_means[i] + cis[i] for i in range(3)]
        y_lows = [y_means[i] - cis[i] for i in range(3)]
        y_range = max(y_highs) - min(y_lows)
        
        sig_start = max(y_highs) + 0.1*y_range
        sig_size = 0.1*y_range
        
        for j in range(3):
            j1, j2 = xidxs[j]
            x1, x2 = xvals[j]
            if j1 == 0 and j2 == 2:
                start = sig_start + 2*sig_size
            else:
                start = sig_start
            p = ttest_ind(ys[j1], ys[j2]).pvalue
            significance_bars(x1, x2, p, start, ax, dy=sig_size, max_stars=max_stars)
        ax.format(ylim=[min(y_lows)-2*sig_size, sig_start+6*sig_size])
        
    
    elif boxplot:
        # Find boxplot ends
        high_qs = [np.percentile(ys[i], 75) + \
        1.5 * (np.percentile(ys[i], 75) - np.percentile(ys[i], 25)) \
        for i in range(3)]
        high_qs = [min(high_qs[i], max(ys[i])) for i in range(3)]
        high_q = max(high_qs)
        low_qs = [np.percentile(ys[i], 25) - \
        1.5 * (np.percentile(ys[i], 75) - np.percentile(ys[i], 25)) \
        for i in range(3)]
        low_qs = [max(low_qs[i], min(ys[i])) for i in range(3)]
        low_q = min(low_qs)
        dy = high_q / 20
        
        xidxs = [(0, 1), (1, 2), (0, 2)]
        xvals = [(0, 0.95), (1.05, 2), (0, 2)]
        for j in range(3):
            j1, j2 = xidxs[j]
            x1, x2 = xvals[j]
            if j1 == 0 and j2 == 2:
                if bar_dy_ratio is None:
                    high_q = high_q + 1.5*dy
                else:
                    high_q = high_q + bar_dy_ratio*dy
            p = ttest_ind(ys[j1], ys[j2]).pvalue
            significance_bars(x1, x2, p, high_q, ax, dy=dy, max_stars=max_stars)
        
        highlim = high_q+3.5*dy if lim_dy_ratio is None else high_q+lim_dy_ratio*dy
        ax.format(ylim=[low_q-dy, highlim])
    
    else:
        # barplot
        y = max([np.mean(y) for y in ys])
        dy = y / 15
        xidxs = [(0, 1), (1, 2), (0, 2)]
        xvals = [(0, 0.95), (1.05, 2), (0, 2)]
        for j in range(3):
            j1, j2 = xidxs[j]
            x1, x2 = xvals[j]
            if j1 == 0 and j2 == 2:
                if bar_dy_ratio is None:
                    y = y + 2*dy
                else:
                    y = y + bar_dy_ratio*dy
            p = ttest_ind(ys[j1], ys[j2]).pvalue
            significance_bars(x1, x2, p, y, ax, dy=dy, max_stars=max_stars)

        highlim = y+3.5*dy if lim_dy_ratio is None else y+lim_dy_ratio*dy
        ax.format(ylim=[0, highlim])
        

def color_boxplot(boxplot, c, ax):
    for patch in boxplot['boxes']:
        patch = mpatches.PathPatch(patch.get_path(), color=c)
        ax.add_artist(patch)

def significance_bars(x1, x2, pvalue, y, ax, dy=0.05, max_stars=3):
    stars = ''
    if pvalue < 0.05:
        stars = '*'
    if pvalue < 0.005:
        stars = '**'
    if pvalue < 0.0005:
        stars = '***'

    stars = stars[:max_stars]

    if len(stars) > 0:
        ax.plot([x1, x1, x2, x2], [y+dy, y+2*dy, y+2*dy, y+dy], c='k')
        ax.text((x1+x2)/2, y+2.2*dy, stars, va='center', ha='center')

def barplot_annotate_brackets(num1, num2, data, center, height, 
                              yerr=None, dh=.05, barh=.05, fs=None, 
                              maxasterix=3, ax=None):
    """ 
    Annotate barplot with p-values.

    :param num1: number of left bar to put bracket over
    :param num2: number of right bar to put bracket over
    :param data: string to write or number for generating asterixes
    :param center: centers of all bars (like plt.bar() input)
    :param height: heights of all bars (like plt.bar() input)
    :param yerr: yerrs of all bars (like plt.bar() input)
    :param dh: height offset over bar / bar + yerr in axes coordinates (0 to 1)
    :param barh: bar height in axes coordinates (0 to 1)
    :param fs: font size
    :param maxasterix: maximum number of asterixes to write (for very small p-values)
    """

    if type(data) is str:
        text = data
    else:
        # * is p < 0.05
        # ** is p < 0.005
        # *** is p < 0.0005
        # etc.
        text = ''
        p = .05

        while data < p:
            text += '*'
            p /= 10.

            if maxasterix and len(text) == maxasterix:
                break

        if len(text) == 0:
            text = 'n. s.'

    lx, ly = center[num1], height[num1]
    rx, ry = center[num2], height[num2]

    if yerr:
        ly += yerr[num1]
        ry += yerr[num2]

    ax_y0, ax_y1 = plt.gca().get_ylim()
    dh *= (ax_y1 - ax_y0)
    barh *= (ax_y1 - ax_y0)

    y = max(ly, ry) + dh

    barx = [lx, lx, rx, rx]
    bary = [y, y+barh, y+barh, y]
    mid = ((lx+rx)/2, y+barh)

    if ax is None:    
        plt.plot(barx, bary, c='black')
    else:
        ax.plot(barx, bary, c='black')

    kwargs = dict(ha='center', va='bottom')
    if fs is not None:
        kwargs['fontsize'] = fs

    if ax is None:
        plt.text(*mid, text, **kwargs)
    else:
        ax.text(*mid, text, **kwargs)
        

def add_p_star(x, y, pvalue, ax, red=False):
    stars = ''
    if pvalue < 0.05:
        stars = '*'
    if pvalue < 0.005:
        stars = '**'
    if pvalue < 0.0005:
        stars = '***'
    
    color = 'red' if red else 'black'
    ax.text(x, y, stars, ha='center', color=color) 
    

'''
Miscellaneous functions
'''

def create_lasso_selector_plot(data):
    '''
    Given 2D data of shape (N, 2), create a plot where lasso can select chunks of points
    and return which indexes belonged to each cluster
    
    returns:
        cluster_idxs (list of indices 1 per cluster), selector (object)

    Note in Jupyter notebook, must run 
        %matplotlib tk
    first to make the plot in a pop out
    '''
    xmin, ymin = data.min(axis=0)
    xmax, ymax = data.max(axis=0)
    
    subplot_kw = dict(xlim=(xmin, xmax), ylim=(ymin, ymax), autoscale_on=False)
    fig, ax = plt.subplots(subplot_kw=subplot_kw)

    pts = ax.scatter(data[:, 0], data[:, 1], s=80)
    selector = SelectFromCollection(ax, pts)

    def accept(event):
        if event.key == "enter":
            # print("Selected points:")
            # print(selector.xys[selector.ind])

            selector.onenter(selector.ind)
            ax.set_title("")
            fig.canvas.draw()

    fig.canvas.mpl_connect("key_press_event", accept)
    ax.set_title("Press enter to accept selected points.")
    plt.show()
    
    return selector.cluster_idxs, selector


class SelectFromCollection:
    """
    Select indices from a matplotlib collection using `LassoSelector`.

    Selected indices are saved in the `ind` attribute. This tool fades out the
    points that are not part of the selection (i.e., reduces their alpha
    values). If your collection has alpha < 1, this tool will permanently
    alter the alpha values.

    Note that this tool selects collection objects based on their *origins*
    (i.e., `offsets`).

    Parameters
    ----------
    ax : `~matplotlib.axes.Axes`
        Axes to interact with.
    collection : `matplotlib.collections.Collection` subclass
        Collection you want to select from.
    alpha_other : 0 <= float <= 1
        To highlight a selection, this tool sets all selected points to an
        alpha value of 1 and non-selected points to *alpha_other*.
    """

    def __init__(self, ax, collection, alpha_on=0.1, alpha_off=0.01):
        self.canvas = ax.figure.canvas
        self.collection = collection
        self.alpha_on = alpha_on
        self.alpha_off = alpha_off

        self.xys = collection.get_offsets()
        self.Npts = len(self.xys)
        
        self.colored_clusters = 1
        
        self.cluster_idxs = []

        # Ensure that we have separate colors for each object
        self.fc = collection.get_facecolors()
        if len(self.fc) == 0:
            raise ValueError('Collection must have a facecolor')
        elif len(self.fc) == 1:
            self.fc = np.tile(self.fc, (self.Npts, 1))

        self.lasso = LassoSelector(ax, onselect=self.onselect)
        self.ind = []

    def onselect(self, verts):
        path = Path(verts)
        self.ind = np.nonzero(path.contains_points(self.xys))[0]
        self.fc[:, -1] = self.alpha_off
        self.fc[self.ind, -1] = self.alpha_on
        self.collection.set_facecolors(self.fc)
        self.canvas.draw_idle()

    def onenter(self, idxs):
        self.cluster_idxs.append(idxs)
        new_color = np.concatenate([rgb_colors[self.colored_clusters], [self.alpha_on]])
        self.fc[idxs] = new_color
        self.fc[:, -1] = self.alpha_on
        self.colored_clusters += 1
        self.collection.set_facecolors(self.fc)
        self.canvas.draw_idle()

        
    def disconnect(self):
        self.lasso.disconnect_events()
        self.fc[:, -1] = 1
        self.collection.set_facecolors(self.fc)
        self.canvas.draw_idle()


