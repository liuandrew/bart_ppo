import proplot as pplt
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.widgets import LassoSelector
from matplotlib.path import Path
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

'''
File for helping with some plotting functions
'''


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



from scipy.stats import ttest_ind

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


