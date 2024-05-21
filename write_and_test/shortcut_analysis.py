import numpy as np
from evaluation import *
from model_evaluation import *
from plotting_utils import *
from umap import UMAP
from tqdm import tqdm
import proplot as pplt
import itertools
from collections import defaultdict
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from pathlib import Path
import pandas as pd
import scipy
from representation_analysis import draw_character

turn_speed = 0.3
move_speed = 10

shortcut_target = np.array([150, 250])
path_target = np.array([25, 250])
goal_target = np.array([275, 275])




'''
================================================================
Clustering measures
================================================================
Various measures of clustering
Primarily point cloud Euclidean Wasserstein and silhouette scores
'''

def two_cloud_wasserstein(X, Y, normalize=True, all_data=None, min_max=None):
    '''
    Compute the Wasserstein distance between 2 point clouds
    by finding which pairs of points minimizes distances between the two
    
    X, Y: [N, features] arrays of cloud data
    normalize: if True, normalize wasserstein based on the maximum L2 distance possible
        between both sets
    all_data: [N, features] optionally pass this to calculate maximum L2 distance rather than
        X and Y, for example if X and Y are a subset
        e.g., two_cloud_wasserstein(activ2d[color==0], activ2d[color==1], all_data=activ2d)
    min_max: [2, features] optionally instead of letting the function calculate L2 normalization, pass
        mins and maxes precalculated
        e.g., min_max = np.vstack([activ.min(axis=0), activ.max(axis=0)])
    
    returns:
        mean_dist,
        x_ind, y_ind: the indices of X and Y that are used to minimize distances
    '''
    
    distance_matrix = distance.cdist(X, Y, 'euclidean')
    x_ind, y_ind = linear_sum_assignment(distance_matrix)
    dists = distance_matrix[x_ind, y_ind]
    
    if len(x_ind) == 0:
        return 0.0, x_ind, y_ind, dists
    
    wasserstein_dist = dists.sum()
    mean_dist = wasserstein_dist / len(x_ind)
    
    
    if all_data is not None:
        min_max = np.vstack([all_data.min(axis=0), all_data.max(axis=0)])
    elif normalize:
        all_data = np.vstack([X, Y])
        min_max = np.vstack([all_data.min(axis=0), all_data.max(axis=0)])
    
    if min_max is not None:
        L2 = np.linalg.norm(min_max[1, :] - min_max[0, :])
        mean_dist = mean_dist / L2
        dists = dists / L2
    
    return mean_dist, x_ind, y_ind, dists



def pairwise_silhouette_score(a, labels):
    '''Compute pairwise silhouette score'''
    silscores = np.zeros((5, 5))
    # silscores = []
    for i, j in itertools.product(range(5), range(5)):
        if j > i and (labels == i).sum() > 0 and (labels == j).sum() > 0:
            silscores[i, j] = two_set_silhouette_score(a[labels == i], a[labels == j])
            # silscores.append(two_set_silhouette_score(a[labels == i], a[labels == j]))
            
    if len(silscores) > 0:
        # return np.mean(silscores)
        return silscores
    else:
        return 0
    
    
def one_vs_all_silhouette_score(a, labels):
    silscores = np.zeros(5)
    for i in range(5):
        silscores[i] = two_set_silhouette_score(a[labels == i], a[labels != i])
    return silscores

    
def two_set_silhouette_score(x1, x2):
    '''Find the silhouette score between two point clouds of data'''
    X = np.vstack([x1, x2])
    labels = [0]*len(x1) + [1]*len(x2)
    score = silhouette_score(X, labels)    
    return score


def shortcut_decomposed_silhouettes(a, labels):
    '''Take clusters from decompose_shortcut_trajectories and calculate some 
    cluster distances'''
    silscores = []

    # Green vs orange - heading towards entrance vs shortcut
    i, j = 1, 2
    if (labels == i).sum() > 0 and (labels == j).sum() > 0:
        silscores.append(two_set_silhouette_score(a[labels == i], a[labels == j]))
    else:
        silscores.append(0)
    
    # Red vs purple - corridor from shortcut vs entrance
    i, j = 3, 4
    if (labels == i).sum() > 0 and (labels == j).sum() > 0:
        silscores.append(two_set_silhouette_score(a[labels == i], a[labels == j]))
    else:
        silscores.append(0)

    # start vs end - before corridor vs after corridor
    idxs1 = np.isin(labels, [0, 1, 2])
    idxs2 = np.isin(labels, [3, 4])
    if idxs1.sum() > 0 and idxs2.sum() > 0:
        silscores.append(two_set_silhouette_score(a[idxs1], a[idxs2]))
    else:
        silscores.append(0)
        
    return silscores


'''
================================================================
Plotting functions
================================================================
'''

def colored_activ2d_plot(activ2d, labels, pos=None, ax=None,
                         split_activ=False, format_ax=True,
                         s=5, angle=None, colors=None, alpha=0.1):
    '''
    Flexible plotting of 2d projected activations and coloring
    
    activ2d: 2d projection of population activity
    labels: array of labels for each point
        -1: unlabeled color
    pos: pass array of positions if wanting to plot these
    split_activ: if True, split each label into its on subaxes
    format_ax: whether to handle formatting as standard
    s: plot point size
    
    angle: pass 1D array of angles if wanting to plot these
        If passed, will use triangles for drawing position plots    
    colors: pass in collection of [r, g, b] values from [0, 1] to color with
    '''
    num_classes = (np.unique(labels) != -1).sum()
    num_cols = 1
    if split_activ:
        num_cols += num_classes
    if pos is not None:
        num_cols += num_classes
    if colors is None:
        colors = rgb_colors
    
    if ax is None:
        fig, ax = pplt.subplots(ncols=num_cols, sharey=False)
    
    if format_ax:
        if split_activ:
            xmin, ymin = activ2d.min(axis=0)-1
            xmax, ymax = activ2d.max(axis=0)+1
            ax[:num_classes+1].format(xlim=[xmin, xmax], ylim=[ymin, ymax])
        
        if pos is not None:
            if split_activ:
                ax[num_classes+1:].format(xlim=[0,300], ylim=[0,300])
            else:
                ax[1:].format(xlim=[0,300], ylim=[0,300])
                
            
    if (labels == -1).sum() > 0:
        idxs = labels == -1
        ax[0].scatter(activ2d[idxs, 0], activ2d[idxs, 1], alpha=0.01, color='gray', s=s)
    for i in range(num_classes):
        idxs = labels == i
        ax[0].scatter(activ2d[idxs, 0], activ2d[idxs, 1], alpha=alpha, color=colors[i+1], s=s)
        if split_activ:
            ax[i+1].scatter(activ2d[idxs, 0], activ2d[idxs, 1], alpha=alpha, color=colors[i+1], s=s)
        if pos is not None:
            pax = ax[i+1+num_classes] if split_activ else ax[i+1]
            
            if angle is not None:
                draw_shortcut_maze_light(ax=pax)
                draw_char_positions(pos[idxs], angle[idxs], pax, color=colors[i+1])
            else:
                pax.scatter(pos[idxs, 0], pos[idxs, 1], alpha=alpha, color=colors[i+1], s=s)
                

def plot_two_cloud_wasserstein(X, Y, normalize=True, all_data=None, min_max=None, one_ax=True,
                               colors=[None, None]):
    '''
    Given two point clouds, plot the distances used to move points close together
    and visualize how the two cloud wasserstein is computed
    Note this will make most sense on 2D data, but if higher dimensional data is
    given we'll calculate the wasserstein using all dimensions, but only pot the first two
    
    normalize, all_data, min_max: normalization options, the same as from
        two_cloud_wasserstein()
    one_ax: if True plot onto one axis, if False plot onto 3
    colors: optionally pass list of 2 colors to color the points with                  
    '''
    
    dist, x_inds, y_inds, dists = two_cloud_wasserstein(X, Y, normalize, all_data, min_max)

    if one_ax:
        fig, ax = pplt.subplots()
        ax.scatter(X[:, 0], X[:, 1], alpha=0.3, color=colors[0])
        ax.scatter(Y[:, 0], Y[:, 1], alpha=0.3, color=colors[1])
        
    else:
        fig, ax = pplt.subplots(ncols=3)
        ax[0].scatter(X[:, 0], X[:, 1], alpha=0.3, color=colors[0])
        ax[1].scatter(Y[:, 0], Y[:, 1], alpha=0.3, color=colors[1])

    for i in range(len(x_inds)):
        x_ind = x_inds[i]
        y_ind = y_inds[i]
        if one_ax:
            ax.plot([X[x_ind, 0], Y[y_ind, 0]], 
                    [X[x_ind, 1], Y[y_ind, 1]], color='gray', alpha=0.3)      
        else:
            ax[2].plot([X[x_ind, 0], Y[y_ind, 0]], 
                    [X[x_ind, 1], Y[y_ind, 1]], color='gray', alpha=0.3)
    
    if all_data is None:
        all_data = np.vstack([X, Y])
    xmin = all_data[:,0].min()
    ymin = all_data[:,1].min()
    xmax = all_data[:,0].max()
    ymax = all_data[:,1].max()
    
    ax.format(xlim=[xmin, xmax], ylim=[ymin, ymax],
             title=['', '', f'{dist:.3f}'])
    return ax, dist




def draw_shortcut_maze(shortcut_open=True, ax=None):
    '''Draw the walls for a shortcut maze on the given axis'''
    if ax is None:
        fig, ax = pplt.subplots()
    
    shortcut_probability = 1 if shortcut_open else 0
    env = gym.make('ShortcutNav-v0', shortcut_probability=shortcut_probability, 
                   render_character=False,
                   wall_colors=1.5)
    env.render('human', ax=ax)
    ax.format(xlim=[0, 300], ylim=[0, 300])
    


def draw_shortcut_maze_light(shortcut_open=True, ax=None, wall_thickness=5, draw_goal=False):
    '''Draw the shortcut maze manually on a light background'''
    if ax is None:
        fig, ax = pplt.subplots()
        ax.format(xlim=[0, 300], ylim=[0, 300])
        
    rect = plt.Rectangle([50, 250-wall_thickness/2], 75, wall_thickness, fc=(0.2, 0.2, 0.2))
    ax.add_patch(rect)
    rect = plt.Rectangle([175, 250-wall_thickness/2], 125, wall_thickness, fc=(0.2, 0.2, 0.2))
    ax.add_patch(rect)
    
    if not shortcut_open:
        rect = plt.Rectangle([125, 250-wall_thickness/2], 50, wall_thickness, fc=(0.9, 0.4, 0.9))
        ax.add_patch(rect)
    if draw_goal:
        rect = plt.Rectangle([262.5, 262.5], 25, 25, fc=(0.2, 0.7, 0.2))
        ax.add_patch(rect)
    
    
def draw_char_positions(pos, angle, ax=None, size=10, color=rgb_colors[0], alpha=0.2):
    '''Draw character positions, used in colored_activ2d_plot'''
    if ax is None:
        fig, ax = pplt.subplots()
        ax.format(xlim=[0, 300], ylim=[0, 300])
    
    for i in range(len(pos)):
        draw_character(pos[i], angle[i], size, ax, color, alpha)

'''
================================================================
Miscellanious helper functions
================================================================
'''

def get_ep_lens(ep_pos):
    ep_lens = [len(ep) for ep in ep_pos]
    return ep_lens


def ep_split_res(target, ep_lens):
    data = []
    cur_idx = 0
    for l in ep_lens:
        data.append(target[cur_idx:cur_idx+l])
        cur_idx += l
    return data

def ep_stack_activations(res, combine=False, half=False):
    '''Stack activations dictionaries from episodic evaluation calls
    
    combine: whether to combine all activations so there are no episodic separations
    half: use half precision to save space
    returns: list of dictionaries
        Ex. activs[ep]['shared_activations'][layer_num]
    '''    
    activs = []
    for ep in range(len(res['activations'])):
        activs.append(stack_activations(res['activations'][ep], half=half))
        
    if combine:
        stacked_activs = {}
        for key in activs[0]:
            stacked_activs[key] = torch.hstack(
                [activs[ep][key] for ep in range(len(activs))])
        activs = stacked_activs
    return activs


def stack_activations(activation_dict, also_ret_list=False, half=False):
    '''
    Activations passed back from a FlexBase forward() call can be appended, e.g.
    all_activations = []
    for ...:
        all_activations.append(actor_critic.act(..., with_activations=True)['activations'])
        
    This will result in a list of dictionaries
    
    This function converts all_activations constructed in this way into a dictionary,
    where each value of the dictionary is a tensor of shape
    [layer_num, seq_index, activation_size]
    
    Args:
        also_ret_list: If True, will also return activations in a list one-by-one
            rather than dict form. Good for matching up with labels and classifier ordering
            from train classifiers function
        half: use half precision to save space
    '''
    stacked_activations = defaultdict(list)
    list_activations = []
    keys = activation_dict[0].keys()
    
    for i in range(len(activation_dict)):
        for key in keys:
            num_layers = len(activation_dict[i][key])
            
            if num_layers > 0:
                # activation: 1 x num_layers x activation_size
                activation = torch.vstack(activation_dict[i][key]).reshape(1, num_layers, -1)
                # stacked_activations: list (1 x num_layers x activation_size)
                stacked_activations[key].append(activation)
    
    for key in stacked_activations:
        activations = torch.vstack(stacked_activations[key]) # seq_len x num_layers x activation_size
        activations = activations.transpose(0, 1) # num_layers x seq_len x activation_size
        if half:
            stacked_activations[key] = activations.half()
        else:
            stacked_activations[key] = activations
        
        #Generate activations in list form
        if also_ret_list:
            for i in range(activations.shape[0]):
                list_activations.append(activations[i])
    
    if also_ret_list:
        return stacked_activations, list_activations
    else:
        return stacked_activations
    
    
def activs_list_to_dict(activs):
    '''Convert a list of activs (1 dict per episode) into
    a stacked dict'''
    stack_activs = {}
    for activ in activs:
        for activ_type in activ:
            if activ_type not in stack_activs:
                stack_activs[activ_type] = []
            stack_activs[activ_type].append(activ[activ_type])
    
    for activ_type in stack_activs:
        stack_activs[activ_type] = torch.hstack(stack_activs[activ_type])
    return stack_activs


def comb_policy_res(shortcut_res, chk, vis_n=5, first_n_eps=50,
                   dist_idx=1, get_activ2d=True, get_activ=False,
                   activ_type='shared_activations', activ_layer=1):
    '''
    Combine shortcut_res['ns'] and shortcut_res['ws'] positions and 
    activations
    
    vis_n: how many steps after seeing shortcut to consider
    first_n_eps: how many eps were considered from each res type
        (activ2d were computed using 50 so this should generally be fixed)
    dist_idx: which min_dist umap to use (idx 0-3)
        dists = [0.15, 0.25, 0.5, 0.75]
    ret_all: whether to return vis, shortcut_avail, shortcut_used
    
    activ_type, activ_layer: used if get_activ is set to True
    '''
    res1 = shortcut_res['ns'][chk]
    res2 = shortcut_res['ws'][chk]
    pos1 = res1['ep_pos'][:first_n_eps]
    pos2 = res2['ep_pos'][:first_n_eps]
    pos = pos1+pos2

    vis = res1['vis'][:first_n_eps] + res2['vis'][:first_n_eps]
    shortcut = res1['shortcut'][:first_n_eps] + res2['shortcut'][:first_n_eps]
    shortcut_used = [[check_shortcut_usage(p)] for p in pos1] + [[check_shortcut_usage(p)] for p in pos2]
    color = vision_coloring(vis, shortcut, num_steps=vis_n)
    
    res = {
        'color': color,
        'ep_pos': pos,
        'vis': vis,
        'shortcut_avail': shortcut,
        'shortcut_used': shortcut_used
    }
    
    if get_activ2d:
        activ2d = shortcut_res['activ2d'][chk][dist_idx]
        res['activ2d'] = activ2d
        
    if get_activ:
        activs1 = activs_list_to_dict(res1['activs'][:first_n_eps])
        activs2 = activs_list_to_dict(res2['activs'][:first_n_eps])
        activ1 = activs1[activ_type][activ_layer]
        activ2 = activs2[activ_type][activ_layer]
        res['activ'] = torch.vstack([activ1, activ2])
        
    return res

'''
================================================================
Learning curve helper methods
================================================================
Methods for combining learning curve data from continued experiments
and for finding the first points that performance crosses threshold
to line up learning curves
'''

def combine_cont_df(exp_format='shortcut_wc1.5p{p}_t{t}',
                    cont_format='shortcut_wc1.5p{p}_cont_t{t}',
                    formatter={}, subdir='shortcut_wc2', concat=True):
    '''
    Combine a run dataframe with a continuation if it exists
    Otherwise simply return the original df
    '''
    
    runs = Path('../runs')/subdir
    folder_names = [f.name for f in runs.iterdir()]
    run_names = [f.name.split('__')[0] for f in runs.iterdir()]
    
    exp_name = exp_format.format(**formatter)
    cont_name = cont_format.format(**formatter)

    exp_idx = run_names.index(exp_name)
    exp_file = runs/folder_names[exp_idx]/'tflog.csv'
    df = pd.read_csv(exp_file)

    if cont_name in run_names:
        cont_idx = run_names.index(cont_name)
        cont_file = runs/folder_names[cont_idx]/'tflog.csv'
        df2 = pd.read_csv(cont_file)
        if not concat:
            return df, df2
        df = pd.concat([df, df2], ignore_index=True)
    return df
        
    
def get_run_df_metric(df, metric='length', alpha=0.01, ignore_first=100):
    '''
    From a run df loaded with combine_cont_df or just pd.read_csv,
    perform the usual average_runs method of getting out a vector
    of x and y to plot or analyze
    '''
    shortcut_to_key = {
        'value_loss': 'losses/value_loss',
        'policy_loss': 'losses/policy_loss',
        'aux_loss': 'losses/auxiliary_loss',
        'return': 'charts/episodic_return',
        'length': 'charts/episodic_length'
    }
    
    if metric in shortcut_to_key:
        metric = shortcut_to_key[metric]
    
    df = df[df['metric'] == metric]
    ewm = df['value'].ewm(alpha=alpha).mean()
    inter = scipy.interpolate.interp1d(df['step'], ewm)
    min_x, max_x = df.iloc[0]['step'], df.iloc[-1]['step']
    x = np.arange(min_x, max_x, 200)
    y = inter(x)
    
    x = x[ignore_first:]
    y = y[ignore_first:]
    return x, y


def get_first_shortcut_performance(t, lens, below_y=180, batch=64, ret_chk=True):
    '''Find when the agent first shows "signs of life", as in average
    escape time drops below below_y. We find that using an 
    exponentially weighted running mean with alpha=0.01 and below_y=180
    gives pretty consistent results
    
    
    return:
        ret_chk: the nearest multiple-of-10 checkpoint where this occurs
        else: the actual first index
    '''
    below = np.argwhere(lens < below_y)
    if len(below) == 0:
        return False
    first = below.squeeze()[0]
    
    first_t = t[first]
    first_chk = first_t / (batch*100)
    first_chk = int(round(first_chk, -1)) #round to nearest 10
    
    if ret_chk:
        return first_chk
    else:
        return first
    
    
    
chks1 = np.arange(0, 150, 10) #0-1e6
chks2 = np.arange(160, 300, 20) #1e6-2e6
chks3 = np.arange(300, 600, 40) #2e6-4e6
chks4 = np.arange(620, 930, 60) #4e6-6e6
chk_sched = np.concatenate([chks1, chks2, chks3, chks4])

def get_first_last_performing_chks(exp_format='plumtosc_sharedn_plumsched1_task{task}p{p}_t{t}',
                                   cont_format='plumtosc_sharedn_plumsched1_task{task}p{p}_cont_t{t}',
                                   formatter={'p': 0.4, 'task': 1.7, 't': 0}, verbose=False, subdir='plumtosc',
                                   follow_chk_sched=True):
    '''Specific function for getting first and last checkpoints to test
    starting from first performing
    
    If return False, it means either not enough chks available or 
        never reached the required performance to start
    '''    
    # Get first checkpoint where performance is sufficient
    df = combine_cont_df(exp_format=exp_format,
                         cont_format=cont_format,
                         formatter=formatter, subdir=subdir)
    x, y = get_run_df_metric(df, ignore_first=1000)
    first_chk = get_first_shortcut_performance(x, y)
    
    exp_name = exp_format.format(**formatter)
    
    folder = Path(f'../saved_checkpoints/{subdir}/{exp_name}')
    fnames = [f.name for f in folder.iterdir()]
    
    # Get the last chk available in saved_checkpoints folder in chk_sched
    chks = first_chk + chk_sched
    last_chk_sched_idx = -1
    for i, chk in enumerate(chks):
        if f'{chk}.pt' not in fnames:
            last_chk_sched_idx = i-1
            break
    
    if verbose:
        print(f'{p}_{t}', first_chk)
    
    if first_chk is False:
        return False
    else:
        return first_chk, last_chk_sched_idx
    





'''
================================================================
Point coloring methods
================================================================
Methods for assigning points into predefined clusters based on
where the agent was and the trajectory it ended up taking
'''

def decision_coloring(ep_vis, ep_shortcut, num_steps=10):
    '''
    Generate labels for points in trajectory num_steps after the shortcut is seen
    and can be based on whether shortcut is used in the ep or available in the ep
    '''
    colorings = np.full(len(np.concatenate(ep_vis)), -1)
    i = 0
    for ep in range(len(ep_vis)):
        if ep_shortcut[ep][0]:
            label = 1 #shortcut available
        else:
            label = 0 #shortcut not available
        first = np.argmax(ep_vis[ep])
        
        colorings[i:i+num_steps] = label
        i += len(ep_vis[ep])
    return colorings


def vision_decision_coloring(ep_vis, ep_shortcut_avail, ep_shortcut_used, end=10, start=0,
                             skip_late_sights=False):
    '''
    Generate labels for points in trajectory num_steps after the shortcut is seen, split into
    labels of
    0: no shortcut available
    1: shortcut available and not taken
    2: shortcut available and taken
    
    start: how many steps after seeing the shortcut to start considering
    end: how many steps after seeing the shortcut to stop considering
    skip_late_sights: if first sighting occurs above corridor, 
    '''
    colorings = np.full(len(np.concatenate(ep_vis)), -1)
    i = 0
    for ep in range(len(ep_vis)):
        if not ep_shortcut_avail[ep][0]:
            label = 0 #shortcut not available
        else:
            if ep_shortcut_used[ep][0]:
                label = 2 #shortcut available and used
            else:
                label = 1 #shortcut available but not used
        first = np.argmax(ep_vis[ep])
        if i+start+first < 0:
            colorings[:i+end+first] = label
        else:
            colorings[i+start+first:i+end+first] = label
        i += len(ep_vis[ep])
    return colorings



def comparative_vision_decision_coloring(ep_vis, ep_shortcut_used, used=True, end=10, start=0):
    '''
    Very specific coloring: since our policy was tested on 50 eps of closed, 50 open,
        use this to find the episodes where shortcut was closed, but on the corresponding
        open episode the shortcut was used (or seen and not used)
    E.g., if shortcut_used on ep 62, color episode 12 as True (the corresponding closed
        shortcut episode).
    used: if True, give episodes where shortcut was used. If False, give episodes
        where shortcut was seen but not used
    '''
    colorings = np.full(len(np.concatenate(ep_vis)), -1)
    
    # Create a boolean array for corresponding closed shortcut episodes
    eps = np.concatenate(ep_shortcut_used)[50:]
    if not used:
        eps = ~eps
    eps = np.concatenate([eps, np.full((50,), False)])    
    
    i = 0
    for ep in range(len(ep_vis)):
        if eps[ep]:
            label = 1 #shortcut not available
            first = np.argmax(ep_vis[ep])        
            colorings[i+start:i+end] = label
        i += len(ep_vis[ep])
    return colorings


def decompose_shortcut_trajectories(res=None, ep_pos=None, ep_shortcut_used=None, ret_labels=True,
                                   skip_start=True, top_right_only=False):
    '''
    Decompose trajectories in shortcut environment into "optimistic" clusters based on
    where the agent was at each time step
    
    skip_start: don't make the first 5 steps a separate cluster, giving us a total of 4 unique clusters
    top_right_only: if True, only consider right half of corridor. This makes red points better
        intersect with purple ones
    
    Can either give res dictionary (which should 'ep_pos', 'pos', and 'shortcut_used')
    or directly give an ep_pos and ep_shortcut (where shortcut is used in each ep) list
    '''
    if res is not None:
        ep_pos = res['ep_pos']
        ep_lens = get_ep_lens(ep_pos)
        pos = res['pos']
        ep_shortcut_used = np.array([e[0] for e in ep_split_res(res['shortcut_used'], ep_lens)])
        num_steps = len(pos)
    elif ep_pos is not None and ep_shortcut_used is not None:
        num_steps = len(np.vstack(ep_pos))
    else:
        raise Exception('Either res or ep_pos+ep_shortcut_used must be given')
        
    if not skip_start:
        # First 5 start points
        ep_idxs = [np.arange(len(e)) for e in ep_pos]
        ep_idxs = [e < 5 for e in ep_idxs]
        start = np.concatenate(ep_idxs)
    
    if skip_start:
        first = 0
    else:
        first = 5
        
    # In case ep_shortcut_used is a list of lists e.g., [[True], [False]]
    #  let's convert it to just a list e.g., [True, False]
    if type(ep_shortcut_used[0]) == list:
        ep_shortcut_used = np.array(ep_shortcut_used).squeeze()
        
    # Leading to entrance - orange
    ep_idxs = []
    for ep in range(len(ep_pos)):
        p = ep_pos[ep]
        idxs = np.full(ep_pos[ep].shape[0], False)

        first_above = np.argwhere(p[:, 1] > 250)
        if len(first_above) > 0 and not ep_shortcut_used[ep]:
            idxs[first:first_above[0, 0]] = True
        ep_idxs.append(idxs)
    before_entrance = np.concatenate(ep_idxs)
        
    # Leading to shortcut - green
    ep_idxs = []
    for ep in range(len(ep_pos)):
        p = ep_pos[ep]
        idxs = np.full(ep_pos[ep].shape[0], False)

        first_above = np.argwhere(p[:, 1] > 250)
        if len(first_above) > 0 and ep_shortcut_used[ep]:
            idxs[first:first_above[0, 0]] = True
        ep_idxs.append(idxs)
    before_shortcut = np.concatenate(ep_idxs)
    
    # After entrance - red
    ep_idxs = []
    for ep in range(len(ep_pos)):
        p = ep_pos[ep]
        idxs = np.full(ep_pos[ep].shape[0], False)

        if top_right_only:
            first_above = np.argwhere((p[:, 1] > 250) & (p[:, 0] > 150))
        else:
            first_above = np.argwhere(p[:, 1] > 250)
            
        if len(first_above) > 0 and not ep_shortcut_used[ep]:
            idxs[first_above[0, 0]:] = True
        ep_idxs.append(idxs)
    after_entrance = np.concatenate(ep_idxs)
        
    # After shortcut - purple
    ep_idxs = []
    for ep in range(len(ep_pos)):
        p = ep_pos[ep]
        idxs = np.full(ep_pos[ep].shape[0], False)

        if top_right_only:
            first_above = np.argwhere((p[:, 1] > 250) & (p[:, 0] > 150))
        else:
            first_above = np.argwhere(p[:, 1] > 250)
        if len(first_above) > 0 and ep_shortcut_used[ep]:
            idxs[first_above[0, 0]:] = True
        ep_idxs.append(idxs)
    after_shortcut = np.concatenate(ep_idxs)
    
    if skip_start:
        no_cluster = ~(before_entrance | before_shortcut | after_entrance | after_shortcut)
    else:
        no_cluster = ~(start | before_entrance | before_shortcut | after_entrance | after_shortcut)
    
    if ret_labels:
        labels = np.full(num_steps, -1)
        
        if skip_start:
            first = 0
        else:
            labels[start] = 0
            first = 1
        labels[before_entrance] = 0 + first
        labels[before_shortcut] = 1 + first
        labels[after_entrance] = 2 + first
        labels[after_shortcut] = 3 + first
        return labels, no_cluster
    else:
        return [start, before_entrance, before_shortcut, after_entrance, after_shortcut], no_cluster

def closed_seen_taken_coloring(ep_pos, ep_shortcut_avail, ep_shortcut_used, ignore_incomplete=True):
    '''
    Color trajectory points based on whether shortcut was closed/seen/taken,
    and only consider points that are below the corridor
    ignore_incomplete: Ignore episodes where the goal was not reached
    '''
    colorings = np.full(len(np.concatenate(ep_pos)), -1)
    i = 0

    # In case ep_shortcut_used is a list of lists e.g., [[True], [False]]
    #  let's convert it to just a list e.g., [True, False]
    if type(ep_shortcut_used[0]) == list:
        ep_shortcut_used = np.array(ep_shortcut_used).squeeze()
    if type(ep_shortcut_avail[0]) == list:
        ep_shortcut_avail = np.array(ep_shortcut_avail).squeeze()

    for ep in range(len(ep_pos)):
        p = ep_pos[ep]
        first_above = np.argwhere(p[:, 1] > 250)
        if len(first_above) > 0:
            first_above = first_above[0, 0]
        else:
            first_above = len(p) - 1
            
        if not ep_shortcut_avail[ep]:
            label = 0 #shortcut not available
        else:
            if ep_shortcut_used[ep]:
                label = 2 #shortcut available and used
            else:
                label = 1 #shortcut available but not used
        
        if ignore_incomplete and len(p) > 201:
            label = -1 # goal not reached in time

        colorings[i:i+first_above] = label
            
        i += len(p)

    return colorings

def closed_seen_taken_corridor(ep_pos, ep_shortcut_avail, ep_shortcut_used, top_right_only=True,
                               ignore_incomplete=True):
    '''
    Color trajectory points based on whether shortcut was closed/seen/taken,
    and only consider points that are above the corridor
    ignore_incomplete: Ignore episodes where the goal was not reached
    top_right_only: if True, only consider right half of corridor. This makes red points better
        intersect with purple ones
    '''
    colorings = np.full(len(np.concatenate(ep_pos)), -1)
    i = 0
    
    # In case ep_shortcut_used is a list of lists e.g., [[True], [False]]
    #  let's convert it to just a list e.g., [True, False]
    if type(ep_shortcut_used[0]) == list:
        ep_shortcut_used = np.array(ep_shortcut_used).squeeze()
    if type(ep_shortcut_avail[0]) == list:
        ep_shortcut_avail = np.array(ep_shortcut_avail).squeeze()

    for ep in range(len(ep_pos)):
        p = ep_pos[ep]
        
        if top_right_only:
            first_above = np.argwhere((p[:, 1] > 250) & (p[:, 0] > 150))
        else:
            first_above = np.argwhere(p[:, 1] > 250)

        if len(first_above) > 0:
            first_above = first_above[0, 0]
        else:
            first_above = len(p) - 1
            
        if not ep_shortcut_avail[ep]:
            label = 0 #shortcut not available
        else:
            if ep_shortcut_used[ep]:
                label = 2 #shortcut available and used
            else:
                label = 1 #shortcut available but not used
        
        if ignore_incomplete and len(p) > 201:
            label = -1 # goal not reached in time

        colorings[i+first_above:i+len(p)] = label
            
        i += len(p)

    return colorings
    

'''
================================================================
Shortcut analysis functions
================================================================
Used for determining performance of an agent in the shortcut environment
Most importantly check_shortcut_usage()
'''

def get_moves_to_target(start_pos, target_pos, start_angle, perfect_angle=False):
    '''
    Get the direction and turns needed to head towards a certain direction
    
    perfect_angle: if True, assume we can reach the perfect angle needed rather
        than requiring harder turns
    '''
    target_vector = target_pos - start_pos
    target_angle = np.arctan2(target_vector[1], target_vector[0])

    # Calculate the angle needed to head in vector direction to target
    #  and how many turns are needed to get there
    # theta = abs(target_angle - angle) % (2 * np.pi)
    # min_angle_dist = 2 * np.pi - theta if theta > np.pi else theta
    # direction = -1 if theta > np.pi else 1
    
    angle_dist1 = target_angle - start_angle
    angle_dist2 = min(abs(2*np.pi+angle_dist1), abs(2*np.pi-angle_dist1))
    sign1 = np.sign(angle_dist1)
    angle_dist1 = abs(angle_dist1)

    if angle_dist1 < angle_dist2:
        direction = sign1
    else:
        direction = -1 * sign1
    min_angle_dist = min(angle_dist1, angle_dist2)
    
    turns = np.round(min_angle_dist / turn_speed)
    
    if perfect_angle:
        resulting_angle = target_angle
    else:
        resulting_angle = (direction * turns * turn_speed + start_angle) % (2*np.pi)
    resulting_vector = np.array([np.cos(resulting_angle), np.sin(resulting_angle)])

    return resulting_angle, resulting_vector, turns


def compute_path_steps(pos, angle, target, ret_pos=False, perfect_angle=False, 
                      ret_steps=False):
    '''Compute a theoretical best bath through the given target
    
    ret_pos: whether to return the intermediate positions
    perfect_angle: whether to assume that we can turn perfectly to the correct angle
    ret_steps: whether to return the individual steps used'''
    
    # Find number of steps to get above y of 250 via shortcut
    angle1, dir1, turns1 = get_moves_to_target(pos, target, angle, perfect_angle)
    steps1 = np.ceil((250 - pos[1]) / (move_speed * dir1[1]))
    steps1 = steps1//2
    pos1 = steps1 * move_speed * dir1 + pos

    angle1b, dir1b, turns1b = get_moves_to_target(pos1, target, angle1, perfect_angle)
    steps1b = np.ceil((250 - pos1[1]) / (move_speed * dir1b[1]))
    pos1b = steps1b * move_speed * dir1b + pos1


    angle2, dir2, turns2 = get_moves_to_target(pos1b, goal_target, angle1b, perfect_angle)
    steps2 = np.ceil((262.5 - pos1b[0]) / (move_speed * dir2[0]))
    steps2 = steps2//2
    pos2 = steps2 * move_speed * dir2 + pos1b

    angle2b, dir2b, turns2b = get_moves_to_target(pos2, goal_target, angle2, perfect_angle)
    steps2b = np.ceil((262.5 - pos2[0]) / (move_speed * dir2b[0]))
    pos2b = steps2b * move_speed * dir2b + pos2

    total_steps = turns1 + turns1b + turns2 + turns2b + steps1 + steps1b + steps2 + steps2b
    
    if ret_pos:
        if ret_steps:
            return total_steps, [pos1, pos1b, pos2, pos2b], [turns1, turns1b, turns2, turns2b, steps1, steps1b, steps2, steps2b]
        return total_steps, [pos1, pos1b, pos2, pos2b]
    elif ret_steps:
        return total_steps, [turns1, turns1b, turns2, turns2b, steps1, steps1b, steps2, steps2b]
    
    return total_steps


def check_shortcut_usage(p, ret_arrays=False, req_finish=True):
    '''
    Check if shortcut was used in a trajectory
    
    req_finish: if True, only count episodes where the goal was reached (assume this is true if
      len(p) < 202)
    '''
    # Check that x values are within range
    x1 = (p[:-1, 0] > 125) & (p[:-1, 0] < 175)
    x2 = (p[1:, 0] > 125) & (p[1:, 0] < 175)

    # Check that y values cross over
    y1 = (p[:-1, 1] < 250)
    y2 = (p[1:, 1] > 250)

    used_shortcut = ((y1 & y2) & (x1 | x2)).any() 
    
    if req_finish:
        fin = len(p) < 202
        used_shortcut = used_shortcut & fin
    
    if ret_arrays:
        return used_shortcut, [x1, x2, y1, y2]
    
    return used_shortcut





'''
================================================================
Evaluation helper functions
================================================================
Some functions that help perform evaluations, although we end up not using these very often

Main function: 
shortcut_test: 50 episodes of shortcut open and 50 closed. Can used fixed trajectory or policy

'''



def shortcut_test(exp_name=None, trial=0, chk=0, plum_pos=-1, with_fa=False,
                  model=None, obs_rms=None):
    
    fa = pickle.load(open('data/shortcut/forced_actions_v2', 'rb'))
    if model is None:
        if exp_name is None:
            raise Exception('need exp_name or model/obs_rms')
        model, obs_rms = load_chk(exp_name, trial=trial, chk=chk, subdir='')
    
    num_episodes=50
    all_res = {
        'ws': [],
        'ns': []
    }
    if with_fa:
        forced_actions = fa['ns']
    else:
        forced_actions = None
    res = forced_action_evaluate(model, obs_rms, env_name='ShortcutNav-v0', 
                           env_kwargs={'character_reset_pos': 3,
                                      'shortcut_probability': 0,
                                      'wall_colors': 1.5,
                                      'plum_pos': plum_pos},
                       seed=1, with_activations=True, data_callback=shortcut_visdata_callback,
                           num_episodes=num_episodes, forced_actions=forced_actions)

    '''Collect up to first+nth step of seeing shortcut
    where first is the first step seeing the shortcut'''
    activs = ep_stack_activations(res, half=True)

    all_res['ns'].append({
        'activs': activs,
        'actions': res['actions'],
        'ep_pos': res['data']['pos'],
        'ep_angle': res['data']['angle'],
        'vis': res['data']['shortcut_vis'],
        'shortcut': res['data']['shortcut']
    })
    if with_fa:
        forced_actions = fa['ws']
    else:
        forced_actions = None

    res = forced_action_evaluate(model, obs_rms, env_name='ShortcutNav-v0', 
                   env_kwargs={'character_reset_pos': 3,
                              'shortcut_probability': 1,
                              'wall_colors': 1.5,
                              'plum_pos': plum_pos},
               seed=1, with_activations=True, data_callback=shortcut_visdata_callback,
                   num_episodes=num_episodes, forced_actions=forced_actions)

    '''Collect up to first+nth step of seeing shortcut
    where first is the first step seeing the shortcut'''
    activs = ep_stack_activations(res, half=True)

    all_res['ws'].append({
        'activs': activs,
        'actions': res['actions'],
        'ep_pos': res['data']['pos'],
        'ep_angle': res['data']['angle'],
        'vis': res['data']['shortcut_vis'],
        'shortcut': res['data']['shortcut']
    })
        
    return all_res




def test_shortcut_use_rate(model, obs_rms, character_reset_pos=3, n_eps=100, env_kwargs={}):
    kw = {'character_reset_pos': character_reset_pos, 'shortcut_probability': 1.}
    envkw = env_kwargs.copy()
    for key, value in kw.items():
        envkw[key] = value
    res1 = evaluate(model, obs_rms, env_kwargs=envkw, env_name='ShortcutNav-v0',
                  num_episodes=n_eps, data_callback=shortcut_data_callback)
    shortcuts_used = np.array([check_shortcut_usage(p) for p in res1['data']['pos']])
    return np.sum(shortcuts_used) / n_eps


def test_shortcut_agent(model, obs_rms, character_reset_pos=0, n_eps=20, env_kwargs={}):
    '''base_env_kwargs: env_kwargs to change for the agent - e.g. wall_colors'''

    kw = {'character_reset_pos': character_reset_pos, 'shortcut_probability': 1.}
    envkw = env_kwargs.copy()
    for key, value in kw.items():
        envkw[key] = value
    res1 = evaluate(model, obs_rms, env_kwargs=envkw, env_name='ShortcutNav-v0',
                  num_episodes=n_eps, data_callback=shortcut_data_callback)

    kw = {'character_reset_pos': character_reset_pos, 'shortcut_probability': 0}
    envkw = env_kwargs.copy()
    for key, value in kw.items():
        envkw[key] = value
    res2 = evaluate(model, obs_rms, env_kwargs=envkw, env_name='ShortcutNav-v0',
                  num_episodes=n_eps, data_callback=shortcut_data_callback)

    # Check how often shortcuts were used when available
    shortcuts_used = np.array([check_shortcut_usage(p) for p in res1['data']['pos']])

    # Check how well agent did compared to an "optimal" path
    shortcut_starts = []
    shortcut_optimal_steps = []
    for i in range(n_eps):
        # First position and angle of the episode
        p = res1['data']['pos'][i][0]
        a = res1['data']['angle'][i][0]
        shortcut_starts.append((p, a))
        shortcut_optimal_steps.append(compute_path_steps(p, a, shortcut_target, perfect_angle=True))

    path_optimal_steps = []
    path_starts = []
    for i in range(n_eps):
        p = res2['data']['pos'][i][0]
        a = res2['data']['angle'][i][0]
        path_starts.append((p, a))
        path_optimal_steps.append(compute_path_steps(p, a, path_target, perfect_angle=True))

    shortcut_actual_steps = [len(r) for r in res1['rewards']]
    path_actual_steps = [len(r) for r in res2['rewards']]
    
    return {
        'shortcuts_used': shortcuts_used,
        'shortcut_use_rate': np.sum(shortcuts_used)/n_eps,
        'shortcut_actual_steps': shortcut_actual_steps,
        'shortcut_theoretical_steps': np.array(shortcut_optimal_steps).squeeze() - 2,
        'shortcut_starts': shortcut_starts,
        'path_actual_steps': path_actual_steps,
        'path_theoretical_steps': np.array(path_optimal_steps).squeeze() - 2,
        'path_starts': path_starts
    }

def shortcut_color_pos(pos):
    rectangles = [ #each rectangle given by x1, x2, y1, y2
        [0, 125, 250, 300], #red
        [125, 300, 250, 300], #green
        [0, 300, 0, 250], #blue

        [0, 100, 150, 250], #purple
        [100, 200, 150, 250], #cyan
    ]

    colors = [
        [0.9, 0., 0., 0.3],
        [0., 0.9, 0., 0.3],
        [0., 0., 0.9, 0.3],

        [0.9, 0., 0.9, 0.3],
        [0., 0.9, 0.9, 0.3]
    ]
    
    rect_idxs = []
    
    for rectangle in rectangles:
        x1, x2, y1, y2 = rectangle
        idxs = (pos[:, 0] >= x1) & (pos[:, 0] <= x2) & (pos[:, 1] >= y1) & (pos[:, 1] <= y2)
        rect_idxs.append(idxs)

    return {
        'rect_idxs': rect_idxs,
        'colors': colors
    }

def test_shortcut_activations(model=None, obs_rms=None, exp_string='shortcutnav_fcp{prob}reset{reset}batch{batch}',
                              prob=0.1, reset=3, batch=64, trial=0, exp_name=None, chk=300,
                              umap_min_dist=0.75, skip_activs=True, skip_umap=True, subdir='shortcut_resets',
                              env_kwargs={}, forced_actions=None, seed=None, test_only=False,
                              shortcut_probability=0.5, data_callback=shortcut_data_callback):
    
    if model is None:
        if exp_name is None:
            exp_name = exp_string.format(prob=prob, reset=reset, batch=batch)
        model, obs_rms = load_chk(exp_name, chk, trial, subdir=subdir)
    
    
    # shortcut_res = test_shortcut_agent(model, obs_rms, reset, env_kwargs=env_kwargs)
    shortcut_use_rate = test_shortcut_use_rate(model, obs_rms, reset, env_kwargs=env_kwargs)
    
    kw = {'character_reset_pos': reset, 'shortcut_probability': shortcut_probability}
    envkw = env_kwargs.copy()
    for key, value in kw.items():
        envkw[key] = value
    
    if forced_actions is None:
        res = evaluate(model, obs_rms, env_name='ShortcutNav-v0', env_kwargs=envkw,
                    data_callback=data_callback, with_activations=True,
                       num_episodes=100, seed=seed)
    elif forced_actions is True:
        fa = pickle.load(open('data/shortcut/forced_actions', 'rb'))
        forced_actions = fa['forced_actions']
        seed = fa['seed']

        res = forced_action_evaluate(model, obs_rms, env_name='ShortcutNav-v0', env_kwargs=envkw,
                    data_callback=data_callback, with_activations=True,
                       num_episodes=100, forced_actions=forced_actions, seed=seed)
    else:
        res = forced_action_evaluate(model, obs_rms, env_name='ShortcutNav-v0', env_kwargs=envkw,
                    data_callback=data_callback, with_activations=True,
                       num_episodes=100, forced_actions=forced_actions, seed=seed)
        
    
    # Process activations
    activs = ep_stack_activations(res, False)
    ep_pos = res['data']['pos']
    pos = np.vstack(ep_pos)
    angle = np.vstack(res['data']['angle']).reshape(-1)
    activ = torch.hstack([a['shared_activations'] for a in activs])[1]
    
    # Process shortcut availability and whether it was used
    ep_shortcut_avail = []
    ep_shortcut_used = []
    
    ep_shortcut_used_single = []
    ep_shortcut_avail_single = []
    for i in range(len(activs)):
        num_steps = activs[i]['shared_activations'].shape[1]
        ep_shortcut_avail.append(np.full((num_steps, 1), res['data']['shortcut'][i][0]))
        shortcut_used = check_shortcut_usage(ep_pos[i])
        ep_shortcut_used.append(np.full((num_steps, 1), shortcut_used))
        
        ep_shortcut_avail_single.append(res['data']['shortcut'][i][0])
        ep_shortcut_used_single.append(shortcut_used)
        
    shortcut_avail = np.vstack(ep_shortcut_avail).reshape(-1)
    shortcut_used = np.vstack(ep_shortcut_used).reshape(-1)
    
    
    if not test_only:
        umap = UMAP(min_dist=umap_min_dist)
        activ2d = umap.fit_transform(activ)
        
        xlim1, ylim1 = activ2d.min(axis=0)
        xlim2, ylim2 = activ2d.max(axis=0)
        norm_activ2d = activ2d.copy()
        norm_activ2d[:, 0] = (norm_activ2d[:, 0] - xlim1) / (xlim2 - xlim1)
        norm_activ2d[:, 1] = (norm_activ2d[:, 1] - ylim1) / (ylim2 - ylim1)
        
        chunks = shortcut_color_pos(pos)
    
        data = {
            'ep_pos': res['data']['pos'],
            'pos': pos,
            'angle': angle,
            'actions': np.vstack(res['actions']).reshape(-1),
            'activ2d': activ2d,
            'shortcut_avail': shortcut_avail,
            'shortcut_used': shortcut_used,
            'ep_shortcut_used': ep_shortcut_used_single,
            'ep_shortcut_avail': ep_shortcut_avail_single,
            'pos_chunks': chunks,
            'shortcut_use_rate': shortcut_use_rate,
        }
        if not skip_activs:
            activs = ep_stack_activations(res, True)
            data['activs'] = activs
        if not skip_umap:
            data['umap'] = umap
            
    else:
        data = {
            'ep_pos': res['data']['pos'],
            'pos': pos,
            'angle': angle,
            'actions': np.vstack(res['actions']).reshape(-1),
            'shortcut_avail': shortcut_avail,
            'shortcut_used': shortcut_used,
            'ep_shortcut_used': ep_shortcut_used_single,
            'ep_shortcut_avail': ep_shortcut_avail_single,
        }
        if not skip_activs:
            activs = ep_stack_activations(res, True)
            data['activs'] = activs

    return data


def test_shortcut_activations_chks(exp_string='shortcutnav_fcp{prob}reset{reset}batch{batch}',
                              prob=0.1, reset=3, batch=64, trial=0, exp_name=None, chks=[40, 70, 100, 150, 200, 300, 400],
                              verbose=1, umap_min_dist=0.5, subdir='shortcut_resets', skip_activs=True,
                              env_kwargs={}, forced_actions=None, seed=None, test_only=False, shortcut_probability=0.5):
    all_shortcut_res = []
    
    if verbose > 0:
        chks = tqdm(chks)
    
    for chk in chks:
        shortcut_res = test_shortcut_activations(exp_string=exp_string, 
                                                 prob=prob, reset=reset, batch=batch,
                                                 trial=trial, chk=chk, exp_name=exp_name, 
                                                umap_min_dist=umap_min_dist, subdir=subdir, skip_activs=skip_activs,
                                                env_kwargs=env_kwargs, forced_actions=forced_actions,
                                                seed=seed, test_only=test_only, shortcut_probability=shortcut_probability)
        all_shortcut_res.append(shortcut_res)
    return all_shortcut_res


def plot_shortcut_umap(shortcut_res, plot_type=0, ax=None, normalize=True):
    '''
    Plot some umap heatmaps
    plot_type:
        -1: Plain umap no coloring
        0: Color based on whether shortcut was available
        1: Color based on what position chunk was it
        2: Color based on position chunk and plot onto multiple axes (6 total)
        3: Color based on whether shortcut was used for trajectory
    '''
    if ax is None and plot_type in [-1, 0, 1, 3]:
        fig, ax = pplt.subplots()
    if ax is None and plot_type in [2]:
        fig, ax = pplt.subplots(ncols=6)
    
    activ2d = shortcut_res['activ2d'].copy()
    if normalize:
        activ2d = normalize_umap(activ2d)
        
    if plot_type == -1:
        ax.scatter(activ2d.T[0], activ2d.T[1], alpha=0.3)
    elif plot_type == 0:
        shortcut_avail = shortcut_res['shortcut_avail']
        ax.scatter(activ2d.T[0], activ2d.T[1], c=shortcut_avail*2-1, alpha=0.3)
    elif plot_type == 1:
        colors = shortcut_res['pos_chunks']
        for i, idxs in enumerate(colors['rect_idxs']):
            color = colors['colors'][i]
            a = activ2d[idxs]
            ax.scatter(a.T[0], a.T[1], color=color, alpha=0.3)
        
    elif plot_type == 2:
        colors = shortcut_res['pos_chunks']
        for i, idxs in enumerate(colors['rect_idxs']):
            color = colors['colors'][i]
            a = activ2d[idxs]
            ax[0].scatter(a.T[0], a.T[1], color=color, alpha=0.3)
            ax[i+1].scatter(a.T[0], a.T[1], color=color, alpha=0.3)
            
    elif plot_type == 3:
        ur_map_idxs = shortcut_res['pos_chunks']['rect_idxs'][1]
        shortcut_used = shortcut_res['shortcut_used']

        a1 = activ2d[ur_map_idxs & shortcut_used]
        ax.scatter(a1.T[0], a1.T[1], color='green', alpha=0.3)
        a2 = activ2d[ur_map_idxs & ~shortcut_used]
        ax.scatter(a2.T[0], a2.T[1], color='red', alpha=0.3)

        X = np.vstack([a1, a2])
        labels = [0]*len(a1) + [1]*len(a2)

        if len(a1) > 0 and len(a2) > 0:
            score = silhouette_score(X, labels)
        else:
            score = 0
            
        return score
    
    
def save_forced_actions(res, seed):
    '''Take a trajectory res and format into a forced action dict that can be saved'''
    if 'ep_pos' not in res and 'ep_actions' not in res and 'ep_lens' not in res:
        raise Exception('Need some sort of ep_lens, ep_pos, or ep_actions to work with')
    
    if 'ep_actions' not in res:
        if 'ep_lens' not in res:
            ep_lens = get_ep_lens(res['ep_pos'])
        ep_actions = ep_split_res(res['actions'], ep_lens)
        
    forced_actions = dict(zip(range(100), ep_actions))
    fa = {
        'forced_actions': forced_actions,
        'seed': seed,
    }
    return fa
    
    
def normalize_umap(um2d):
    xlim1, ylim1 = um2d.min(axis=0)
    xlim2, ylim2 = um2d.max(axis=0)
    um2d[:, 0] = (um2d[:, 0] - xlim1) / (xlim2 - xlim1)
    um2d[:, 1] = (um2d[:, 1] - ylim1) / (ylim2 - ylim1)
    return um2d

    
def compute_shortcut_silhouette(shortcut_res):
    activ2d = shortcut_res['activ2d'].copy()
    ur_map_idxs = shortcut_res['pos_chunks']['rect_idxs'][1]
    shortcut_used = shortcut_res['shortcut_used']

    a1 = activ2d[ur_map_idxs & shortcut_used]
    a2 = activ2d[ur_map_idxs & ~shortcut_used]
    X = np.vstack([a1, a2])
    labels = [0]*len(a1) + [1]*len(a2)

    if len(a1) > 0 and len(a2) > 0:
        score = silhouette_score(X, labels)
    else:
        score = 0
        
    return score



n_clusters = 5
cluster_labels = ['Start', 'Entrance', 'Shortcut', 'Corridor Entrance', 'Near Platform']




def kmeans_activ_pos_plot(pos, activ2d, labels=None, ax=None, n_clusters=5, km=None):
    '''
    Plot kmeans activ2d plot with positions
    Can give a combination of
    activ2d: perform kmeans on this
    activ2d/km: perform kmeans using an already fit km object
    labels: use existing labels
    '''
    format_ax = False
    if ax is None:
        format_ax = True
        fig, ax = pplt.subplots(ncols=n_clusters+1, sharey=False, sharex=False)
        
    if labels is None:
        if km is None:
            km = KMeans(n_clusters=n_clusters)
            labels = km.fit_predict(activ2d)
        else:
            labels = km.predict(activ2d)

    cluster_range = km.n_clusters if km is not None else n_clusters
    for i in range(cluster_range):
        idxs = labels == i
        ax[0].scatter(activ2d[idxs].T[0], activ2d[idxs].T[1], color=rgb_colors[i], alpha=0.1)
        ax[i+1].scatter(pos[idxs].T[0], pos[idxs].T[1], color=rgb_colors[i], alpha=0.1)
        
    if format_ax:
        ax[1:].format(xlim=[0, 300], ylim=[0, 300])
      



'''
================================================================
Predefined cluster methods
================================================================
Different method of "clustering" UMAP activations by finding which points belong
to predefined clusters based on where the agent was
Then, analyze based on how well separated the clusters are
'''


def pairwise_silscore_plot(silscores, ax=None, vmax=None, draw_cbar=True):
    '''Plot pairwise silhouette scores from pairwise_silhouette_score into
    a heatmap with text inside each block showing the numerical value
    
    vmax: Give this for a manual vmax. Note that we assume a vmin of 0
    draw_cbar: determine whether a colorbar should be drawn. This can be turned
        off if making multiple subplots and drawing a shared cbar
    '''
    if ax is None:
        fig, ax = pplt.subplots()
    
    if vmax is None:
        max_score = np.max(silscores)
    else:
        max_score = vmax
    cbar = ax.imshow(silscores, vmax=vmax)
    if draw_cbar:
        ax.colorbar(cbar)

    for i, j in itertools.product(range(5), range(5)):
        if j <= i:
            continue
        score = silscores[i, j]
        color = 'white' if score > (max_score/2) else 'black'
        ax.text(j, i, f'{score:.2f}', ha='center', va='center', color=color)

    ax.format(xformatter=['Start', 'Early Entr.', 'Early Short.','Corr. Entr.', 'Corr. Short.'], xlocator=range(5),
              xrotation=45,
              yformatter=['Start', 'Early Entr.', 'Early Short.','Corr. Entr.', 'Corr. Short.'], ylocator=range(5))

    return cbar

      
        
'''
================================================================
KMeans clustering methods
================================================================
'''        
        
        
def shortcut_km_cluster_prob2(ep_labels, pos):
    cluster_names = ['start', 'entry', 'shortcut', 'corridor', 'platform']
    cluster_idxs = [] #which label is each cluster most likely to be
    cluster_probs = []
    n_clusters=5
    labels = np.concatenate(ep_labels)

    no_remap_conflict = True
    first_labels = np.array([l[0] for l in ep_labels])
    for i in range(n_clusters):
        # are positions mostly above or below corridor?
        p = pos[labels == i]
        above_corridor = (p[:, 1] > 250).sum() / len(p)
        left = (p[:, 0] < 175).sum() / len(p)

        p2 = p[(p[:, 1] > 240) & (p[:, 1] < 260)]
        if len(p2) > 4:
            left2 = (p2[:, 0] < 125).sum() / len(p2)
            right2 = 1 - left2
        else:
            left2 = 0.
            right2 = 0.
        start = (first_labels == i).sum() / len(first_labels)

        prios = []
        if left > 0.5: #mostly left corridor
            above_prios = [3, 4]
        else:
            above_prios = [4, 3]

        if left2 > 0.5: #mostly left entrance
            below_prios = [1, 2]
        else:
            below_prios = [2, 1]

        if start > 0.5:
            prios += [0]
            if above_corridor > 0.5:
                prios += above_prios
                prios += below_prios
            else:
                prios += below_prios
                prios += above_prios
        else:
            if above_corridor > 0.5:
                prios += above_prios
                prios += below_prios
            else:
                prios += below_prios
                prios += above_prios
            prios += [0]

        for j, prio in enumerate(prios):
            if prio not in cluster_idxs:
                cluster_idxs.append(prio)
                break
            no_remap_conflict = False
                
        probs = [start, (1-above_corridor)*left2, (1-above_corridor)*right2,
                              above_corridor*left, above_corridor*(1-left)]
        cluster_probs.append(probs)
    
    # Note this is indexed as cluster_probs[cluster_idx][cluster_label likelihood]
    cluster_probs = np.vstack(cluster_probs)
    
    relabel_map = {i: cluster_idxs[i] for i in range(n_clusters)}
    
    reorder_idxs = np.argsort(list(relabel_map.values()))
    keys = list(relabel_map.keys())
    values = list(relabel_map.values())
    relabel_map2 = {}
    for idx in reorder_idxs:
        relabel_map2[values[idx]] = keys[idx]
    reordered_cluster_probs = cluster_probs[list(relabel_map2.values()), :]
    
    # Calculate transition probabilities while we are here
    reordered_ep_labels = [relabel_cluster(e, relabel_map) for e in ep_labels]
    transition_probs = shortcut_km_transition_probs(reordered_ep_labels)

    return {
        'cluster_probs': cluster_probs, 
        'relabel_map': relabel_map, 
        'cluster_names': cluster_names, 
        'no_remap_conflict': no_remap_conflict,
        'reordered_cluster_probs': reordered_cluster_probs,
        'reordered_transition_probs': transition_probs
    }


def relabel_cluster(labels, relabel_map):
    '''
    Relabel a batch of labels to fit the relabelled schema
    '''
    relabelled = labels.copy()
    for i in range(len(labels)):
        relabelled[i] = relabel_map[labels[i]]
    return relabelled


def shortcut_km_transition_probs(ep_labels, n_clusters=5, as_probs=True):
    '''
    Given episodic labels, track how likely transitions are from one cluster to the next
    
    as_probs: compute transition probabilities instead of counts
    '''
    transitions = np.zeros((n_clusters, n_clusters))
    for l in ep_labels:
        for i in range(len(l)-1):
            start = l[i]
            end = l[i+1]
            transitions[start][end] += 1
            
    if as_probs:
        for i in range(n_clusters):
            transitions[i] = transitions[i] / transitions[i].sum()
            
    return transitions



def shortcut_cluster_analysis(s, normalize=True):
    '''Pass a shortcut_res object to perform clustering analysis'''
    n_clusters = 5
    activ2d = s['activ2d'].copy()
    if normalize:
        xmin, ymin = activ2d.min(axis=0)
        xmax, ymax = activ2d.max(axis=0)
        activ2d[:, 0] = (activ2d[:, 0] - xmin) / (xmax - xmin)
        activ2d[:, 1] = (activ2d[:, 1] - ymin) / (ymax - ymin)
    
    pos = s['pos']
    ep_lens = get_ep_lens(s['ep_pos'])
    km = KMeans(n_clusters=n_clusters)
    labels = km.fit_predict(activ2d)
    ep_labels = ep_split_res(labels, ep_lens)

    res = shortcut_km_cluster_prob2(ep_labels, pos)
    relabel_map = res['relabel_map']
    
    labels = relabel_cluster(labels, relabel_map)
    
    relabelled_centers = np.zeros(km.cluster_centers_.shape)
    for i in range(n_clusters):
        relabelled_centers[relabel_map[i]] = km.cluster_centers_[i]
        
    res['labels'] = labels
    res['ep_labels'] = ep_split_res(labels, ep_lens)
    res['cluster_centers'] = relabelled_centers
    
    t = res['reordered_transition_probs'].copy()
    for i in range(len(t)):
        t[i, i] = 0
        if t[i].sum() != 0:
            t[i] = t[i] / t[i].sum()
        else:
            t[i] = 0.
    res['t_prob'] = t
    res['km'] = km
    
    res['ep_activ2d'] = ep_split_res(activ2d, ep_lens)
    res['activ2d'] = activ2d
    return res


