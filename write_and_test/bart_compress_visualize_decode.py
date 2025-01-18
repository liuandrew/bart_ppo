import matplotlib.pyplot as plt
import matplotlib
import proplot as pplt
import numpy as np
import pandas as pd
from torch import nn, optim
import torch
from plotting_utils import rgb_colors, get_color_from_colormap
from bart_representation_analysis import comb_pca, normalize_obs

from scipy.stats import ttest_ind
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, Lasso, LassoCV
from sklearn.preprocessing import StandardScaler
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from sklearn.metrics import (
    silhouette_score, 
    mean_squared_error, 
    root_mean_squared_error,
    r2_score
)
from captum.attr import LayerIntegratedGradients, LayerConductance, IntegratedGradients
    


pplt.rc.update({'font.size': 10})

color_to_idx = {"red": 0, "orange": 1, "yellow": 2,
                        "gray": 4, "purple": 4}
idx_to_color = {0: "red", 1: "orange", 2: "yellow",
                            3: "gray", 4: "purple"}
bart_plot_colors = {0: 'deep red', 
                    1: 'orange', 
                    2: 'goldenrod'}


'''
Labels for plotting convenience
'''
give_rew = ['giverew_', 'fixprev_']
postfixes = ['pop0', 'pop0.1', 'pop0.2', 'pop0.4']
models = [1.0, 1.2, 1.5, 1.7, 2.0]
trials = range(3)
chks = np.arange(10, 243, 30)

# give_labels = ['Rew not shown', 'Rew shown', 'Prev Action Shown']
pop_labels = ['0', '-0.1', '-0.2', '-0.4']
give_labels = ['Reward shown', 'Not shown']
pop_vals = [0, -0.1, -0.2, -0.4]
p_labels = ['1.0',' 1.2', '1.5', '1.7', '2.0']
chk_labels = [str(c) for c in chks]
chk_axis = 'Checkpoint'
pop_axis = 'Punishment on pop'
p_axis = 'p'

iterators = [give_rew, postfixes, models, trials, chks]
# iterators = [postfixes, models, trials, chks]
iterators_idxs = [range(len(i)) for i in iterators]
sizes = [len(i) for i in iterators]


'''

Data convenience
A lot of code is repeated to get data out of a res dictionary
from an evaluation call, this simplifies things a bit

'''

def get_impulsivity_data(res, impulsive_thres=0.2, ep=None, layer='shared1',
                         load_global=True):
    '''
    Collect commbined data used in impulsivity analysis
    Note, load_global will not generally work as expected on import,
        as globals() adds it to the file I believe
    '''
    if ep is not None:
        if type(ep) != list:
            ep = [ep]
        v = np.vstack([res['values'][e] for e in ep])
        ap = np.vstack([res['action_probs'][e] for e in ep])[:, 1]
        if layer == 'rnn_hxs':
            activ = np.vstack([res['rnn_hxs'][e] for e in ep])
        else:
            activ = np.vstack([res['activations'][layer][e] for e in ep])
    else:
        v = np.vstack(res['values'])
        ap = np.vstack(res['action_probs'])[:, 1]
        if layer == 'rnn_hxs':
            activ = np.vstack(res['rnn_hxs'])
        else:
            activ = np.vstack(res['activations'][layer])
        
    imp_steps = ap < impulsive_thres
    
    if load_global:
        globals()['v'] = v
        globals()['ap'] = ap
        globals()['activ'] = activ
        globals()['imp_steps'] = imp_steps

    return {
        'v': v,
        'ap': ap,
        'activ': activ,
        'imp_steps': imp_steps
    }

    
def get_data_from_summary(res, idx, use_longer_keys=False):
    lens = res['all_lens'][idx]
    values = res['values'][idx]
    action_probs = res['action_probs'][idx]
    bsizes = [np.mean(b[b != 0]) for b in res['last_sizes'][idx]]

    vs = []
    aps = []
    final_vs = []
    final_errs = []
    for ep in range(17):
        l = int(lens[ep])
        v = values[ep][:l]
        ap = action_probs[ep][:l]
        steps = np.arange(len(ap))
        imp_steps = ap < 0.5
        x = steps[imp_steps]
        smoothed_ap = pd.Series(ap[imp_steps]).ewm(alpha=0.01).mean()
        smoothed_v = pd.Series(v[imp_steps]).ewm(alpha=0.01).mean()
        final_vs.append(smoothed_v.iloc[-1])
        final_errs.append(smoothed_ap.iloc[-1])
        vs.append(v)
        aps.append(ap)
    if use_longer_keys:
        return {
            'values': vs,
            'action_prob': aps,
            'final_v': final_vs,
            'final_err': final_errs,
            'bsizes': bsizes,
        }
    else:
        return {
            'v': vs,
            'ap': aps,
            'final_v': final_vs,
            'final_err': final_errs,
            'bsizes': bsizes,
        }

    


def split_by_ep(res, data):
    '''After combining steps, return back to episode based split'''
    ep_lens = [len(o) for o in res['obs']]

    ep_data = []
    cur_step = 0
    for ep_len in ep_lens:
        ep_data.append(data[cur_step:cur_step+ep_len])
        cur_step += ep_len
    return ep_data


def split_by_lens(lens, data):
    cur = 0
    ep_data = []
    for l in lens:
        ep_data.append(data[cur:cur+l])
        cur = cur + l
    return ep_data


    
def print_model(idx):
    '''Print out the model type of the given index
    idx: 4-tuple (h, i, j, k)'''
    mstr = ''
    if idx[0] == 0:
        mstr += 'No Rew Info, '
    else:
        mstr += 'Rew Given, '
        
    mstr += f'Pop: {pop_labels[idx[1]]}, '
    mstr += f'p: {p_labels[idx[2]]}, '
    mstr += f'Trial {idx[3] + 1}'
    print(mstr)
    
'''

Compression methods

Methods for compressing activation data. We have 3 main methods:
1. PCA
2. Kmeans
3. Bottleneck networks

Note that comb_pca in the bart_representation_analysis.py file also falls
into this area

'''

def get_cluster_activations(res, layer='shared1', labels=None, kmeans=None, k=5, orientation=None,
                            random_state=0, ret_activ=False):
    '''
    Use kmeans on hidden state data to cluster the data after scaling
    labels: if passed, use predetermined kmeans labels
    kmeans: if passed, use an already fit KMeans model, rather than fitting a new one
    ret_activ: if True, pass back the normalized, oriented, and clustered activations
    '''
    if layer == 'rnn_hxs':
        activ = np.vstack(res['rnn_hxs'])
    else:
        activ = np.vstack(res['activations'][layer])
    if orientation is not None:
        activ = activ * orientation
    data = activ.T  # change to [64, T]
    scaler = TimeSeriesScalerMeanVariance()
    data_normalized = scaler.fit_transform(data[:, :, np.newaxis])  # Shape becomes [64, T, 1]
    data_normalized = data_normalized.squeeze()  # Back to shape [T, 64]
    if labels is None:
        if kmeans is None:
            kmeans = KMeans(n_clusters=k, random_state=0)
        else:
            k = kmeans.n_clusters
        labels = kmeans.fit_predict(data_normalized)
    else:
        k = np.max(labels) + 1
    
    
    data_normalized = data_normalized.T 
    cluster_data = [data_normalized[:, labels == i] for i in range(k)]
    cluster_activations = np.vstack([c.mean(axis=1) for c in cluster_data]).T

    if ret_activ:
        return cluster_activations, labels, kmeans, cluster_data, data_normalized
    else:
        return cluster_activations, labels, kmeans



def find_k_cluster_activations(res, layer='shared1', require_ap_explained=True):
    '''
    Determine an optimal k value for kmeans clustering on activation
    data
    require_ap_explained: If True, require that k also captures sufficient
        information that would be able to predict action (80% accuracy compared to
        uncompressed activity)
        ! Probably could change this to be some other measure of data loss
    '''
    imp = get_impulsivity_data(res, layer=layer, load_global=False)
    activ = imp['activ']
    ap = imp['ap'].reshape(-1, 1)
    
    data = activ.T  # change to [64, T]
    scaler = TimeSeriesScalerMeanVariance()
    data_normalized = scaler.fit_transform(data[:, :, np.newaxis])  # Shape becomes [64, T, 1]
    data_normalized = data_normalized.squeeze()  # Back to shape [64, T]

    max_k = 10  # Maximum number of clusters to try
    silhouette_scores = []
    r2_scores = []
    ks = range(2, max_k + 1)

    lr = LinearRegression()
    lr.fit(activ, ap)
    ypred = lr.predict(activ)
    best_r2 = r2_score(ap, ypred)
    
    for k in ks:
        kmeans = KMeans(n_clusters=k, random_state=0)
        labels = kmeans.fit_predict(data_normalized)
        silhouette_avg = silhouette_score(data_normalized, labels)
        silhouette_scores.append(silhouette_avg)

        # Test how good the cluster compression explains behavior
        cluster_activations, _, _ = get_cluster_activations(res, layer, kmeans=kmeans)
        lr = LinearRegression()
        lr.fit(cluster_activations, ap)
        ypred = lr.predict(cluster_activations)
        r2_scores.append(r2_score(ap, ypred))
        
    r2_scores = np.array(r2_scores)
    silhouette_scores = np.array(silhouette_scores)
    
    # Ensure that enough info is kept in clusters to keep r2 prediction high
    if require_ap_explained:
        min_r2_idx = np.argmax(r2_scores > 0.7*best_r2)
        best_k = np.argmax(silhouette_scores[min_r2_idx:]) + min_r2_idx + 2
    else:
        best_k = np.argmax(silhouette_scores) + 2
        
    return best_k




def kmeans_oriented_activations(res, layer='rnn_hxs', require_ap_explained=True):
    '''
    Perform k-means but also flip activations to allow for reversed activations to cluster
    together. Automatically determine the appropriate orientations of each nodes activity
    according to which orientation has better clustering
    
    layer: shared0/shared1/critic0/critic1/actor0/actor1/rnn_hxs
        rnn_hxs is slightly different than shared1 by 1 timestep, it is the input at time t
        whereas shared1 is the output at time t
    '''
    imp = get_impulsivity_data(res, layer=layer, load_global=False)
    
    activ = imp['activ']
    ap = imp['ap'].reshape(-1, 1)
    combactiv = np.hstack([activ, -activ])
    data = combactiv.T
    scaler = TimeSeriesScalerMeanVariance()
    data_normalized = scaler.fit_transform(data[:, :, np.newaxis])  # Shape becomes [64, T, 1]
    data_normalized = data_normalized.squeeze()  # Back to shape [64, T]


    max_k = 20  # Maximum number of clusters to try
    silhouette_scores = []
    r2_scores = []
    ks = range(2, max_k + 1)

    lr = LinearRegression()
    lr.fit(activ, ap)
    ypred = lr.predict(activ)
    best_r2 = r2_score(ap, ypred)

    for k in ks:
        kmeans = KMeans(n_clusters=k, random_state=0)
        labels = kmeans.fit_predict(data_normalized)
        silhouette_avg = silhouette_score(data_normalized, labels)
        silhouette_scores.append(silhouette_avg)

        # Test how good the cluster compression explains behavior
        cluster_activations, _, _ = get_cluster_activations(res, layer, kmeans=kmeans)
        lr = LinearRegression()
        lr.fit(cluster_activations, ap)
        ypred = lr.predict(cluster_activations)
        r2_scores.append(r2_score(ap, ypred))
    r2_scores = np.array(r2_scores)
    silhouette_scores = np.array(silhouette_scores)
    
    # Ensure that enough info is kept in clusters to keep r2 prediction high
    if require_ap_explained:
        min_r2_idx = np.argmax(r2_scores > 0.7*best_r2)
        best_k = np.argmax(silhouette_scores[min_r2_idx:]) + min_r2_idx + 2
    else:
        best_k = np.argmax(silhouette_scores) + 2
    
    # Orient activations
    cluster_activ, labels, kmeans = get_cluster_activations(res, layer, k=best_k)
    orientation = []
    labels = []
    for i in range(64):
        right_dist = np.linalg.norm(kmeans.cluster_centers_ - activ[:, i], axis=1)
        left_dist = np.linalg.norm(kmeans.cluster_centers_ + activ[:, i], axis=1)

        if right_dist.min() < left_dist.min():
            orientation.append(1)
            labels.append(np.argmin(right_dist))
        else:
            orientation.append(-1)
            labels.append(np.argmin(left_dist))
            
    orientation = np.array(orientation)
    # Prune any clusters that remain unused
    labels = np.array(labels)
    cluster_counts = [(labels == i).sum() for i in range(7)]
    new_k = (np.array(cluster_counts) != 0).sum()
    cluster_activations, labels, kmeans = get_cluster_activations(res, layer, k=new_k,
                                                                  orientation=orientation)
    return new_k, cluster_activations, labels, kmeans, orientation
        



class BottleneckNetwork(nn.Module):
    def __init__(self, compress_layer=2, input_size=64):
        super(BottleneckNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, compress_layer)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(compress_layer, 1)

    def forward(self, x):
        bottleneck = self.relu(self.fc1(x))
        out = self.fc2(bottleneck)
        return out, bottleneck


def train_bottleneck_compressor(res, y=None, n_components=3, layer='shared1'):
    '''
    Train a bottleneck network to ccompress activation data
    Notably, this will try to comopress data in a way that is most beneficial
        to encode the target y variable.
    If y is not specified, this network will be focused on preserving
        activities related to the "impulsivity" metric
    '''
    nn_model = BottleneckNetwork(compress_layer=n_components)
    criterion = nn.MSELoss()
    opt = optim.Adam(nn_model.parameters(), lr=0.01)

    # eps = np.concatenate([np.arange(6), np.arange(8, 17)])

    a = np.vstack(res['activations'][layer])
    x = torch.tensor(a)
    if y is None:
        ap = np.vstack(res['action_probs'])[:, 1]
        unc = ap[ap < 0.2]
        a = a[ap < 0.2]
        x = torch.tensor(a)
        y = torch.tensor(unc).reshape(-1, 1)
    else:
        y = torch.tensor(y)
    # print(x.shape, y.shape)

    nn_model.train()
    for epoch in range(500):
        opt.zero_grad()
        out, _ = nn_model(x)
        loss = criterion(out, y)
        
        loss.backward()
        opt.step()

    nn_model.eval()
    with torch.no_grad():
        y_pred, _ = nn_model(x)
    
    mse_nn = mean_squared_error(y, y_pred)
    r_squared_nn = r2_score(y, y_pred)

    return nn_model, (mse_nn, r_squared_nn)

'''

Visualization methods

Various methods to help with visualizaing how compression worked
or how compressed activities change with the different balloon conditions

'''


def visualize_cluster_activations(res, klabels, ep=8, layer='shared1', 
                                  orientation=None,
                                  step1=100, step2=200, ax=None,
                                  format=True, include_components=True):
    '''
    Visualize the compositionn of each cluster across a single episode

    format
    '''
    k = np.max(klabels) + 1
    if k > 20:
        print('k > 20, not creating plot')
        return None
    if orientation is None:
        orientation = np.ones(len(klabels))

    cluster_activations, _, _, cluster_data, _ = get_cluster_activations(res, layer, klabels, ret_activ=True,
                                                                         orientation=orientation)
    cluster_activ = split_by_ep(res, cluster_activations)[ep]
    
    if ax is None:
        fig, ax = pplt.subplots(nrows=k, sharex=True, sharey=True, 
                                figwidth=5, refaspect=4)
        
    for i in range(k):
        
        a = split_by_ep(res, cluster_data[i])[ep]
        if include_components:
            for j in range(cluster_data[i].shape[1]):
                ax[i].plot(a[step1:step2, j], alpha=0.5)
        ax[i].plot(cluster_activ[step1:step2, i], c='black')
        if format:
            ax[i].format(title=f'Cluster {i+1}')
    if format:
        ax.format(suptitle='Time Series Clusters Visualization', xlabel='Time', ylabel='Normalized Value')


def visualize_cluster_activations_combined(res, klabels, cluster_activ, eps=[2, 12], layer='rnn_hxs',
                                           orientation=None, include_components=True):
    '''
    Make a combined visualization of cluster activations for small balloon,
    large balloon, and over all time periods

    include_components: whether to also plot the component pieces
    '''
    k = cluster_activ.shape[1]
    sizes = np.round(np.arange(0.2, 1.01, 0.05), 2)
    fig, ax = pplt.subplots(nrows=k, ncols=3, figwidth=8, refaspect=3,
                            wspace=0)
    visualize_cluster_activations(res, klabels, ep=eps[0], layer='rnn_hxs',
                                orientation=orientation, ax=ax[:, 0],
                                format=False, include_components=include_components)
    visualize_cluster_activations(res, klabels, ep=eps[1], layer='rnn_hxs',
                                orientation=orientation, ax=ax[:, 1],
                                format=False, include_components=include_components)
    for i in range(k):
        ax[i, 2].plot(cluster_activ[:, i], c='gray7')

    ax.format(leftlabels=[f'Cluster {i+1}' for i in range(k)],
            toplabels=[f'$\mu={sizes[eps[0]]}$', f'$\mu={sizes[eps[1]]}$', 
                       'All eps'])


def get_size_episode_coverage(res):
    '''
    Get episodes out of the 17 episodes that capture the different sizes the agent
    achieved, to use for visualization rather than using all episodes
    '''
    ep_idxs = []
    kept_means = []
    means = np.array([np.mean(s[-10:]) for s in res['data']['last_size']])
    for ep in range(17):
        if np.all(np.abs(np.array(kept_means) - means[ep]) > 0.02):
            kept_means.append(means[ep])
            ep_idxs.append(ep)
    return ep_idxs
    


def visualize_smoothed_cluster_episodes(res, klabels, layer='shared1',
                                        color_by_mu=False, minimal_size_coverage=False):
    '''
    Visualize how clustered acivations work smoothed across episodes and colored
    by balloon size
    color_by_mu: if True, color lines based on true balloon size, otherwise
        color by the average size the agent achieved on the episode
    minimal_size_coverage: if True, only use as many episodes as needed to see range of 
        sizes achieved
    '''
    k = np.max(klabels) + 1
    if k > 20:
        print('k > 20, not creating plot')
        return None
    fig, ax = pplt.subplots(nrows=k, sharex=True, sharey=False, 
                             figwidth=5, refaspect=4)
    
    if minimal_size_coverage:
        ep_idxs = get_size_episode_coverage(res)
    else:
        ep_idxs = range(17)

    if color_by_mu:
        bsizes = np.arange(0.2, 1.01, 0.05)
        vmin = 0.2
        vmax = 1
    else:
        bsizes = []
        for ep in ep_idxs:
            popped = np.array(res['data']['popped'][ep])
            sizes = np.array(res['data']['last_size'][ep])
            bsizes.append(np.mean(sizes[~popped]))
        vmin = np.min(bsizes)
        vmax = np.max(bsizes) + 0.05
        
    for i in range(k):
        for j, ep in enumerate(ep_idxs):
            s = bsizes[j]
            c = get_color_from_colormap(s, vmin, vmax, to_hex=False) 
            cluster_activ = res['activations'][layer][ep][:, klabels == i].mean(axis=1)
            ax[i].plot(list(pd.Series(cluster_activ).ewm(alpha=0.01).mean()),
                    c=c)

    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
    cbar = fig.colorbar(sm)
    cbar.set_label(label='$\mu$', rotation=0, labelpad=10)
    ax[0].format(title='RNN Clusters')
    ax.format(xlabel='time step')
    for i in range(k):
        ax[i].format(ylabel=f'Cluster {i+1}')
 
 
 
def visualize_smoothed_pca_episodes(res, n_components=3, layer='shared1',
                                        color_by_mu=False, minimal_size_coverage=False):
    '''
    Visualize how clustered acivations work smoothed across episodes and colored
    by balloon size
    color_by_mu: if True, color lines based on true balloon size, otherwise
        color by the average size the agent achieved on the episode
    minimal_size_coverage: if True, only use as many episodes as needed to see range of 
        sizes achieved
    '''
    fig, ax = pplt.subplots(nrows=n_components, sharex=True, sharey=False, 
                             figwidth=5, refaspect=4)
    
    if minimal_size_coverage:
        ep_idxs = get_size_episode_coverage(res)
    else:
        ep_idxs = range(17)

    if color_by_mu:
        bsizes = np.arange(0.2, 1.01, 0.05)
        vmin = 0.2
        vmax = 1
    else:
        bsizes = []
        for ep in ep_idxs:
            popped = np.array(res['data']['popped'][ep])
            sizes = np.array(res['data']['last_size'][ep])
            bsizes.append(np.mean(sizes[~popped]))
        vmin = np.min(bsizes)
        vmax = np.max(bsizes) + 0.05
        
    pcas = comb_pca(res, layer)
    for i in range(n_components):
        for j, ep in enumerate(ep_idxs):
            s = bsizes[j]
            c = get_color_from_colormap(s, vmin, vmax, to_hex=False) 
            pca = pcas[ep][:, i]
            ax[i].plot(list(pd.Series(pca).ewm(alpha=0.01).mean()),
                    c=c)

    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
    cbar = fig.colorbar(sm)
    cbar.set_label(label='$\mu$', rotation=0, labelpad=10)
    ax[0].format(title='RNN PCAs')
    ax.format(xlabel='time step')
    for i in range(n_components):
        ax[i].format(ylabel=f'PCA {i+1}')
        
        
            
def visualize_episode_values(res, color_by_mu=False, minimal_size_coverage=False,
                             cbar_label=False, ax=None):
    '''
    Visualize how the agent's value prediction looks
    '''
    if minimal_size_coverage:
        ep_idxs = get_size_episode_coverage(res)
    else:
        ep_idxs = range(17)

    if color_by_mu:
        bsizes = np.arange(0.2, 1.01, 0.05)
        vmin = 0.2
        vmax = 1
    else:
        bsizes = []
        for ep in ep_idxs:
            popped = np.array(res['data']['popped'][ep])
            sizes = np.array(res['data']['last_size'][ep])
            bsizes.append(np.mean(sizes[~popped]))
        vmin = np.min(bsizes)
        vmax = np.max(bsizes) + 0.05
        
    if ax is None:
        fig, ax = pplt.subplots()
    for j, ep in enumerate(ep_idxs):
        s = bsizes[j]
        c = get_color_from_colormap(s, vmin, vmax, to_hex=False)
        ax.plot(list(pd.Series(res['values'][ep].reshape(-1)).ewm(alpha=0.1).mean()), 
                c=c, alpha=0.5)
    ax.format(xlabel='time step', ylabel='Value prediction')
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
    cbar = ax.colorbar(sm)
    if cbar_label:
        cbar.set_label(label='$\mu$', rotation=0, labelpad=10)
    

def visualize_episode_values_from_summary(res, idx, with_erratic=True,
                                          color_by_mu=False, cbar_label=False,
                                          ax=None):
    r = get_data_from_summary(res, idx)
    bsizes = r['bsizes']
    if color_by_mu:
        bsizes = np.arange(0.2, 1.01, 0.05)
        vmin = 0.2
        vmax = 1
    else:
        vmin = np.min(bsizes)
        vmax = np.max(bsizes) + 0.05
        
    if ax is None:
        ncols = 2 if with_erratic else 1
        fig, ax = pplt.subplots(ncols=ncols, sharey=False)
    for j, ep in enumerate(range(17)):
        s = bsizes[j]
        c = get_color_from_colormap(s, vmin, vmax, to_hex=False)
        ax_ = ax[1] if with_erratic else ax
        ax_.plot(list(pd.Series(r['v'][ep].reshape(-1)).ewm(alpha=0.1).mean()), 
                c=c, alpha=0.5)
        ax_.format(xlabel='time step', ylabel='Value prediction')

        if with_erratic:
            ap = r['ap'][ep]
            steps = np.arange(len(ap))
            imp_steps = ap < 0.5
            x = steps[imp_steps]
            smoothed_ap = pd.Series(ap[imp_steps]).ewm(alpha=0.01).mean()
            ax[0].plot(x, smoothed_ap, c=c, alpha=0.5)
            
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
    ax_ = ax[1] if with_erratic else ax
    cbar = ax_.colorbar(sm)
    if cbar_label:
        cbar.set_label(label='$\mu$', rotation=0, labelpad=10)


def visualize_smoothed_nn_episodes(res, nn_model, layer='shared1',
                                        color_by_mu=False, minimal_size_coverage=False):
    '''
    Visualize how bottleneck nn activations work smoothed across episodes and colored
    by balloon size
    color_by_mu: if True, color lines based on true balloon size, otherwise
        color by the average size the agent achieved on the episode
    minimal_size_coverage: if True, only use as many episodes as needed to see range of 
        sizes achieved
    '''
    n = nn_model.fc1.out_features
    fig, ax = pplt.subplots(nrows=n, sharex=True, sharey=False, 
                             figwidth=5, refaspect=4)
    
    if minimal_size_coverage:
        ep_idxs = get_size_episode_coverage(res)
    else:
        ep_idxs = range(17)

    if color_by_mu:
        bsizes = np.arange(0.2, 1.01, 0.05)
        vmin = 0.2
        vmax = 1
    else:
        bsizes = []
        for ep in ep_idxs:
            popped = np.array(res['data']['popped'][ep])
            sizes = np.array(res['data']['last_size'][ep])
            bsizes.append(np.mean(sizes[~popped]))
        vmin = np.min(bsizes)
        vmax = np.max(bsizes) + 0.05
        
    pcas = comb_pca(res, layer)
    for j, ep in enumerate(ep_idxs):
        a = res['activations'][layer][ep]
        with torch.no_grad():
            _, compressed = nn_model(a)
        for i in range(n):
            s = bsizes[j]
            c = get_color_from_colormap(s, vmin, vmax, to_hex=False) 
            feat = compressed[:, i]
            ax[i].plot(list(pd.Series(feat).ewm(alpha=0.01).mean()),
                    c=c)

    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
    cbar = fig.colorbar(sm)
    cbar.set_label(label='$\mu$', rotation=0, labelpad=10)
    ax[0].format(title='RNN PCAs')
    ax.format(xlabel='time step')
    for i in range(n):
        ax[i].format(ylabel=f'PCA {i+1}')



def visualize_regressor_coefficients(coefs, by_clusters=True, ax=None,
                                     legend=True):
    '''
    Visualize the coefficients found for each of the regression models
    coefs: should be of shape [3, N] where N is the number of clusters/PCAs
    by_clusters: just used for labeling
    '''
    if ax is None:
        fig, ax = pplt.subplots(refaspect=4, figwidth=6)
    labels = ['Ramp', 'Value', 'Impulsivity']
    for i, coef in enumerate(coefs):
        k = len(coef)
        x = np.arange(k)+i*0.2
        idxs = coef != 0
        y = np.abs(coef)
        y[~idxs] = 0
        ax.bar(x, y, width=0.15, label=labels[i])
        
    xlabel = 'Cluster number' if by_clusters else 'PCA'
    ax.format(xlocator=np.arange(10)+0.2,
            xformatter=[str(x) for x in range(1, 11)],
            xlabel=xlabel,
            ylabel='Coefficient',
            title='Coefficients for regressors')
    if legend:
        ax.legend(loc='ur')
    return ax



def plot_cluster_grads(grads, labels, ax=None, bar=False):
    cgrads = cluster_grads(grads, labels)
    k = np.max(labels)+1
    ymax = grads.max()
    ymin = grads.min()
    
    if ax is None:
        fig, ax = pplt.subplots()
    for i in range(k):
        if bar:
            ax.bar(i, np.mean(cgrads[i]), c=rgb_colors[0])
        else:
            ax.boxplot(i, cgrads[i])
            t = ttest_ind(cgrads[i], grads)
            p = t.pvalue
            s = t.statistic
            star = ''
            if p < 0.05:
                star = '*'
            elif p < 0.005:
                star = '**'
            elif p < 0.0005:
                star = '***'
            if s < 0:
                star_color = 'red'
            else:
                star_color = 'black'
            ax.text(i, ymax*1.03, star, ha='center', c=star_color)
    if not bar:
        ax.format(ylim=[ymin*0.85, ymax*1.2])

    ax.format(xlabel='Cluster number', ylabel='',
              xlocator=range(k), xformatter=[str(i+1) for i in range(k)])



def visualize_cluster_connectivity(model=None, obs_rms=None, res=None, cluster_activ=None, labels=None,
                                   val_grads=None, act_grads=None, influences=None, cluster_influence=None,
                                   give=False, ep=8, step1=100, step2=130, k=None):
    '''
    Create cluster connectivity plot based on an individual evaluation result set
    Ex. (Default mode, collect data and run)
    idx, model, obs_rms, r = select_random_model(load_models=True)
    k, cluster_activ, labels, kmeans, orientation = kmeans_oriented_activations(r)
    visualize_cluster_connectivity(model, obs_rms, r, cluster_activ, labels)
    
    Ex. (Existing data mode, influences provided)
    val_grads = ares['val_grads'][idx]
    act_grads = ares['action_grads'][idx]
    influences = ares['rnn_hx_influences'][idx]
    labels = ares['cluster_labels'][idx]
    cluster_activ = ares['cluster_activations'][idx]
    visualize_cluster_connectivity(cluster_activ=cluster_activ, labels=labels,
                                val_grads=val_grads, act_grads=act_grads,
                                influences=influences)
    
    Ex. (Existing data mode, cluster activ and influencces provided)
    act_grads = forced_ares['all_action_grads']
    val_grads = forced_ares['all_value_grads']
    cinfluences = forced_ares['meta_cluster_influences']
    labels = forced_ares['labels']
    cactiv = forced_ares['kmeans'].cluster_centers_.T
    visualize_cluster_connectivity(val_grads=val_grads, act_grads=act_grads,
                                labels=labels, cluster_influence=cinfluences,
                                cluster_activ=cactiv, step1=700, step2=740) 
    '''
    if k is None:
        k = np.max(labels) + 1
    if model is not None and obs_rms is not None:
        # Default mode: require
        #  model, obs_rms, res, cluster_activ, labels
        # Measure cluster influences on each other
        cumu_influences = compute_rnn_hxs_influences(model, res, max_unroll=4, nsteps=300)
        cluster_influence = get_cluster_influences(cumu_influences, labels)
        # Measure cluster influences on val and pol
        val_scores, pol_scores = get_val_and_action_scores(model, obs_rms, res, labels,
                                                        give=give, plot=False)
        a = split_by_ep(res, cluster_activ)[ep]
    elif val_grads is not None and act_grads is not None and \
         cluster_influence is None:
        # Existing data mode: require
        #  labels, val_grads, act_grads, influences, cluster_activ
        cluster_influence = get_cluster_influences(influences, labels)
        a = cluster_activ
        val_scores, pol_scores = get_val_and_action_scores(labels=labels,
                                                           val_grads=val_grads,
                                                           act_grads=act_grads)
    elif val_grads is not None and act_grads is not None and \
         cluster_influence is not None:
        # Existing data mode: require
        #  labels, val_grads, act_grads, cluster_influence, cluster_activ
        a = cluster_activ
        val_scores, pol_scores = get_val_and_action_scores(labels=labels,
                                                           val_grads=val_grads,
                                                           act_grads=act_grads)
    else:
        a = cluster_activ
        val_scores, pol_scores = None, None
        
    t = np.arange(step1, step2)
    ts_list = [a[step1:step2, i] for i in range(k)]

    if res is not None:
        bsteps = np.array([s for s in res['data']['balloon_step'][ep] if s in range(step1, step2)])
        bstep_r = np.array(res['rewards'][ep][bsteps] > 0)

    ymins = cluster_activ.min(axis=0)
    ymaxs = cluster_activ.max(axis=0)
    W = cluster_influence

    theta = np.linspace(0, 2 * np.pi, k, endpoint=False)
    R = 1
    x = R * np.cos(theta)
    y = R * np.sin(theta)

    fig = pplt.figure(figwidth=6)
    ax = fig.add_subplot(111)
    ax.set_xlim(-1.5 * R, 1.5 * R)
    ax.set_ylim(-1.5 * R, 1.5 * R)
    ax.set_aspect('equal')
    ax.axis('off')

    trans = ax.transData.transform
    inv = fig.transFigure.inverted().transform
    w = 0.1
    for i in range(k):
        x_fig = (R + 2.5*w) * np.cos(theta[i])
        y_fig = (R + 2.5*w) * np.sin(theta[i])
        x_fig2, y_fig2 = inv(trans((x_fig, y_fig)))
        ax_inset = fig.add_axes([x_fig2 - w*0.9, y_fig2 - w*0.6, w, w])
        # ax_inset.plot(t, ts_list[i], color='black', zorder=1)
        ax_inset.scatter(t, ts_list[i], color='black', zorder=1, s=10)
        
        if res is not None:
            ax_inset.scatter(bsteps[bstep_r], a[bsteps[bstep_r], i], c=rgb_colors[2], s=15, zorder=2)
            ax_inset.scatter(bsteps[~bstep_r], a[bsteps[~bstep_r], i], c=rgb_colors[3], s=15, zorder=2)
        ax_inset.set_xticks([])
        ax_inset.set_yticks([])
        ax_inset.set_ylim([ymins[i], ymaxs[i]])
        # ax_inset.axis('off')
        cstr = f'C{i+1} '
        if val_scores and val_scores[i] is not None:
            cstr += f'V:{val_scores[i]:.2f} '
        if pol_scores and pol_scores[i] is not None:
            cstr += f'$\pi$:{pol_scores[i]:.2f} '
        
        ax.text(0, 1.1, cstr, transform=ax_inset.transAxes)
        
        
    # circle = plt.Circle((0, 0), radius=R, 
    #                     fill=False, edgecolor='black', linewidth=2)
    # ax.add_patch(circle)

    min_weight = np.min(W)
    max_weight = np.max(np.abs(W))

    for i in range(k):
        for j in range(k):
            if i != j and W[i, j] != 0:
                x_start, y_start = x[j], y[j]
                x_end, y_end = x[i], y[i]
                
                if i > j:
                    offset = -0.05
                    dir_x = x_end - x_start
                    dir_y = y_end - y_start

                    # Calculate perpendicular unit vector
                    length = np.sqrt(dir_x**2 + dir_y**2)
                    unit_perp_x = -dir_y / length
                    unit_perp_y = dir_x / length

                    # Offset the line in the perpendicular direction
                    x_start = x_start + offset * unit_perp_x
                    y_start = y_start + offset * unit_perp_y
                    x_end = x_end + offset * unit_perp_x
                    y_end = y_end + offset * unit_perp_y

                
                weight = W[i, j]
                color = 'red' if weight > 0 else 'blue'
                alpha = (np.abs(weight) / max_weight)**1.7

                ax.annotate(
                    '',
                    xy=(x_end, y_end),
                    xytext=(x_start, y_start),
                    arrowprops=dict(
                        arrowstyle='->',
                        color=color,
                        alpha=alpha,
                        linewidth=2,
                        shrinkA=20,
                        shrinkB=20,
                    )
                )

'''

Activity decoding methods
The core of the analysis looks at how well we can decode certain aspects of
behavior or "intent" from the activity of the network
These methods will primarily emmploy Lasso regression to determine which
nodes are important to some particular behavior measures

'''
    
def train_lasso_with_value_for_impulsivity(res, activ=None, alpha=1e-3, impulsive_thres=0.2):
    '''
    Train a Lasso regression to determine the importance of certain features
    on impulsivity measurements
    impulsive_thres: threshold below which we consider steps impulsive
    activ: optionally pass specific activations, such as cluster activations
    '''
    imp = get_impulsivity_data(res, impulsive_thres=impulsive_thres, load_global=False)
    v = imp['v']
    ap = imp['ap']
    imp_steps = imp['imp_steps']
    if activ is None:
        activ = imp['activ']
    value_estimation = v[imp_steps].reshape(-1)
    impulsivity = ap[imp_steps].reshape(-1)
    activ = activ[imp_steps]
    
    X = np.column_stack((value_estimation, activ))
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    lasso = Lasso(alpha=alpha)
    lasso.fit(X_scaled, impulsivity)

    # 4. Identifying Significant Neurons
    # Extract coefficients and feature names
    coefficients = lasso.coef_
    intercept = lasso.intercept_

    # Create a DataFrame for easy interpretation
    feature_names = ['value_estimation'] + [f'activ_{i}' for i in range(activ.shape[1])]
    coef_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coefficients
    })

    # Filter out features with non-zero coefficients
    significant_features = coef_df[coef_df['Coefficient'] != 0]
    return significant_features



def compute_regressor_coefficients(res, layer='shared1', by_clusters=True):
    '''
    Generate coefficients for regression models to explain different portions of
    behavior in meta bart agents
    by_clusters: if True, use kmeans clustering
        else, use PCA components
    '''
    
    if by_clusters:
        # Perform kmeans on the activities of shared layer
        k = find_k_cluster_activations(res, layer=layer)
        cluster_activ, labels, kmeans = get_cluster_activations(res, layer=layer, k=k)
        # Scale new cluster compressed activations
        scaler = TimeSeriesScalerMeanVariance()
        cluster_norm = scaler.fit_transform(cluster_activ[:, :, np.newaxis])
        cluster_norm = cluster_norm.reshape(cluster_activ.shape)
    else:
        cluster_activ = np.vstack(comb_pca(res, n_components=6, layer=layer))
        scaler = TimeSeriesScalerMeanVariance()
        cluster_norm = scaler.fit_transform(cluster_activ[:, :, np.newaxis])
        cluster_norm = cluster_norm.reshape(cluster_activ.shape)
        
    # Set up data
    actions = np.vstack(res['actions'])
    imp = get_impulsivity_data(res, layer=layer, load_global=False)
    values = imp['v']
    ap = imp['ap']
    imp_steps = imp['imp_steps']
    impuls = ap[imp_steps]
    impuls = np.log(impuls)
    # activ = imp['activ']
    # comb_activ = np.hstack([values, cluster_norm])

    # Measure coefficient importances for each cluster of activity

    # Decision ramp model
    act_model = LogisticRegression(penalty='l1', solver='liblinear', C=0.001)
    act_model.fit(cluster_norm, actions)
    ypred = act_model.predict(cluster_norm)
    act_score = f1_score(actions.reshape(-1), ypred)
    act_coef = act_model.coef_[0]

    # Value model
    val_model = Lasso(alpha=0.01)
    val_model.fit(cluster_norm, values)
    ypred = val_model.predict(cluster_norm)
    val_score = r2_score(values, ypred)
    val_coef = val_model.coef_

    # Impulsivity model
    if len(impuls) == 0:
        imp_score = 0
        imp_coef = np.zeros(val_coef.shape)
    else:
        imp_model = LassoCV(cv=10, random_state=0, max_iter=100000, tol=0.001,
                        alphas=np.logspace(-1, 0, 100))
        imp_model.fit(cluster_norm[imp_steps, :], impuls)
        ypred = imp_model.predict(cluster_norm[imp_steps, :])
        imp_score = r2_score(impuls, ypred)
        imp_coef = imp_model.coef_
    coefs = np.vstack([act_coef, val_coef, imp_coef])
    scores = np.array([act_score, val_score, imp_score])
    return coefs, scores


    
def cluster_grads(grads, labels):
    '''Given some computed gradients and cluster labels, sort the gradients
    into their clusters'''
    k = np.max(labels)+1
    clustered_grads = []
    for i in range(k):
        clustered_grads.append(grads[labels == i].reshape(-1))
    return clustered_grads


def test_integrated_gradients(model, obs_rms, res, labels=None, test='value',
                              plot=True, give=False, ax=None, bar=True,
                              multiply_by_inputs=True, fixed_obs=True):
    '''
    Use integrated gradients to see the influence of each hidden node on
    certain output
    
    test: 'value'/'action'
    labels: used for plotting cluster grads after finding influence
    plot: whether to make plot
    give: whether testing an agent that has giverew and needs larger obs
    bar_plot: use bar plots instead of boxplots
    '''
    rnn_hxs = np.vstack(res['rnn_hxs'])
    idxs = np.arange(rnn_hxs.shape[0])
    idxs = np.random.permutation(idxs)
    targets = rnn_hxs[idxs[:50]]
    rand_obs = np.vstack(res['obs'])[idxs[:50]]

    if give:
        base_obs = np.array([0, 1, 0, 0, 0, 0, 0.6, 0, 0]) # defines size 0 balloon with color and prev action nothing
    else:
        base_obs = np.array([0, 1, 0, 0, 0, 0, 0.6, 0]) # defines size 0 balloon with color and prev action nothing
    base_obs = normalize_obs(base_obs, obs_rms)

    # hxs should have shape [1, N, hidden_size]
    # obs with batch_first=False have shape [1, N, obs_size]
    def forward(hxs, base_obs):
        n = hxs.shape[0]
        hxs = hxs.reshape(1, n, 64)
        obs = np.full((1, n, len(base_obs)), base_obs)
        obs = torch.tensor(obs, dtype=torch.float32)
        x = model.base.shared0(obs)
        x = model.base.gru(x, hxs)[0]
        x = x.reshape(n, 64)

        if test == 'value':
            x = model.base.critic1(x)
            val = model.base.critic_head(x)
            return val
        elif test == 'action':
            x = model.base.actor0(x)
            x = model.base.actor1(x)
            dist = model.dist(x)
            return dist.probs

    if test == 'value':
        target_idx = 0
    elif test == 'action':
        target_idx = 1

    grads = []
    cond = IntegratedGradients(forward, multiply_by_inputs=multiply_by_inputs)
    for i, target in enumerate(targets):
        target_hxs = torch.tensor(target.reshape(1, 1, 64))
        if fixed_obs:
            obs = base_obs
        else:
            obs = rand_obs[i]
        grad = cond.attribute(target_hxs, target=target_idx,
                            additional_forward_args=obs)
        grad = grad.squeeze()
        grads.append(grad)

    grads = torch.vstack(grads).numpy()
    grads = np.abs(grads)
    grads = np.mean(grads, axis=0)

    if plot:
        plot_cluster_grads(grads, labels, ax=ax, bar=bar)

    return grads



def compute_rnn_hxs_influences(model, res, nsteps=100, max_unroll=3):
    '''
    Test how much influence each recurrent layer node on the otheres
    Computes whether node i influenced node j in the direction it was
        already moving, and we call this "activating" as opposed to
        "inhibiting" if it exerts opposite influence
        Although, this notion might be incorrect, and might be better
        conceptualize in relation to the direction node j provides information
        in, but overall this is easier to work with for now.
    '''
    n = nsteps

    rnn_hxs = np.vstack(res['rnn_hxs'])
    obs = np.vstack(res['obs'])
    idxs = np.arange(5, rnn_hxs.shape[0]-5)
    idxs = np.random.permutation(idxs)

    all_influences = []

    for j in range(n):
        unroll = np.random.randint(1, max_unroll+1)
        start = idxs[j]
        end = start + unroll
        # prev = rnn_hxs[start-1:end-1]
        # next = rnn_hxs[start+1:end+1]
        prev = rnn_hxs[start-1]
        next = rnn_hxs[start+1]
        cur = rnn_hxs[start]

        diff = cur - prev
        mag = cur.reshape(-1)
        # move = np.sign(next - cur)
        move = np.sign(rnn_hxs[start+unroll-1] - rnn_hxs[start+unroll])

        cur = torch.tensor(cur.reshape(1, 1, 64), requires_grad=True)
        
        o = torch.tensor(obs[start:end], dtype=torch.float32).unsqueeze(1)
        x = model.base.shared0(o)
        x = model.base.gru(x, cur)[1]
        # print(x)

        influences = []
        for i in range(64):
            model.zero_grad()
            x[0, 0, i].backward(retain_graph=True)
            grad = cur.grad.squeeze()
            cur.grad = None
            # influences.append(grad * diff) # absolute influence
            # influences.append(grad * diff * move[i]) # relative influence, activating/inhibiting
            # influences.append(grad * move[i]) # relative influence, activating/inhibiting
            influences.append(grad * mag * move[i]) # relative influence, activating/inhibiting
            
        influences = torch.stack(influences).detach().numpy()
        for i in range(64):
            influences[i, i] = 0
        all_influences.append(influences)
        
    cumu_influences = np.array(all_influences).mean(axis=0)
    return cumu_influences



def compute_gradient_influences(model, res, nsteps=100, max_unroll=3,
                                test='value', pure_grad=True):
    '''
    Compute a signed measure of influence of a node on the agent output
    
    Core idea: 
        - Select N random steps from the agent's history
        - Select a random number M of steps in the future to simulate
        - Measure how the RNN changes based on these M steps thaat the agent experienced
        - Compute the final value or button press probability (depending on if test
          is "value" or "action")
        - Take the backward pass gradient, i.e. measure the influence that node i
          had on the output from M steps in the past
        - If pure_grad is False, also multiply the gradient by the actual activation
          of the node, which may measure how much actual contribution to output it provides
    '''
    n = nsteps

    rnn_hxs = np.vstack(res['rnn_hxs'])
    obs = np.vstack(res['obs'])
    idxs = np.arange(max_unroll, rnn_hxs.shape[0]-max_unroll)
    idxs = np.random.permutation(idxs)

    all_influences = []

    for j in range(n):
        unroll = np.random.randint(1, max_unroll+1)
        start = idxs[j]
        end = start + unroll
        prev = rnn_hxs[start-1]
        next = rnn_hxs[start+1]
        cur = rnn_hxs[start]

        diff = cur - prev
        activation = cur.reshape(-1)
        move = np.sign(rnn_hxs[start+unroll-1] - rnn_hxs[start+unroll])

        cur = torch.tensor(cur.reshape(1, 1, 64), requires_grad=True)
        
        o = torch.tensor(obs[start:end], dtype=torch.float32).unsqueeze(1)
        x = model.base.shared0(o)
        x = model.base.gru(x, cur)[1]

        if test == 'value':
            x = model.base.critic1(x)
            targ = model.base.critic_head(x)
        elif test == 'action':
            x = model.base.actor0(x)
            x = model.base.actor1(x)
            dist = model.dist(x)
            targ = dist.probs[0, 0, 1]

        model.zero_grad()
        targ.backward(retain_graph=True)
        grad = cur.grad.squeeze()
        cur.grad = None
        if pure_grad:
            all_influences.append(grad)
        else:
            all_influences.append(grad * activation) 
        
    cumu_influences = np.array(all_influences).mean(axis=0)
    return cumu_influences



def get_cluster_influences(influences, labels, k=None, ):
    '''
    Organize rnn_hx_influences from compute_rnn_hxs_influences into cluster-wise interactions
    Array will have indices (i, j) meaning the average influence of node in cluster
        j to nodes of cluster i
    '''
    if k is None:
        k = max(labels)+1
    
    cluster_influence = np.zeros((k, k))

    for i in range(k):
        for j in range(k):
            nodes_i = np.where(labels == i)[0]
            nodes_j = np.where(labels == j)[0]
            # Influences from nodes of cluster j to nodes of cluster i
            ij_influence = influences[nodes_i][:, nodes_j]
            if ij_influence.shape[0] > 0 and ij_influence.shape[1] > 0:
                cluster_influence[i, j] = np.nanmean(ij_influence)
            else:
                cluster_influence[i, j] = np.nan
    return cluster_influence

    
def get_val_and_action_scores(model=None, obs_rms=None, res=None, labels=None, 
                              val_grads=None, act_grads=None,
                              give=False,
                              plot=False):
    '''
    For given cluster labels, compute val and action scores based on t-test
    of a cluster's nodes influence on value (action) relative to all nodes influence
    on value (action)
    
    Can calculate by actually getting the gradient here, or just passing
    val grad and act grads that are precomputed
    '''
    if plot:
        fig, axs = pplt.subplots(ncols=2, sharey=False)
        axs.format(ylabel='Gradient contributed')
        axs[0].format(title='Value output contribution')
        axs[1].format(title='Policy output contribution')
    
    ax = axs[0] if plot else None
    k = np.max(labels) + 1
    val_scores = []
    if val_grads is None:
        val_grads = test_integrated_gradients(model, obs_rms, res, labels, test='value',
                                        give=give, bar=True, plot=plot, ax=ax)
    c_val_grads = cluster_grads(val_grads, labels)
    for i in range(k):
        t = ttest_ind(c_val_grads[i], val_grads)
        if t.statistic > 0:
            val_scores.append(1 - t.pvalue)
        else:
            val_scores.append(None)
    ax = axs[1] if plot else None
    pol_scores = []
    if act_grads is None:
        act_grads = test_integrated_gradients(model, obs_rms, res, labels, test='action',
                                        give=give, bar=True, plot=plot, ax=ax)
    c_act_grads = cluster_grads(act_grads, labels)
    for i in range(k):
        t = ttest_ind(c_act_grads[i], act_grads)
        if t.statistic > 0:
            pol_scores.append(1 - t.pvalue)
        else:
            pol_scores.append(None)

    return val_scores, pol_scores