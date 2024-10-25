import matplotlib.pyplot as plt
import matplotlib
import proplot as pplt
import numpy as np
import pandas as pd
from torch import nn, optim
import torch
from plotting_utils import rgb_colors, get_color_from_colormap
from bart_representation_analysis import comb_pca

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, Lasso, LassoCV
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from sklearn.metrics import silhouette_score, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


pplt.rc.update({'font.size': 10})

color_to_idx = {"red": 0, "orange": 1, "yellow": 2,
                        "gray": 4, "purple": 4}
idx_to_color = {0: "red", 1: "orange", 2: "yellow",
                            3: "gray", 4: "purple"}
bart_plot_colors = {0: 'deep red', 
                    1: 'orange', 
                    2: 'goldenrod'}


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
        activ = np.vstack([res['activations'][layer][e] for e in ep])
    else:
        v = np.vstack(res['values'])
        ap = np.vstack(res['action_probs'])[:, 1]
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


def split_by_ep(res, data):
    '''After combining steps, return back to episode based split'''
    ep_lens = [len(o) for o in res['obs']]

    ep_data = []
    cur_step = 0
    for ep_len in ep_lens:
        ep_data.append(data[cur_step:cur_step+ep_len])
        cur_step += ep_len
    return ep_data
    
'''

Compression methods

Methods for compressing activation data. We have 3 main methods:
1. PCA
2. Kmeans
3. Bottleneck networks

Note that comb_pca in the bart_representation_analysis.py file also falls
into this area

'''

def get_cluster_activations(activ, kmeans=None, k=5):
    '''
    Use kmeans on hidden state data to cluster the data after scaling
    km: if passed, use an already fit KMeans model, rather than fitting a new one
    '''
    data = activ.T  # change to [64, T]
    scaler = TimeSeriesScalerMeanVariance()
    data_normalized = scaler.fit_transform(data[:, :, np.newaxis])  # Shape becomes [64, T, 1]
    data_normalized = data_normalized.squeeze()  # Back to shape [64, T]
    if kmeans is None:
        kmeans = KMeans(n_clusters=k, random_state=0)
    labels = kmeans.fit_predict(data_normalized)
    
    cluster_data = [data_normalized[labels == i] for i in range(kmeans.n_clusters)]
    cluster_activations = np.vstack([c.mean(axis=0) for c in cluster_data]).T
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
        cluster_activations, _, _ = get_cluster_activations(activ, kmeans)
        lr = LinearRegression()
        lr.fit(cluster_activations, ap)
        ypred = lr.predict(cluster_activations)
        r2_scores.append(r2_score(ap, ypred))
        
    r2_scores = np.array(r2_scores)
    silhouette_scores = np.array(silhouette_scores)
    
    # Ensure that enough info is kept in clusters to keep r2 prediction high
    if require_ap_explained:
        min_r2_idx = np.argmax(r2_scores > 0.8*best_r2)
        best_k = np.argmax(silhouette_scores[min_r2_idx:]) + min_r2_idx + 2
    else:
        best_k = np.argmax(silhouette_scores) + 2
        
    return best_k



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


def visualize_cluster_activations(res, klabels, ep=8, layer='shared1'):
    '''
    Visualize the compositionn of each cluster across a single episode
    '''
    k = np.max(klabels) + 1
    if k > 20:
        print('k > 20, not creating plot')
        return None

    activ = res['activations'][layer][ep]
    cluster_data = [activ[:, klabels == i] for i in range(k)]
    fig, axs = pplt.subplots(nrows=k, sharex=True, sharey=True, 
                             figwidth=5, refaspect=4)
        
    for i, ax in enumerate(axs):
        ax.format(title=f'Cluster {i}')
        
        for j in range(cluster_data[i].shape[1]):
            ax.plot(cluster_data[i][:, j], alpha=0.5)
        ax.plot(cluster_data[i].mean(axis=1), c='black')
    fig.format(suptitle='Time Series Clusters Visualization', xlabel='Time', ylabel='Normalized Value')


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
        
        
            
def visualize_episode_values(res, color_by_mu=False, minimal_size_coverage=False):
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
        
    fig, ax = pplt.subplots()
    for j, ep in enumerate(ep_idxs):
        s = bsizes[j]
        c = get_color_from_colormap(s, vmin, vmax, to_hex=False)
        ax.plot(list(pd.Series(res['values'][ep].reshape(-1)).ewm(alpha=0.1).mean()), 
                c=c, alpha=0.5)
    ax.format(xlabel='time step', ylabel='Value prediction')
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
    cbar = fig.colorbar(sm)
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