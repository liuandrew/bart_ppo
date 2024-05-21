import gym
import torch
import gym_nav
import numpy as np

import sys
sys.path.append('../')
from evaluation import *
from model_evaluation import *
from trajectories import *
from shortcut_analysis import *
from explore_analysis import *
from population_representations import *
from representation_analysis import *

from tqdm import tqdm
from umap import UMAP
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cdist

from pathlib import Path

pplt.rc.update({'font.size': 10})




def find_consecutive(boolean_array, consec_needed=2):
    '''Find the index of the first time that boolean_array is True consec_needed
    times in a row. E.g., for finding first time SUR >= 0.3 for 2 checkpoints'''
    consec = 0
    for j in range(len(boolean_array)):
        if boolean_array[j]:
            consec += 1
        else:
            consec = 0
        if consec == consec_needed:
            return j
    return -1
    

def spatial_heatmap(pos, y, extent=(5, 295), num_grid=30, sigma=10):
    '''
    Calculate spatial heatmaps for given position and y values
    pos: [N, 2] points
    y: [N, d] corresponding weights for each position point. For example, if passing
        64 dimensional activations, d=64
    extent: x, y extents to take spatial averaging of
    num_grid: how many grid points to consider
    sigma: smoothing factor
    '''
    y = np.array(y)
    
    if len(y.shape) > 2:
        raise Exception('y array must be at most 2 dimensions')
    
    
    grid = np.linspace(5, 295, num_grid)
    xs, ys = np.meshgrid(grid, grid)
    ys = ys[::-1] # flip ys so that when calling ax.imshow we get the expected orientation
    
    # Calculate weighted distances from each grid point to each pos position
    x_dists = pos[:, 0] - xs.reshape(-1, 1)
    y_dists = pos[:, 1] - ys.reshape(-1, 1)
    
    x_dists = x_dists**2
    y_dists = y_dists**2
    dists = x_dists + y_dists
    
    # Gaussian kernel distances to each grid point
    g = np.exp(-dists / (2*sigma**2))
    
    if len(y.shape) == 1:
        weighted_smooth = (g * y)
        heatmaps = np.sum(weighted_smooth, axis=1) / np.sum(g, axis=1)
        heatmaps = heatmaps.reshape(num_grid, num_grid)
    elif len(y.shape) == 2:
        weighted_smooth = (g[:, :, np.newaxis] * y)
        heatmaps = np.sum(weighted_smooth, axis=1) / np.sum(g, axis=1)[:, np.newaxis]
        heatmaps = heatmaps.reshape(num_grid, num_grid, y.shape[1])
        heatmaps = heatmaps.transpose([2, 0, 1])
    
    return heatmaps


def circular_gaussian_filter_fixed_angles(angles, weights, sigma=3, num_angles=100):
    '''Gaussian filter across angles'''
    # Sort angles and corresponding weights
    sorted_indices = np.argsort(angles)
    sorted_angles = angles[sorted_indices]
    sorted_weights = weights[sorted_indices]

    # Wrap the angles and weights around the circle to handle boundary cases
    wrapped_angles = np.concatenate((wrap_angle(sorted_angles - 2 * np.pi), sorted_angles, wrap_angle(sorted_angles + 2 * np.pi)))
    wrapped_weights = np.concatenate((sorted_weights, sorted_weights, sorted_weights))

    # Create uniformly spaced angles
    uniform_angles = np.linspace(-np.pi, np.pi, num_angles, endpoint=False)
    
    # Apply Gaussian filter
    filtered_weights = gaussian_filter1d(wrapped_weights, sigma, axis=0)

    # Extract the filtered weights corresponding to the original angles
    filtered_weights = filtered_weights[len(angles): 2 * len(angles)]


    if len(weights.shape) > 1:
        uniform_filtered_weights = np.zeros((weights.shape[1], num_angles))
        for i in range(weights.shape[1]):
            uniform_filtered_weights[i] = np.interp(uniform_angles, sorted_angles, filtered_weights[:, i])
            
    else:
        # Interpolate the filtered weights for the uniformly spaced angles
        uniform_filtered_weights = np.interp(uniform_angles, sorted_angles, filtered_weights)

    return uniform_angles, uniform_filtered_weights



def weighted_circular_statistics(weights, num_angles=100):
    '''Computed weighted circular statistics to compute coherence of angle activations'''
    angles = get_uniform_angles(num_angles)
    
    # Convert x, y values to angles
    # angles = np.arctan2(y_values, x_values)
    if np.sum(weights) == 0:
        return 0, 0

    # Calculate the weighted mean direction
    
    mean_sin = np.average(np.sin(angles), weights=weights)
    mean_cos = np.average(np.cos(angles), weights=weights)
    mean_direction = np.arctan2(mean_sin, mean_cos)

    # Calculate the weighted mean resultant length
    mean_resultant_length = np.sqrt(mean_sin**2 + mean_cos**2)

    return mean_direction, mean_resultant_length


def get_uniform_angles(num_angles=100):
    '''Get angles from -pi to pi to plot with'''
    uniform_angles = np.linspace(-np.pi, np.pi, num_angles, endpoint=False)
    return uniform_angles

def angle_hm_plot(weights, num_angles=100, ax=None):
    '''Make a plot of angle heatmap'''
    uniform_angles = get_uniform_angles(num_angles)
    
    if ax is None:
        fig, ax = pplt.subplots()
        
    x, y = np.cos(uniform_angles), np.sin(uniform_angles)
    return ax.scatter(x, y, c=weights)

    
def compute_cumulative_variance(hms):
    '''
    Given a set of heatmaps (spatial or angular), compute the descending
    explained variances of a SVD (same as explained_variance_ratios_ from a PCA).
    
    hms: array of shape [n, grid_size] where n is the number of nodes, grid_size is the
        spatial or angular grid
    
        Note if hms is of shape [n, grid, grid] (like in spatial heatmaps), we'll reshape
        it to [n, grid*grid]
    '''
    if len(hms.shape) > 2:
        hms = hms.reshape(hms.shape[0], -1)
    
    U, S, Vt = np.linalg.svd(hms)
    
    variance_ratios = S**2 / np.sum(S**2)
    cumulative_variance = np.cumsum(variance_ratios)
    
    return cumulative_variance


def flatten_hm(hm, threshold=0.4):
    '''Turn a heatmap or set of heatmaps into sharpened versions
    that only show high/close to zero/low activations
    
    Note: returns a copy rather than changing inplace'''
    
    hm = hm.copy()
    hm[hm < -threshold] = -1
    hm[hm > threshold] = 1
    hm[(hm >= -threshold) & (hm <= threshold)] = 0
    
    return hm

'''Local variability measure'''

def get_sliding_window_vector(window_size=5):
    '''
    Get a vector representing the indexes to perform vectorized sliding window computation
    If doing a bunch of calls to local_variability, this decreases computational overhead to pass in
    '''
    heatmap_size = 30
    edge_start = (window_size - 1) // 2
    
    window_base = np.meshgrid(np.arange(edge_start, heatmap_size-edge_start), np.arange(edge_start, heatmap_size-edge_start))
    window_shape = np.meshgrid(np.arange(-edge_start, edge_start+1), np.arange(-edge_start, edge_start+1))

    basex, basey = window_base[0].reshape(-1, 1), window_base[1].reshape(-1, 1)
    shapex, shapey = window_shape[0].reshape(-1), window_shape[1].reshape(-1)

    windowx = basex + shapex
    windowy = basey + shapey
    return [windowx, windowy]


def local_variability(heatmap, window_size=5, window=None):
    '''
    Compute the mean local standard deviation across sliding windows of heatmaps
    heatmaps: should be of shape (N, 30, 30) where N is the number of heatmaps
    window_size: should be an odd number
    window: pass in [windowx, windowy] from get_sliding_window_vector
        to decrease computation time
    '''
    if window is None:
        window = get_sliding_window_vector(window_size)
    windowx, windowy = window
    
    windowed_hms = heatmap[:np.newaxis, windowx, windowy]
    stds = np.mean(np.std(windowed_hms, axis=2), axis=1)
    return stds



'''
KMeans functions
'''

def get_best_negative_cluster_pairs(centers, targets=None):
    '''
    Given a set of cluster centers of shape [N, 900] or [N, 30, 30],
    get the best pairings of clusters with their negatives
    targets: if wanting pairs to be ordered in a certain way, pass a [N//2, 900] collection
        of heatmaps to try to prioritize by
    '''
    if len(centers.shape) == 2:
        centers = centers.reshape(-1, 900)
    dist = cdist(centers, -centers)
    n = centers.shape[0]
    used = []
    pairs = []
    idxs = np.argsort(dist)
    for i in range(n):
        if i not in used:
            for j in range(n):
                closest = idxs[i, j]
                if closest not in used:
                    used.append(i)
                    used.append(closest)
                    pairs.append([i, closest])
                    break
                
    if targets is not None:
        new_pairs = []
        used = []
        dist = cdist(targets, centers)
        for i in range(n//2):
            for j in range(n//2):
                # print(j)
                closest = np.argsort(dist[i])[j]
                if closest not in used:
                    for k in range(n//2):
                        if closest in pairs[k]:
                            pair = pairs[k]
                            pair_idx = pair.index(closest)
                    used.append(pair[0])
                    used.append(pair[1])
                    new_pairs.append([pair[pair_idx], pair[1-pair_idx]])
                    break
        pairs = new_pairs
        
    return pairs


def kmean_label_pair_mapping(pairs, labels, comb_pairs=False):
    '''Take a list of KMeans label predictions and map them to the matching
    pair of negative clusters. Assumes pairs are given as positive/negative pair,
    although this is quite arbitrary'''
    
    # turn pairs into dict objects
    positive_dict = {}
    negative_dict = {}
    comb_dict = {}
    for i, pair in enumerate(pairs):
        positive_dict[pair[0]] = i
        negative_dict[pair[1]] = i
        
        comb_dict[pair[0]] = i
        comb_dict[pair[1]] = i
        
    if comb_pairs:
        remapped_labels = []
        for label in labels:
            remapped_labels.append(comb_dict[label])
        return remapped_labels
    
    else:
        positive_labels = []
        negative_labels = []
        for label in labels:
            if label in positive_dict:
                positive_labels.append(positive_dict[label])
            else:
                negative_labels.append(negative_dict[label])
        return positive_labels, negative_labels    
        
        
def count_kmean_freqs(labels, k=8):
    '''Given labels, turn them into frequency counts'''
    idxs, label_counts = np.unique(labels, return_counts=True)
    counts = np.zeros(k, dtype='int')
    for i in range(len(idxs)):
        idx = idxs[i]
        counts[i] = label_counts[i]
    return counts


def hm_count_kmean_freqs(hm, k, comb_pairs=False, pairs=None, targets=None):
    '''
    Given a heatmaps array of size [N, 30, 30] or [N, 900] where N is the number
    of heatmaps (nodes), count cluster label frequencies
    
    Uses 'k' from global, which is assumed to be the KMeans cluster model, and
        'pairs' from global, which is assumed to be 
    
    
    comb_pairs: whether to combine clusters that are positive/negatives of each other
        If False, returns pos_counts and neg_counts
        If True, returns combined counts
        
    pairs: if manually selecting pairs of clusters, pass in a list of 2-item lists
    targets: if wanting pairs to be ordered in a certain way, pass a [N, 900] collection
        of heatmaps to try to prioritize by
    '''
    if hm.shape[-1] == 30:
        hm = hm.reshape(-1, 900)
    
    labels = k.predict(hm)
    if pairs is None:
        pairs = get_best_negative_cluster_pairs(k.cluster_centers_, targets)
    if comb_pairs:
        re_labels = kmean_label_pair_mapping(pairs, labels, comb_pairs)
        counts = count_kmean_freqs(re_labels, k=k.cluster_centers_.shape[0]//2)
        return counts
    
    else:
        k_size = k.cluster_centers_.shape[0]//2
        pos_labels, neg_labels = kmean_label_pair_mapping(pairs, labels, comb_pairs)
        pos_counts, neg_counts = count_kmean_freqs(pos_labels, k_size), \
                                 count_kmean_freqs(neg_labels, k_size)
        return pos_counts, neg_counts
            