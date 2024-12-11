import matplotlib.pyplot as plt
import matplotlib
import proplot as pplt
import numpy as np
import pandas as pd
from torch import nn, optim
import torch
from plotting_utils import rgb_colors, get_color_from_colormap, create_color_map
from bart_representation_analysis import comb_pca, normalize_obs, starts_and_ends

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
from scipy.interpolate import interp1d


'''

Data helpers
Main job here is to split up each response into start and end points
of each balloon, then interpolate or stretch for quantitative comparison

'''


def segment_responses(activ, starts, ends, normalize_by_start=True, stretch=True,
                      target_length=None):
    """
    Segment time series based on stimulus periods and mark modified responses.
    normalize_by_start: if True, normalize such that every response starts at 0
        we might not want this because some responses could likely differ in their start
        based on prior trial outcome
    stretch: if True, stretch the responses to have equal length according to the
        longest response
    """
    responses = []
    for start, end in zip(starts, ends):
        response = activ[start:end]
        if normalize_by_start:
            response = response - response[0]
        responses.append(response)

    if stretch:
        responses = stretch_responses(responses, target_length)
    return responses


def stretch_responses(responses, target_length=None):
    """Stretch each response to a uniform length using interpolation."""
    if target_length is None:
        lens = [len(r) for r in responses]
        target_length = max(lens)
    
    stretched_responses = []
    for response in responses:
        original_length = len(response)
        
        if response.ndim == 2:
            # Expect to receive responses of size [T, N]
            response = response.T
            
        if original_length < 2:
            continue
        x_original = np.linspace(0, 1, original_length)
        x_target = np.linspace(0, 1, target_length)
        interpolator = interp1d(x_original, response, kind='linear', fill_value="extrapolate")
        stretched_responses.append(interpolator(x_target))
    return np.array(stretched_responses)


def prep_response_info(res, ep=0):
    starts, ends = starts_and_ends(res)
    s, e = starts[ep], ends[ep]
    popped = res['data']['popped'][ep]
    # popped_steps = np.array(res['data']['balloon_step'][ep])[popped]
    # return s, e, popped_steps
    
    end_sizes = np.array(res['data']['last_size'][ep])
    return s, e, end_sizes


'''

Pop sensitivity analysis
Analyze which nodes have significant response differences following a pop

'''

def compute_pop_sensitivity_scores_ep(res, ep=0, layer='rnn_hxs', normalize=True):
    '''
    Compute how pop sensitive each node in a layer is
    normalize: will use a specific normalization of standard deviation + 0.5 to account
        for some activities that just don't change much and prevent them from dominating
    '''
    if layer == 'rnn_hxs':
        activ = res['rnn_hxs'][ep]
    else:
        activ = res['activations'][layer][ep]

    starts, ends = starts_and_ends(res)
    s, e = starts[ep], ends[ep]
    popped = res['data']['popped'][ep]
    popped_steps = np.array(res['data']['balloon_step'][ep])[popped]

    pop_sensitivity_scores = []
    for i in range(64):
        a = activ[:, i]
        standard_responses, pop_responses = segment_responses(a, s, e, popped_steps)
        max_length = max(map(len, standard_responses + pop_responses))
        resp1 = stretch_responses(standard_responses, max_length)
        resp2 = stretch_responses(pop_responses, max_length)
        mean_resp1 = np.mean(resp1, axis=0)
        mean_resp2 = np.mean(resp2, axis=0)
        mad = np.mean(np.abs(mean_resp1 - mean_resp2))
        if normalize:
            mad = mad / (mean_resp1.std() + mean_resp2.std() + 0.2)
        pop_sensitivity_scores.append(mad)
        
    return np.array(pop_sensitivity_scores)

    
'''
Stimuli testing
A typical set of predetermined visual stimuli we will test is
- 50 trials of 0.5 size balloon
- 1 trial of a [0.2, 0.8] size balloon, called the stimuli
- 20 trials of 0.5 size balloon
This will let us record how agents respond to the fixed stimuli.
The below functions aid in generating the right responses to directly feed to the model

Perform stimuli experiment does the whole experiment directly
'''

def create_meta_stimuli(obs_rms=None, size=0.5, pop=False, rew=None, 
                        give=False, fix_bug=False, pop_pun=-0.1):
    '''
    Create a single stimuli of observations
    size: final size of balloon
    pop: if False, assume agent had a pop instead of reward
    give: whether agent is given rew info
    fix_bug: whether to fix the give action bug
    rew: if given, give a different reward than the one expected
    pop_pun: how many points lost on a pop
    '''
    obs_size = 9 if give else 8
    n_steps = int(size * 20) + 1
    
    stimuli = np.zeros((n_steps, obs_size), dtype="float32")
    stimuli[:, 1] = 1 # balloon color
    stimuli[0, 7] = 1 # press balloon action
    stimuli[1:, 6] = 1 # wait action
    if not pop and fix_bug:
        # note that there was a bug in env that did not actually give last action
        stimuli[-1, 7] = 1 # press balloon at end
        stimuli[-1, 6] = 0
    if give:
        if pop:
            stimuli[-1, 8] = pop_pun
        elif rew:
            stimuli[-1, 8] = rew
        else:
            stimuli[-1, 8] = size
    
    sizes = np.arange(0.05, size+0.01, 0.05)
    stimuli[:-1, 5] = sizes

    if obs_rms is not None:
        stimuli = normalize_obs(stimuli, obs_rms)
    
    return stimuli
    
    
def stimuli_train(obs_rms=None, sizes=[], pops=[], rews=[], give=False, fix_bug=False,
                  pop_pun=-0.1):
    '''Create a bunch of meta stimuli, returns the stimuli, starts and ends'''
    all_stim = []
    starts = []
    ends = []
    cur = 0
    for i in range(len(sizes)):
        starts.append(cur)
        stim = create_meta_stimuli(obs_rms, size=sizes[i], pop=pops[i], rew=rews[i],
                                   give=give, fix_bug=fix_bug, pop_pun=pop_pun)
        all_stim.append(stim)
        cur += len(stim) - 1
        ends.append(cur)
        cur += 1
    stim = torch.tensor(np.vstack(all_stim), dtype=torch.float32)
    return stim, starts, ends
    
    
def stimuli_responses(model, stim, starts, ends, normalize_by_start=True, stretch=True,
                      target_length=None):
    start_rnn_hxs = model.get_rnn_hxs().unsqueeze(1)
    x = model.base.shared0(stim.unsqueeze(1))
    rnn_hxs = model.base.gru(x, start_rnn_hxs)[0]
    rnn_hxs = rnn_hxs.squeeze(1).detach()
    responses = segment_responses(rnn_hxs, starts, ends, normalize_by_start=normalize_by_start,
                                  stretch=stretch, target_length=target_length)
    return responses

  
  
def perform_stimuli_experiment(model, obs_rms, give=False, by_rew=True, by_size=True, by_pops=False,
                               pop_pun=-0.1, stim_sizes=0, fix_bug=False):
    '''
    Perform the stimuli experiment used in previous calculations of node behavior
    stim_sizes:
        0: 0.2-0.8 size, every 0.1
        1: 0.2-0.8 size, every 0.05
    '''
    if stim_sizes == 0:
        stim_sizes = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    elif stim_sizes == 1:
        stim_sizes = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 
                      0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
    refsize = 0.5
    stim_idx = 50
    warmup_stims = stim_idx+1
    post_stim_count = 20

    if by_pops:
        num_stims = 2
        responses = np.zeros((num_stims, warmup_stims+post_stim_count, 
                              64, 16)) #16 is the number of steps after stretching
        rews = [refsize]*(warmup_stims+post_stim_count)
        sizes = [refsize]*(warmup_stims+post_stim_count)
        pops = [False]*stim_idx + [True] + [False]*post_stim_count
        stim, starts, ends = stimuli_train(obs_rms, sizes=sizes, pops=pops, rews=rews, give=give,
                                           pop_pun=pop_pun, fix_bug=fix_bug)
        responses[1] = stimuli_responses(model, stim, starts, ends, target_length=16, normalize_by_start=False)

        pops = [False]*stim_idx + [False] + [False]*post_stim_count
        stim, starts, ends = stimuli_train(obs_rms, sizes=sizes, pops=pops, rews=rews, give=give,
                                           pop_pun=pop_pun, fix_bug=fix_bug)
        responses[0] = stimuli_responses(model, stim, starts, ends, target_length=16, normalize_by_start=False)
        
    else:
        num_stims = len(stim_sizes)
        responses = np.zeros((num_stims, warmup_stims+post_stim_count, 
                              64, 16)) #16 is the number of steps after stretching
        for j, size in enumerate(stim_sizes):        
            if by_rew:
                rews = [refsize]*stim_idx + [size] + [refsize]*post_stim_count
            else:
                rews = [refsize]*(warmup_stims+post_stim_count)
            
            if by_size:
                sizes = [refsize]*stim_idx + [size] + [refsize]*post_stim_count
            else:
                sizes = [refsize]*(warmup_stims+post_stim_count)
            stim, starts, ends = stimuli_train(obs_rms, sizes=sizes, pops=[False]*100, rews=rews, give=give,
                                               fix_bug=fix_bug, pop_pun=pop_pun)
            responses[j] = stimuli_responses(model, stim, starts, ends, target_length=16, normalize_by_start=False)
            
    return responses, stim_sizes

'''
Measuring node responses
These functions use the earlier described experimental predetermined inputs
to measure how individual nodes respond to a sudden change in stimuli
'''
# Measure resetting rate - compare everything against a reference 0.5 line
def calculate_response_reset_rate(model, obs_rms, diffs=False,
                                  by_rew=True, by_size=True, by_pops=False, pop_pun=-0.1,
                                  give=False, fix_bug=False):
    '''
    Run a stimulation of 50 0.5 balloons, a surprise size s balloon, and then 10 0.5 balloons
    Determine how long it takes for different nodes to settle back to the reference
        0.5 stimulation line
    For now, we will simply compare the absolute difference across the whole stimulation

    diffs: if True, return the percentage differences seen between the reference response
        and the modified response. Otherwise, calculate which nodes are resetting ones and
        which are permanent.
    by_pops: if True, test how the agent responds differently to pops - only 1 experimental
        trial
    by_rew: if True, test different reward sizes on stimulation trials - can be used in
        combo with by_size
    by_size: if True, test different actual balloon inflation sizes on stimulations

    returns:
        mean_recoveries: (64,) size array of mean recovery time to return to reference
            response. An entry of "-1" indicates a "permanent change" induced in the node
            for a majority of tested sizes
    '''
    stim_sizes = [0.2, 0.3, 0.4, 0.6, 0.7, 0.8]
    refsize = 0.5
    stim_idx = 50
    warmup_stims = stim_idx+1
    post_stim_count = 20

    # get a reference response line
    rews = [refsize]*stim_idx + [refsize] + [refsize]*post_stim_count
    sizes = [refsize]*stim_idx + [refsize] + [refsize]*post_stim_count
    stim, starts, ends = stimuli_train(obs_rms, sizes=sizes, pops=[False]*100, rews=rews, give=give, fix_bug=fix_bug)
    refresponses = stimuli_responses(model, stim, starts, ends, target_length=16, normalize_by_start=False)
    refranges = (refresponses.max(axis=2) - refresponses.min(axis=2))[warmup_stims:]

    if by_pops:
        num_stims = 1
        response_diffs = np.zeros((1, post_stim_count, 64))
        rews = [refsize]*(warmup_stims+post_stim_count)
        sizes = [refsize]*(warmup_stims+post_stim_count)
        pops = [False]*stim_idx + [True] + [False]*post_stim_count
        stim, starts, ends = stimuli_train(obs_rms, sizes=sizes, pops=pops, rews=rews, give=give,
                                           pop_pun=pop_pun, fix_bug=fix_bug)
        responses = stimuli_responses(model, stim, starts, ends, target_length=16, normalize_by_start=False)
        for i in range(post_stim_count):
            diff = np.abs(responses[i+warmup_stims] - refresponses[i+warmup_stims]).mean(axis=1)
            response_diffs[0, i] = diff / refranges[i]
        if diffs:
            return response_diffs
        
    else:
        num_stims = len(stim_sizes)
        response_diffs = np.zeros((num_stims, post_stim_count, 64))
        for j, size in enumerate(stim_sizes):        
            if by_rew:
                rews = [refsize]*stim_idx + [size] + [refsize]*post_stim_count
            else:
                rews = [refsize]*(warmup_stims+post_stim_count)
            
            if by_size:
                sizes = [refsize]*stim_idx + [size] + [refsize]*post_stim_count
            else:
                sizes = [refsize]*(warmup_stims+post_stim_count)
            stim, starts, ends = stimuli_train(obs_rms, sizes=sizes, pops=[False]*100, rews=rews, give=give, fix_bug=fix_bug)
            responses = stimuli_responses(model, stim, starts, ends, target_length=16, normalize_by_start=False)
            for i in range(post_stim_count):
                diff = np.abs(responses[i+warmup_stims] - refresponses[i+warmup_stims]).mean(axis=1)
                response_diffs[j, i] = diff / refranges[i]
        if diffs:
            return response_diffs
    
    # find how long it takes for node to reset back to within 5% of reference    
    recoveries = np.zeros((num_stims, 64))
    for n in range(64):
        for i in range(num_stims):
            recov = response_diffs[i, :, n] < 0.05
            if np.all(~recov):
                recoveries[i, n] = post_stim_count
            else:
                recoveries[i, n] = np.argmax(recov)

    # if any responses take longer than 20 trials to reset on a majority of the size
    #  modifications, call these "permanent change" nodes
    # they will be denoted with a -1 in their mean_recovery rate
    mean_recoveries = np.zeros(64)
    perm_changers = (recoveries == 20).sum(axis=0) > num_stims / 2
    mean_recoveries[perm_changers] = -1
    mean_recoveries[~perm_changers] = recoveries.mean(axis=0)[~perm_changers]

    return mean_recoveries



def calculate_node_reversal_response(model, obs_rms, by_rew=True, by_size=True, 
                                     give=False, fix_bug=False):
    '''
    Run a stimulation of 50 0.5 balloons, a surprise size s balloon, and then 10 0.5 balloons
    Determine whether there is a significant cross over in response in small stimulation
        versus large stimulation over time
    '''
    stim_sizes = [0.2, 0.8]
    refsize = 0.5
    stim_idx = 50
    warmup_stims = stim_idx+1
    post_stim_count = 20

    # get a reference response line
    rews = [refsize]*stim_idx + [refsize] + [refsize]*post_stim_count
    sizes = [refsize]*stim_idx + [refsize] + [refsize]*post_stim_count
    stim, starts, ends = stimuli_train(obs_rms, sizes=sizes, pops=[False]*100, rews=rews, give=give,
                                       fix_bug=fix_bug)
    refresponses = stimuli_responses(model, stim, starts, ends, target_length=16, normalize_by_start=False)
    refranges = (refresponses.max(axis=2) - refresponses.min(axis=2))[warmup_stims:]

    num_stims = len(stim_sizes)
    response_diffs = np.zeros((num_stims, post_stim_count, 64))
    for j, size in enumerate(stim_sizes):        
        if by_rew:
            rews = [refsize]*stim_idx + [size] + [refsize]*post_stim_count
        else:
            rews = [refsize]*(warmup_stims+post_stim_count)
        
        if by_size:
            sizes = [refsize]*stim_idx + [size] + [refsize]*post_stim_count
        else:
            sizes = [refsize]*(warmup_stims+post_stim_count)
        stim, starts, ends = stimuli_train(obs_rms, sizes=sizes, pops=[False]*100, rews=rews, give=give,
                                           fix_bug=fix_bug)
        responses = stimuli_responses(model, stim, starts, ends, target_length=16, normalize_by_start=False)
        for i in range(post_stim_count):
            diff = responses[i+warmup_stims, :, -1] - refresponses[i+warmup_stims, :, -1]
            response_diffs[j, i] = diff / refranges[i]

    # return response_diffs
    reverse_times = np.zeros(64)
    cross_req = 0.05
    diff_signs = np.zeros((post_stim_count, 64))
    for n in range(64):
        for step in range(post_stim_count):
            if step == 0:
                diff_signs[step, n] = np.sign(response_diffs[0, step, n] - response_diffs[1, step, n])
            else:
                diff = response_diffs[0, step, n] - response_diffs[1, step, n]
                if (diff_signs[step-1, n] > 0 and diff < -cross_req) or \
                (diff_signs[step-1, n] < 0 and diff > cross_req):
                    diff_signs[step, n] = -diff_signs[step-1, n]
                else:
                    diff_signs[step, n] = diff_signs[step-1, n]
            
    diff_flips = np.diff(diff_signs, axis=0)
    reverses = (diff_flips != 0).any(axis=0)
    reverse_times[~reverses] = -1
    reverse_times[reverses] = np.argmax(diff_flips, axis=0)[reverses] + 1
    
    return reverse_times

    
    
def linear_vs_quadratic(data, x=None, ret_models=False, ret_mse=False, quad_penalty=1e-6):
    """
    Determines whether a linear or quadratic line of best fit is better for a 1D array of data.
    
    Parameters:
    - data (1D array-like): The data to analyze.
    - x: x values corresponding to data y values if wanted
    - quad_penalty: increase penalty to quad complexity
    
    Returns:
    - result (int): 1 if the linear fit is better, 2 if the quadratic fit is better.
    - linear_params (tuple): Coefficients of the best-fit linear equation.
    - quadratic_params (tuple): Coefficients of the best-fit quadratic equation.
    - h (float): x location of quadratic fit vertex
    """
    # Create an x-axis corresponding to the indices of the data
    if x is None:
        x = np.arange(len(data))
    
    linear_mse = float('inf')
    linear_h = None
    best_linear_models = None
    
    # Iterate over all possible breakpoints (except the first and last points)
    for breakpoint in range(2, len(data) - 2):
        x1, y1 = x[:breakpoint], data[:breakpoint]
        x2, y2 = x[breakpoint:], data[breakpoint:]
        linear_coeffs1 = np.polyfit(x1, y1, 1)
        linear_fit1 = np.polyval(linear_coeffs1, x1)
        linear_mse1 = mean_squared_error(y1, linear_fit1) * len(y1)
        linear_coeffs2 = np.polyfit(x2, y2, 1)
        linear_fit2 = np.polyval(linear_coeffs2, x2)
        linear_mse2 = mean_squared_error(y2, linear_fit2) * len(y2)
        
        mse = linear_mse1 * len(y1) + linear_mse2 * len(y2)
        if mse < linear_mse:
            linear_mse = mse
            linear_h = x[breakpoint]
            best_linear_models = (linear_coeffs1, linear_coeffs2)
    
    # Perform quadratic fit (degree 2 polynomial)
    quadratic_coeffs = np.polyfit(x, data, 2)
    quadratic_fit = np.polyval(quadratic_coeffs, x)
    quadratic_mse = mean_squared_error(data, quadratic_fit) * len(data) + quad_penalty
    
    # Get vertex x location
    a, b, c = quadratic_coeffs
    quadratic_h = -b / (2 * a)
    
    # Compare residuals to determine which fit is better
    # if linear_rss > quadratic_rss and h < 0.7 and h > 0.3:
    #     result = 2
    # else:
    #     result = 1

    if linear_mse > quadratic_mse and quadratic_h < 0.8 and quadratic_h > 0.2:
        result = 2
    else:
        result = 1
    
    if ret_models:
        return result, best_linear_models, tuple(quadratic_coeffs), linear_h, quadratic_h
    if ret_mse:
        return result, linear_mse, quadratic_mse, linear_h, quadratic_h
        
    return result, linear_h, quadratic_h


def calculate_node_stimuli_response_change(model, obs_rms, give=False, 
                                           by_rew=True, by_size=True, fix_bug=False):
    '''
    Test model with a suite of response experiments (50 0.5 size, 1 variable size, 2 more 0.5 size)
    to determine whether each node has a piecewise linear, or quadratic response
    as well as where its "reversal" point is and whether it has a bias of sensitivity towards
    one side of the reversal
    
    returns
        - response_types: 
            -1: no strong persistent effect 2 trials after stim
            1: piecewise linear response wrt stim size
            2: quadratic response wrt stim size
        - reversals: reversal point or center wrt stim size
        - lr_sens_bias: whether node has more sensitivity on left (small balloons) or right
    '''
    responses, stim_sizes = perform_stimuli_experiment(model, obs_rms, stim_sizes=1,
                                                       give=give, by_rew=by_rew, by_size=by_size,
                                                       fix_bug=fix_bug)
    stim_idx = 50
    step = 2 
    lr_bias_lim = 5 # required range ratio difference to count as biased
    lin_quads = np.zeros(64)
    reversals = np.zeros(64)
    end_rel_ranges = np.zeros(64)
    ranges = np.zeros(64)
    lr_sens_bias = np.zeros(64) # 0: no bias, 1: right bias, -1: left bias
    for n in range(64):
        end_resp = responses[:, stim_idx+step, n, -1]
        # lq, _, quad_model, linear_h, quadratic_h = linear_vs_quadratic(end_resp, stim_sizes, ret_models=True)
        lq, linear_h, quadratic_h = linear_vs_quadratic(end_resp, stim_sizes)
        
        lin_quads[n] = lq
        if lq == 1:
            reversals[n] = linear_h
        else:
            reversals[n] = quadratic_h

        end_resp_range = end_resp.max() - end_resp.min()
        resp_range = responses[:, stim_idx+step, n, :].max() - responses[:, stim_idx+step, n, :].min()
        rel_range = end_resp_range / resp_range
        ranges[n] = resp_range
        end_rel_ranges[n] = rel_range 
        
        rev_idx = np.clip(np.argmin(np.abs(np.array(stim_sizes) - reversals[n])), 2, 11)
        left_rng = end_resp[:rev_idx].max() - end_resp[:rev_idx].min()
        right_rng = end_resp[rev_idx:].max() - end_resp[rev_idx:].min()
        if left_rng < right_rng and (right_rng/left_rng > lr_bias_lim):
            lr_sens_bias[n] = 1
        elif left_rng > right_rng and (left_rng/right_rng > lr_bias_lim):
            lr_sens_bias[n] = -1

    # generate summary return data
    # -1 no persistent response change after 2 steps
    # 1: piecewise linear fit
    # 2: quadratic fit
    response_types = np.full(64, -1) 
    response_types[(end_rel_ranges > 0.2) & (lin_quads == 2)] = 2
    response_types[(end_rel_ranges > 0.2) & (lin_quads == 1)] = 1
    
    return response_types, reversals, lr_sens_bias



def calculate_all_single_node_characteristics(model, obs_rms, give=False, fix_bug=False,
                                              recovery_diffs=False, pop_pun=-0.1):
    '''
    Perform each of the single node stimuli experiments and return the results ready
    for use by plotting function
    recovery_diffs: if True, mean_recoveries and pop_recoveries will return actual diffs
        of shape (k, 20, 64) where k is the number of stimuli (6 for size test, 1 for pop test)
    '''
    response_types, turning_points, lr_sens_bias = calculate_node_stimuli_response_change(model, obs_rms,
                                                                                fix_bug=fix_bug, give=give)
    mean_recoveries = calculate_response_reset_rate(model, obs_rms, fix_bug=fix_bug,
                                                    give=give, diffs=recovery_diffs)
    pop_recoveries = calculate_response_reset_rate(model, obs_rms, fix_bug=fix_bug, by_pops=True,
                                                    give=give, diffs=recovery_diffs, pop_pun=pop_pun)
    reversals = calculate_node_reversal_response(model, obs_rms, fix_bug=fix_bug,
                                                    give=give)
    
    return response_types, turning_points, lr_sens_bias, \
        mean_recoveries, pop_recoveries, reversals



'''
Plotting functions

'''
def plot_stretched_responses(standard_responses, pop_responses, ax=None):
    """Plot average responses for standard and modified stimuli after stretching."""
    # Determine the maximum length to align all responses
    if ax is None:
        fig, ax = pplt.subplots()

    lines = []
    
    max_length = max(map(len, standard_responses + pop_responses))
    resp1 = stretch_responses(standard_responses, max_length)
    resp2 = stretch_responses(pop_responses, max_length)
    mean_resp1 = np.mean(resp1, axis=0)
    mean_resp2 = np.mean(resp2, axis=0)

    lines.append(ax.plot(mean_resp1, label='Standard'))
    lines.append(ax.plot(mean_resp2, linestyle='--', label='Post pop'))
    
    mad = np.mean(np.abs(mean_resp1 - mean_resp2))
    norm_factor = np.mean(np.abs(mean_resp1))
    # nmad = mad / norm_factor
    # ax.format(title=f'{mad:.3f}')
    nmad = mad / (mean_resp1.std() + mean_resp2.std() + 0.2)
    ax.format(title=f'{nmad:.3f}')
    
    return lines
    

def plot_fixed_stimulus_experiment(responses, n, stim_sizes=None, show_end_step=2, reversals=None,
                                   plot_trajs=5):
    '''
    To get responses and stim_sizes run
    responses, stim_sizes = perform_stimuli_experiment(model, obs_rms, stim_sizes=1)

    Plot the responses of node n before and after the stimulus
    '''
    stim_idx = 50
    start_idx = 49

    if stim_sizes is None:
        stim_sizes = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
        # stim_sizes = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    
    
    fig, ax = pplt.subplots(ncols=plot_trajs, figwidth=7)
    
    for i in range(plot_trajs):
        for j, size in enumerate(stim_sizes):
            c = get_color_from_colormap(size, 0.2, 0.8, 'vlag')
            ax[i].plot(responses[j, start_idx+i, n, :], c=c)
    ax[-1].colorbar(create_color_map(0.2, 0.8, 'vlag'), label='stim size')
    toplabels = ['50: Reference \n 0.5 trial', '51: Var size \n stim trial']
    for i in range(2, plot_trajs):
        toplabels.append(f'{50+i}: 0.5 \ntrial')
    ax.format(toplabels=toplabels,
            xlabel='time step', ylabel='node activity')

    fig, ax = pplt.subplots(figwidth=3)
    ax.scatter(stim_sizes, responses[:, stim_idx+show_end_step, n, -1], c=stim_sizes, cmap='vlag')
    ax.format(xlabel='Stim size', bel='End response after 2 0.5 trials',
            title='Final step of trial 53 (2 after stim)')

    # print('Relative persistent effect size', end_rel_ranges[n])
    end_resp = responses[:, stim_idx+show_end_step, n, -1]
    
    if reversals is not None:
        rev_idx = np.clip(np.argmin(np.abs(np.array(stim_sizes) - reversals[n])), 2, 11)
        left_rng = end_resp[:rev_idx].max() - end_resp[:rev_idx].min()
        right_rng = end_resp[rev_idx:].max() - end_resp[rev_idx:].min()
        if left_rng < right_rng:
            print('right sens bias', right_rng / left_rng)
        else:
            print('left sens bias', left_rng / right_rng)

def plot_pop_stimulus_experiment(responses, n):
    '''
    To get responses and stim_sizes run
    responses, stim_sizes = perform_stimuli_experiment(model, obs_rms, by_pops=True, by_size=False)

    Plot the responses of node n before and after the stimulus
    '''
    stim_idx = 50
    start_idx = 49
    plot_trajs = 5
    
    fig, ax = pplt.subplots(ncols=plot_trajs, figwidth=7)
    
    pop_labels = ['No pop', 'Pop']
    for i in range(plot_trajs):
        lines = []
        for j in range(2):
            lines.append(ax[i].plot(responses[j, start_idx+i, n, :], 
                                    c=rgb_colors[j], label=pop_labels[j],
                                    alpha=0.7))
    fig.legend(lines, loc='b')
    ax.format(toplabels=['50: Reference \n 0.5 trial', '51: 0.5 -> pop \nstrial', '52: 0.5 trial', '53: 0.5 trial', '54: 0.5 trial'],
            xlabel='time step', ylabel='node activity')


def plot_node_stimuli_summary(response_types, turning_popints, lr_sens_bias,
                              mean_recoveries, pop_recoveries, reversals):
    '''Plot summary results from a calculate_node_stimuli_response_change function call'''
    fig, ax = pplt.subplots(nrows=2, ncols=3, sharey=False, sharex=False)
    ax[0].bar(0, (response_types == 1).sum()/64)
    ax[0].bar(1, (response_types == 2).sum()/64)
    ax[0].format(xlocator=range(2), xformatter=['Linear response', 'Quadratic response'],
                ylabel='Frequency',
                ylim=[0, 1.05], title='Response types wrt stim size')

    ax[1].boxplot(turning_popints[response_types != -1])
    ax[1].format(title='Turning points', ylabel='Stim size', xformatter=[''])

    ax[2].bar(0, (lr_sens_bias == -1).sum()/64)
    ax[2].bar(1, (lr_sens_bias == 1).sum()/64)
    ax[2].format(xlocator=range(2), xformatter=['Left', 'Right'], xlabel='Bias',
                ylabel='Frequency',
                ylim=[0, 1.05], title='LR sensitivity bias')

    fast_recoveries = ((mean_recoveries >= 0) & (mean_recoveries <= 1)).sum()/64
    medium_recoveries = ((mean_recoveries > 1) & (mean_recoveries <= 3)).sum()/64
    long_recoveries = ((mean_recoveries > 3) | (mean_recoveries == -1)).sum()/64
    for i, rec in enumerate([fast_recoveries, medium_recoveries, long_recoveries]):
        ax[3].bar(i, rec)
    ax[3].format(xlocator=range(3), xformatter=['Fast (0-1)', 'Medium (1-3)', 'Slow (3+)'], ylabel='Frequency',
    ylim=[0, 1.05], xlabel='Recovery speed (num balloons)', title='Recovery to stim size change')
    
    fast_recoveries = ((pop_recoveries >= 0) & (pop_recoveries <= 1)).sum()/64
    medium_recoveries = ((pop_recoveries > 1) & (pop_recoveries <= 3)).sum()/64
    long_recoveries = ((pop_recoveries > 3) | (pop_recoveries == -1)).sum()/64
    for i, rec in enumerate([fast_recoveries, medium_recoveries, long_recoveries]):
        ax[4].bar(i, rec)
    ax[4].format(xlocator=range(3), xformatter=['Fast (0-1)', 'Medium (1-3)', 'Slow (3+)'], ylabel='Frequency',
    ylim=[0, 1.05], xlabel='Recovery speed (num balloons)', title='Recovery to pop')

    ax[5].bar(0, (reversals < 0).sum()/64)
    ax[5].bar(1, (reversals >= 0).sum()/64)
    ax[5].format(xlocator=range(2), xformatter=['No Reverse', 'Reverses'], ylabel='Frequency',
    ylim=[0, 1.05], title='Reversals across trials')
    
    ax.format(leftlabels=['T+2 behaviors',  'Long-term behaviors'])
    return ax


def plot_stimuli_reversal(responses, n):
    '''
    Plot the largest and smallest stimuli final step responses over balloon trials to see how they reverse
    '''
    fig, ax = pplt.subplots()
    ax.plot(np.arange(40, 71), responses[0, 40:, n, -1], c=get_color_from_colormap(0.2, 0.2, 0.8, 'vlag'))
    ax.plot(np.arange(40, 71), responses[-1, 40:, n, -1], c=get_color_from_colormap(0.8, 0.2, 0.8, 'vlag'))
    ax.format(xlabel='Trial num', ylabel='End step node activity')