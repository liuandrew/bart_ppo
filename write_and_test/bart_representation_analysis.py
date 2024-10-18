import matplotlib.pyplot as plt
import proplot as pplt
from typing import Union
import itertools
import numpy as np
import pandas as pd
from plotting_utils import rgb_colors
import torch

from sklearn.decomposition import PCA
from scipy.stats import skew, kurtosis
from scipy.optimize import curve_fit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

pplt.rc.update({'font.size': 10})

color_to_idx = {"red": 0, "orange": 1, "yellow": 2,
                        "gray": 4, "purple": 4}
idx_to_color = {0: "red", 1: "orange", 2: "yellow",
                            3: "gray", 4: "purple"}
bart_plot_colors = {0: 'deep red', 
                    1: 'orange', 
                    2: 'goldenrod'}

"""

Misc Helper Functions

General functions for helping to simplify the data coming out of an evaluation
call

"""


def starts_and_ends(res):
    """
    Return the starts and ends of balloons based on 'balloon_step'
    """
    starts = []
    ends = []
    for ep in range(len(res['data']['balloon_step'])):
        end = np.array(res['data']['balloon_step'][ep])
        start = [0]
        for step in end[:-1]:
            start.append(np.argmax(res['actions'][ep].reshape(-1)[step+1:] == 1) + step+1)
        ends.append(end)
        starts.append(start)
    return np.array(starts), np.array(ends)


def unnormalize_obs(obs, obs_rms):
    """
    Convert observation back to unnormalized form
    Size of balloon can be found in obs[5]
    """
    return (obs * np.sqrt(obs_rms.var + 1e-8)) + obs_rms.mean

def get_sizes(res, obs_rms):
    """
    Additional simplification from unnormalize_obs, get sizes directly from 
    observations by unnormalizing
    """
    sizes = []
    for ep in range(len(res['obs'])):
        obs = unnormalize_obs(res['obs'][ep], obs_rms)
        sizes.append(obs[:, 5].round(2))
    return sizes

def comb_pca(res, layer='shared1', n_components=10):
    """
    Perform PCA on a layer of activations
    Combines all episodes in the res call and returns PCAs back in
        per episode format
    """
    activ = res['activations'][layer]
    ep_lens = [len(a) for a in activ]
    all_activ = torch.vstack(activ)
    pca = PCA(n_components=n_components)
    all_pca_a = pca.fit_transform(all_activ)
    pca_as = []

    cur_step = 0
    for ep_len in ep_lens:
        pca_as.append(all_pca_a[cur_step:cur_step+ep_len])
        cur_step += ep_len
        
    return pca_as

"""

Ramp to threshold decision process

Functions designed to analyze how strong of a ramping signal is used in decision
making and in which layer the ramping signal is

"""

def score_logistic_classifiers(res):
    '''
    Score how well decision can be classified by logistic regression
    in each of the layers, with PCAs and with all activations
    '''
    layers = ['shared0', 'shared1', 'actor0', 'actor1', 'critic0', 'critic1']
    f1_scores = np.zeros((6, 11))
    individual_scores = np.zeros((6, 64))
    
    actions  = np.vstack(res['actions'])
    for n, layer in enumerate(layers):
        pca_as = comb_pca(res, layer)
        pcas = np.vstack(pca_as)
        activ = np.vstack(res['activations'][layer])

        lm = LogisticRegression()
        lm.fit(activ, actions)
        ypred = lm.predict(activ)
        f1_scores[n, 0] = f1_score(actions.reshape(-1), ypred)
        
        for j in range(10):
            lm = LogisticRegression()
            lm.fit(pcas[:, j].reshape(-1, 1), actions)
            ypred = lm.predict(pcas[:, j].reshape(-1, 1))
            f1_scores[n, j+1] = f1_score(actions.reshape(-1), ypred)
        
        individual_scores[n] = np.abs(activ).mean(axis=0)*np.abs(lm.coef_).reshape(-1)
    
    return f1_scores, individual_scores


"""

Decision nodes and decision flow

Functions used to help figure out which nodes in the RNN layer are 'decision' nodes
as well as then see how they influence downstream decision processes

"""

def find_decision_nodes(res, model, ep=0):
    """
    Find nodes in the RNN layer that are 'decision' nodes
        by seeing whether they induce a >0.2 decision probability when
        kicked in the direction they would move prior to true decision steps
        on average
    """
    presses = np.argwhere((res['actions'][ep] == 1).reshape(-1)).reshape(-1)
    ends = np.array(res['data']['balloon_step'][ep])
    end_presses = np.intersect1d(presses, ends)
    penult_steps = end_presses - 1

    nsteps = len(penult_steps)
    o = res['obs'][ep][0]
    rnn_hx_mod = torch.tensor(np.zeros((nsteps, 64, 64)), dtype=torch.float)
    obs = torch.tensor(np.zeros((nsteps, 64, o.shape[0])), dtype=torch.float)
    masks = torch.tensor(res['masks'][ep][0])
    probs = np.zeros(nsteps)

    for i, step in enumerate(penult_steps):
        rnn_hx = res['rnn_hxs'][ep][step]
        o = res['obs'][ep][step]
        probs[i] = res['action_probs'][ep][step][1]
        delt_rnn = res['rnn_hxs'][ep][step+1] - res['rnn_hxs'][ep][step]
        delt_rnn = torch.tensor(np.sign(delt_rnn)*2)
        
        
        for j in range(64):
            rnn_hx_mod[i, j] = torch.tensor(rnn_hx)
            obs[i, j] = torch.tensor(o)
            
        for j in range(64):
            rnn_hx_mod[i, j, j] += delt_rnn[j]
    
    obs = obs.reshape(64*nsteps, o.shape[0])
    rnn_hx_mod = rnn_hx_mod.reshape(64*nsteps, 64)
    output = model.act(obs, rnn_hx_mod, masks)
    p = output['probs'][:, 1].detach()
    p = np.array(p).reshape(nsteps, 64)
    scores = (p - probs.reshape(-1, 1)).mean(axis=0)
    decision_nodes = scores > 0.2
    return decision_nodes

def measure_rnn_influence(res, model, ep, step, decision_nodes=None, ap=False,
                          large_kick=False):
    '''Measure how much influence individual nodes or group of nodes have on a certain step
    
    decision_nodes: pass boolean array of size 64 to differentiate decision and non-decision nodes
    ap: if True, compute the influence on action probabilities
    '''
    delt_rnn = res['rnn_hxs'][ep][step+1] - res['rnn_hxs'][ep][step]
    if large_kick:
        delt_rnn = np.sign(delt_rnn) * 2
    rnn_hx = res['rnn_hxs'][ep][step]

    if decision_nodes is not None:
        size = 2
    else:
        size = 64
    
    rnn_hx_mod = torch.tensor(np.full((size, 64), rnn_hx))
    if decision_nodes is not None:
        rnn_hx_mod[0, decision_nodes] += delt_rnn[decision_nodes]
        rnn_hx_mod[1, ~decision_nodes] += delt_rnn[~decision_nodes]

    else:
        for i in range(64):
            rnn_hx_mod[i, i] += delt_rnn[i]
    shared0 = res['activations']['shared0'][ep][step]
    actor0 = res['activations']['actor0'][ep][step]
    obs = res['obs'][ep][step]
    masks = res['masks'][ep][step]

    shared0 = torch.tensor(np.full((size, 64,), shared0))
    actor0 = torch.tensor(np.full((size, 64,), actor0))

    next_rnn_hx = model.base._forward_gru(shared0, rnn_hx_mod, masks)[0]
    # next_rnn_hx = model.base._forward_gru(obs, rnn_hx_mod, masks)[0]
    actor0mod = model.base.actor0(next_rnn_hx)

    delt_actor0 = (actor0mod - actor0).detach()
    
    if ap:
        actor1 = model.base.actor1(actor0mod)
        logits = model.dist(actor1)
        probs = logits.probs
        return delt_actor0, probs

    return delt_actor0
"""

Line fitting

Some functions to easily make best fit lines

"""
def linear_best_fit(x, y):
    """Make a linear line of best fit"""   
    m, b = np.polyfit(x, y, 1)
    ypred = m*x + b
    
    ss_res = np.sum((y - ypred)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r2 = 1 - ss_res / ss_tot

    return (m, b), r2

def p_best_fit(x, y, p):
    """Find a best fit of form f(x) = mx^p + b"""
    x_transformed = np.power(x, p)
    x_design = np.vstack([x_transformed, np.ones(len(x_transformed))]).T
    coefficients, residuals, _, _ = np.linalg.lstsq(x_design, y, rcond=None)
    m, b = coefficients
    ypred = m * x_transformed + b
    
    ss_res = np.sum((y - ypred)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r2 = 1 - ss_res / ss_tot

    return (m, b), r2