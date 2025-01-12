import matplotlib.pyplot as plt
import proplot as pplt
from typing import Union
import itertools
import numpy as np
import pandas as pd
from plotting_utils import rgb_colors
import torch
import pickle
from model_evaluation import (
    forced_action_evaluate_multi,
    meta_bart_multi_callback,
    reshape_parallel_evalu_res,
)

from sklearn.decomposition import PCA
from scipy.stats import skew, kurtosis
from scipy.optimize import curve_fit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from functools import partial
from diptest import diptest

pplt.rc.update({'font.size': 10})

color_to_idx = {"red": 0, "orange": 1, "yellow": 2,
                        "gray": 4, "purple": 4}
idx_to_color = {0: "red", 1: "orange", 2: "yellow",
                            3: "gray", 4: "purple"}
bart_plot_colors = {0: 'deep red', 
                    1: 'orange', 
                    2: 'goldenrod'}

"""

MetaBart Evaluation functions


"""

size = np.arange(0.2, 1.01, 0.05)
fixprev_env_kwargs = [{'meta_setup': 1, 'colors_used': 1, 
                            'max_steps': 2500, 'num_balloons': 50,
                            'inflate_noise': 0, 'fix_prev_action_bug': True,
                            'fix_sizes': [0, s, 0]} for s in size]
give_env_kwargs = [{'meta_setup': 1, 'colors_used': 1, 
                            'max_steps': 2500, 'num_balloons': 50,
                            'inflate_noise': 0, 'give_rew': True,
                            'fix_sizes': [0, s, 0]} for s in size]

fixprev_evalu_ = partial(forced_action_evaluate_multi, data_callback=meta_bart_multi_callback,
                env_name="BartMetaEnv", num_episodes=1, 
                env_kwargs=fixprev_env_kwargs, 
                num_processes=17,
                seed=1,
                deterministic=False,
                with_activations=True)
give_evalu_ = partial(forced_action_evaluate_multi, data_callback=meta_bart_multi_callback,
                env_name="BartMetaEnv", num_episodes=1, 
                env_kwargs=give_env_kwargs, 
                num_processes=17,
                seed=1,
                deterministic=False,
                with_activations=True)


def evalu(model, obs_rms, give_rew=False):
    if give_rew:
        res = give_evalu_(model, obs_rms)
    else:
        res = fixprev_evalu_(model, obs_rms)
    res = reshape_parallel_evalu_res(res, meta_balloons=50)
    return res


def metabart_model_load(idx=None, h=None, i=None, j=None, k=None, l=None):
    if idx is not None:
        h, i, j, k, l = idx
    give_rew = ['giverew_', 'fixprev_']
    postfixes = ['pop0', 'pop0.1', 'pop0.2', 'pop0.4']
    models = [1.0, 1.2, 1.5, 1.7, 2.0]
    trials = range(10)
    chks = np.arange(40, 243, 30)
    give = give_rew[h]
    postfix = postfixes[i]
    model = models[j]
    t = k
    chk = chks[l]
    
    exp_name = f"{give}p{model}n50{postfix}"
    model, (obs_rms, ret_rms) = \
        torch.load(f'../saved_checkpoints/meta_v2/{exp_name}_{t}/{chk}.pt')

    return model, obs_rms
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

def normalize_obs(obs, obs_rms):
    return (obs - obs_rms.mean) / np.sqrt(obs_rms.var + 1e-8) 

def get_sizes(res, obs_rms, last_only=False):
    """
    Additional simplification from unnormalize_obs, get sizes directly from 
    observations by unnormalizing
    """
    sizes = []
    for ep in range(len(res['obs'])):
        obs = unnormalize_obs(res['obs'][ep], obs_rms)
        if last_only:
            bsteps = res['data']['balloon_step'][ep]
            sizes.append(obs[bsteps, 5].round(2))
        else:
            sizes.append(obs[:, 5].round(2))
    return sizes

def comb_pca(res, layer='shared1', n_components=10, eps=None):
    """
    Perform PCA on a layer of activations
    Combines all episodes in the res call and returns PCAs back in
        per episode format
    eps: if passing a list of eps, use those instead of all episodes to form PCA
    """
    activ = res['activations'][layer]
    if eps is not None:
        ep_lens = [len(activ[e]) for e in eps]
        all_activ = torch.vstack([activ[e] for e in eps])
    else:
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


def comb_bottleneck(res, model, layer='shared1', eps=None):
    """
    Perform PCA on a layer of activations
    Combines all episodes in the res call and returns PCAs back in
        per episode format
    eps: if passing a list of eps, use those instead of all episodes to form PCA
    """
    activ = res['activations'][layer]
    if eps is not None:
        ep_lens = [len(activ[e]) for e in eps]
        all_activ = torch.vstack([activ[e] for e in eps])
    else:
        ep_lens = [len(a) for a in activ]
        all_activ = torch.vstack(activ)
        
    _, all_pca_a = model(all_activ)
    all_pca_a = all_pca_a.detach()
    pca_as = []

    cur_step = 0
    for ep_len in ep_lens:
        pca_as.append(all_pca_a[cur_step:cur_step+ep_len])
        cur_step += ep_len
        
    return pca_as

def split_by_lens(lens, data):
    cur = 0
    ep_data = []
    for l in lens:
        ep_data.append(data[cur:cur+l])
        cur = cur + l
    return ep_data


"""

Model/data loading functions

"""


## Code used to generate the best indexes for later analysis
# res = pickle.load(open('data/meta_representation_results', 'rb'))
# all_rew = res['rewards']
# best_idxs = all_rew.sum(axis=(5, 6)).argmax(axis=-1)
# pickle.dump(best_idxs, open('data/meta_representation_best_idxs', 'wb'))

def select_idxs(arr, by='best', idxs=None, axis=-1):
    '''
    Used to select with the best agent idxs from combined results
    by: method of selecting agents
        'best': best agent across tested checkpoints
        'first': first agent to reach >275 total banked size
        'close': closest agent to reach 275 total banked size
    If idxs is None, use the saved best indexes
    '''
    if idxs is None:
        if by == 'best':
            idxs = pickle.load(open('data/meta_representation_best_idxs', 'rb'))
        elif by == 'first': 
            idxs = pickle.load(open('data/meta_representation_first_idxs', 'rb'))
        elif by == 'close': 
            idxs = pickle.load(open('data/meta_representation_close_idxs', 'rb'))
    return np.take_along_axis(arr, 
                              np.expand_dims(idxs, axis=axis),
                              axis=axis).squeeze(axis=axis)

                              
def select_chks(arr, by='first'):
    '''
    Similar to select_idxs but instead convert each index in the arr to a tuple including
    the chk index
    '''
    if by == 'best':
        idxs = pickle.load(open('data/meta_representation_best_idxs', 'rb'))
    elif by == 'first': 
        idxs = pickle.load(open('data/meta_representation_first_idxs', 'rb'))
    elif by == 'close': 
        idxs = pickle.load(open('data/meta_representation_close_idxs', 'rb'))

    if type(arr) == list:
        new_idxs = [i + (idxs[i],) for i in arr]
    elif type(arr) == tuple:
        new_idxs = arr + (idxs[arr],)
    else:
        raise Exception('Arg should be list (for multiple idxs) or tuple (for one idx)')
    return new_idxs

def select_chks_by_dimension(h=None, i=None, j=None, by='first', with_chk=False):
    '''
    Get all indices of models including checkpoints selected by "by", fixing the dimensions provided
    dim:
        h: fix give_rew dimension
        i: fix pop punishment dimension
        j: fix p dimension
    '''
    it1 = range(2)
    it2 = range(4)
    it3 = range(5)
    trials = range(10)
        
    if h is not None:
        it1 = h
        if type(h) != list and type(h) != range:
            it1 = [h]
    if i is not None:
        it2 = i
        if type(i) != list and type(i) != range:
            it2 = [i]
    if j is not None:
        it3 = j
        if type(j) != list and type(j) != range:
            it3 = [j]
    
    idxs = (list(itertools.product(it1, it2, it3, trials)))
    idxs = [tuple(i) for i in idxs]

    if with_chk:
        return select_chks(idxs, by=by)
    return idxs
    
    
    
def select_random_model(size=1, idx=None, by='first', seed=None, load_models=False,
                        with_chk=True):
    '''
    Randomly select a tested model, picking a specific checkpoint
    corresponding to by: 'first'/'best'/'close'
    
    Function is pretty generally useful, so add an option
    idx: 4-tuple of (h,i,j,k), checkpoint will select automatically

    load_models:
        True: return model_idxs, models, obs_rmss, rs
        False: return model_idxs
    '''
    if by == 'best':
        idxs = pickle.load(open('data/meta_representation_best_idxs', 'rb'))
    elif by == 'first': 
        idxs = pickle.load(open('data/meta_representation_first_idxs', 'rb'))
    elif by == 'close': 
        idxs = pickle.load(open('data/meta_representation_close_idxs', 'rb'))

    if seed is not None:
        np.random.seed(seed)
        
    if idx is None:
        total_size = np.prod(idxs.shape)
        models = np.random.choice(np.arange(total_size), size=size, replace=False)
        models = [np.unravel_index(model, idxs.shape) for model in models]
        model_idxs = [model + (idxs[model],) for model in models]
    else:
        model_idxs = [idx + (idxs[idx],)]

    if load_models:
        models = []
        obs_rmss  = []
        rs = []
        for idx in model_idxs:
            h = idx[0]
            model, obs_rms = metabart_model_load(idx)
            give_rew = True if h == 0 else False
            
            r = evalu(model, obs_rms, give_rew)
            models.append(model)
            obs_rmss.append(obs_rms)
            rs.append(r)
        
        if size == 1:
            return model_idxs[0], models[0], obs_rmss[0], rs[0]
        return model_idxs, models, obs_rmss, rs
    
    if not with_chk:
        model_idxs = [idx[:4] for idx in model_idxs]
    if size == 1:
        return model_idxs[0]
    return model_idxs


"""

# Ramp to threshold decision process

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

def find_decision_nodes(res, model, ep=0, threshold=0.2, fixed_stim=True,
                        ret_scores=False):
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
        
        if fixed_stim:
            delt_rnn = torch.tensor(np.sign(delt_rnn)*2)
        else:
            delt_rnn = torch.tensor(delt_rnn * 2)
        
        
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
    decision_nodes = scores > threshold
    
    if ret_scores:
        return decision_nodes, scores
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



def score_decision_flow(res, model, large_kick=False):
    scores = []
    all_dec_nodes = np.full((17, 64), False)
    for ep in range(17):
        presses = np.argwhere((res['actions'][ep] == 1).reshape(-1)).reshape(-1)
        ends = np.array(res['data']['balloon_step'][ep])
        end_presses = np.intersect1d(presses, ends)
        penult_steps = end_presses - 1
        decision_nodes = find_decision_nodes(res, model, ep)
        all_dec_nodes[ep] = decision_nodes
        if decision_nodes.sum() == 0:
            scores.append(np.array([0., 0.]))
            continue
        delt_actor0, delt_rnn_sizes = measure_rnn_influence_multi(res, model, ep, penult_steps,
                                                  decision_nodes=decision_nodes,
                                                  large_kick=large_kick)
        # scores.append(np.abs(delt_actor0).mean(axis=0).var(axis=1).numpy())
        score = np.abs(delt_actor0).mean(axis=(0, 2))
        score[0] = score[0] / decision_nodes.sum()
        score[1] = score[1] / (~decision_nodes).sum()
        scores.append(score)
        
    scores = np.vstack(scores)
    
    return scores, all_dec_nodes

    

def measure_rnn_influence_multi(res, model, ep, steps, decision_nodes=None, ap=False,
                          large_kick=False, critic_change=False):
    '''Measure how much influence individual nodes or group of nodes have on a certain step
    
    decision_nodes: pass boolean array of size 64 to differentiate decision and non-decision nodes
    ap: if True, compute the influence on action probabilities

    critic_change: measure change to critic layer as well
    '''
    nsteps = len(steps)

    if decision_nodes is not None:
        size = 2
    else:
        size = 64
    
    delt_rnn_sizes = torch.zeros((nsteps, size, 64))
    rnn_hx_mod = torch.zeros((nsteps, size, 64))
    shared0 = torch.zeros((nsteps, size, 64))
    actor0 = torch.zeros((nsteps, size, 64))
    critic0 = torch.zeros((nsteps, size, 64))
    masks = np.array([1.], dtype='float32')
    for i, step in enumerate(steps):
        rnn_hx = torch.tensor(res['rnn_hxs'][ep][step])
        delt_rnn = torch.tensor(res['rnn_hxs'][ep][step+1] - res['rnn_hxs'][ep][step])
        if large_kick:
            delt_rnn = np.sign(delt_rnn) * 1

        for j in range(size):
            rnn_hx_mod[i, j] = rnn_hx
            shared0[i, j] = res['activations']['shared0'][ep][step]
            actor0[i, j] = res['activations']['actor0'][ep][step]
            critic0[i, j] = res['activations']['critic0'][ep][step]
        if decision_nodes is not None:
            rnn_hx_mod[i, 0, decision_nodes] += delt_rnn[decision_nodes]
            rnn_hx_mod[i, 1, ~decision_nodes] += delt_rnn[~decision_nodes]
            delt_rnn_sizes[i, 0, decision_nodes] += delt_rnn[decision_nodes]
            delt_rnn_sizes[i, 1, ~decision_nodes] += delt_rnn[~decision_nodes]
        else:
            for j in range(size):
                rnn_hx_mod[i, j, j] += delt_rnn[j]

    rnn_hx_mod = rnn_hx_mod.reshape(nsteps*size, 64)
    shared0 = shared0.reshape(nsteps*size, 64)

    next_rnn_hx = model.base._forward_gru(shared0, rnn_hx_mod, masks)[0]
    actor0mod = model.base.actor0(next_rnn_hx)
    actor0mod = actor0mod.reshape(nsteps, size, 64)
    delt_actor0 = (actor0mod - actor0).detach()
    critic0mod = model.base.critic0(next_rnn_hx)
    critic0mod = critic0mod.reshape(nsteps, size, 64)
    delt_critic0 = (critic0mod - critic0).detach()
    
    if ap:
        actor1 = model.base.actor1(actor0mod.reshape(nsteps*size, 64))
        logits = model.dist(actor1)
        probs = logits.probs.reshape(nsteps, size, 2)[:, :, 1].detach()
        return delt_actor0, probs
    
    if critic_change:
        return delt_actor0, delt_critic0

    if decision_nodes is not None:
        return delt_actor0, delt_rnn_sizes
        
    return delt_actor0


"""
Bimodal RNN structure

Determine whether there is a significant bimodal PCA in the RNN
"""

def find_bimodal_rnn_pca(rnn_hxs, lens, components=5):
    '''
    returns:
        has_bimodal, ep_rnn, ps (diptest p values), n (most bimodal PCA comp), ep_mean_rnn
    '''
    data = rnn_hxs.T  # change to [64, T]
    scaler = TimeSeriesScalerMeanVariance()
    data_normalized = scaler.fit_transform(data[:, :, np.newaxis])  # Shape becomes [64, T, 1]
    rnn_hxs = data_normalized.squeeze().T  # Back to shape [T, 64]
    pca = PCA(n_components=5)
    rnn_pc = pca.fit_transform(rnn_hxs)
    ep_rnn = split_by_lens(lens, rnn_pc)

    ep_mean_rnn = [r.mean(axis=0) for r in ep_rnn]
    ep_mean_rnn = np.vstack(ep_mean_rnn)
    ps = []
    for i in range(ep_mean_rnn.shape[1]):
        _, p = diptest(ep_mean_rnn[:, i])
        ps.append(p)
        
    # find which nodes have important components
    comp = np.abs(pca.components_[i])
    perc = np.percentile(comp, 80)
    bimodal_nodes = (comp > perc) * 1
    
    has_bimodal = min(ps) < 0.05
    n = np.argmin(ps)
    return has_bimodal, ep_rnn, ps, n, ep_mean_rnn

"""

Behavior scoring

Functions to score behavior on meta trials

"""

def score_unconfidence(res, ep=None, method=1):
    """
    Score unconfidence based on action probability on non button press steps
    Can pass an episode of list of episodes to score specific episodes
    method:
        0: take all steps where button was not pressed
        1: take all steps where ap < 0.5
    """
    if ep is not None:
        if type(ep) == int:
            ep = [ep]
        non_presses = (np.vstack([res['actions'][e] for e in ep]) == 0).reshape(-1)
        aps = np.vstack([res['action_probs'][e] for e in ep])[:, 1]
    else:
        non_presses = (np.vstack(res['actions']) == 0).reshape(-1)
        aps = np.vstack(res['action_probs'])[:, 1]
        
    if method == 0:
        unconfidence_score = aps[non_presses].mean()
    elif method == 1:
        unconfidence_score = aps[aps < 0.5].mean()

    return unconfidence_score

def compute_true_returns(rew, gamma=0.99):
    """
    Given a sequence of rewards, compute the actual returns experienced
    by the agent
    """
    returns = np.zeros(len(rew))
    cur_g = 0
    for i in range(len(rew)):
        cur_g = cur_g*0.99 + rew[len(rew)-i-1]
        returns[len(rew)-i-1] = cur_g
    return returns
    
"""

Line fitting

Some functions to easily make best fit lines

"""
def linear_best_fit(x, y):
    """
    Make a linear line of best fit
    returns:
        (m, b), r2
    """   
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