from tqdm import tqdm
from model_evaluation import forced_action_evaluate, load_model_and_env, nav_data_callback
import torch
from torch import nn, optim
from sklearn.metrics import accuracy_score, confusion_matrix
from  sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from scipy.ndimage import gaussian_filter1d, gaussian_filter
from collections import defaultdict
import numpy as np

import matplotlib.pyplot as plt
import proplot as pplt


import sys
sys.path.append('..')
from ppo.model import Policy
from ppo import utils
from ppo.envs import make_vec_envs

WINDOW_SIZE = (300, 300)

num_grid_slices = 5 # how many grid squares to slice maze into
num_grid_points = num_grid_slices**2

xs_grid = np.linspace(0, WINDOW_SIZE[0], num_grid_slices, endpoint=False)
ys_grid = np.linspace(0, WINDOW_SIZE[1], num_grid_slices, endpoint=False)

softmax = nn.Softmax(dim=1)


def color_cycle(cycle='default', idx=None):
    # Colors to assign to auxiliary tasks (they will be assigned in order)
    colors = pplt.Cycle(cycle).by_key()['color']
    hex_to_rgb = lambda h: tuple(int(h.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
    rgb_colors = np.array([hex_to_rgb(color) for color in colors])/255
    
    if idx is not None:
        return rgb_colors[idx]
    else:
        return rgb_colors

# Run the color_cycle function so that rgb_colors are globally available for 
#  any plotting functions from files that run this entire .py file
rgb_colors = color_cycle()

def activation_testing(model, env, x, y, angle):
    """
    Pass a model and corresponding environment, x, y and angle, then 
    get the observation and perform a prediction with model to get activations
    
    returns:
        outputs object from model.base.forward
    
    outputs['activations'] is the activation dict
    note that outputs['activations']['shared_activations'] is the same as the 
        rnn hidden state output
    """
    vis_walls = env.vis_walls
    vis_wall_refs = env.vis_wall_refs
    
    env.character.pos = np.array([x, y])
    env.character.angle = angle
    env.character.update_rays(vis_walls, vis_wall_refs)
    
    obs = torch.tensor(env.get_observation(), dtype=torch.float32).view(1, -1)
    rnn_hxs = torch.zeros(1, model.recurrent_hidden_state_size, dtype=torch.float32)
    masks = torch.zeros(1, 1, dtype=torch.float32)
    
    outputs = model.base.forward(obs, rnn_hxs, masks, with_activations=True)
    outputs['obs'] = obs
    return outputs



def stack_activations(activation_dict, also_ret_list=False):
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
        stacked_activations[key] = activations
        
        #Generate activations in list form
        if also_ret_list:
            for i in range(activations.shape[0]):
                list_activations.append(activations[i])
    
    if also_ret_list:
        return stacked_activations, list_activations
    else:
        return stacked_activations


def find_grid_index(point=None, x=None, y=None):
    '''
    Take a point or x and y coordinates and return the grid index
    that the position falls into. Note that coarseness of grid
    depends on parameter set in representation_analysis.py file
    '''
    if point is not None:
        x = point[0]
        y = point[1]
    elif x is not None and y is not None:
        pass
    else:
        raise Exception('No valid argument combination given')
        
    x_grid_idx = np.max(np.argwhere(x >= xs_grid))
    y_grid_idx = np.max(np.argwhere(y >= ys_grid))
    total_grid_idx = x_grid_idx * num_grid_slices + y_grid_idx
    return total_grid_idx



def train_classifier(x, labels, valid_x=None, valid_labels=None, num_labels=None,
                     lr=0.1, epochs=1000, prog=False):
    '''
    Train an arbitrary classifier
        x (tensor): train X data
        labels (tensor): train labels
        valid_x (tensor, optional): train valid X data
        valid_labels (tensor, optional): train valid labels
        num_labels (int, optional): number of output labels
            for model to output. If not provided, default is
            given by the largest label value in labels
    '''
    if num_labels is None:
        num_labels = labels.max().item() + 1
    linear = nn.Linear(x.shape[1], num_labels)
    optimizer = optim.Adam(linear.parameters(), lr=lr)
    softmax = nn.Softmax(dim=1)
    # optimizer = optim.SGD(linear.parameters(), lr=0.1, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    losses = []
    accuracies = []
    valid_accuracies = []
    if prog:
        it = tqdm(range(epochs))
    else:
        it = range(epochs)
        
    for i in it:
        y = linear(x)
        loss = criterion(y, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        accuracies.append(accuracy_score(softmax(y.detach()).argmax(axis=1), labels))

        if valid_x is not None and valid_labels is not None:
            pred_valid = softmax(linear(valid_x)).argmax(axis=1)
            valid_accuracies.append(accuracy_score(pred_valid, valid_labels))
            
    return linear, losses, accuracies, valid_accuracies


def draw_character(pos, angle, size=10, ax=None, color=None, alpha=1.):
    '''
    Given a position and angle, draw character to the given axis
    '''
    angle1 = angle - 0.3
    angle2 = angle + 0.3
    point1 = [pos[0], pos[1]]
    point2 = [pos[0] - np.cos(angle1)*size, pos[1] - np.sin(angle1)*size]
    point3 = [pos[0] - np.cos(angle2)*size, pos[1] - np.sin(angle2)*size]

    if color is None:
        color = np.array([0.9, 0.9, 0])

    poly = plt.Polygon([point1, point2, point3], fc=color, alpha=alpha)
    if ax is None:
        plt.gca().add_patch(poly)
    else:
        ax.add_patch(poly)

        
# def get_activations(model, obs, masks, pos, device=torch.device('cpu')):
#     eval_recurrent_hidden_states = torch.zeros(
#         num_processes, model.recurrent_hidden_state_size, device=device)

#     with torch.no_grad():
#         outputs = model.base(obs, eval_recurrent_hidden_states, 
#                                masks, deterministic=True, with_activations=True)
    
#     #For some reason, the first ~100ish activations are misaligned with the original
#     #activations if the agent is run with the episodes live, so remove some early misalignment
#     skip = 0
    
#     activations = outputs['activations']
#     stacked = {}
#     for key in activations:
#         substacked = []
#         for i in range(len(activations[key])):
#             activ = activations[key][i][skip:, :]
#             shape = activ.shape
#             substacked.append(activ.reshape(1, shape[0], shape[1]))
#         stacked[key] = torch.vstack(substacked)
#     pos = pos[skip:]
#     # angle = angle[skip:]

#     return stacked, pos
    

def train_position_classifier(model, obs_rms=None, kwargs=None, model_num=None, 
                              train_episodes=30, valid_episodes=5, epochs=100, 
                              seed=0, random_actions=True):
    '''
    Complete automated training of a position decoder
    model (str or Policy): the model to be trained, can either be a
        path to trained model from ../trained_models/ppo/ or a policy directly
    
    model (str):
        model_num (int): Need to also pass the trial number to load
    model (Policy):
        obs_rms, kwargs: These are usually loaded from load_model_and_env
            so when a Policy is passed, need to pass these as well
    '''
    
    if model_num is not None and type(model) == str:
        model, obs_rms, kwargs = load_model_and_env(model, model_num)
    elif obs_rms is not None and kwargs is not None \
        and type(model) == Policy:
        pass
    else:
        raise ValueError('Must pass either model (str) + model_num or model (Policy) + obs_rms + kwargs')
    
    # Generate activations and positions for training and validation
    train_results = perform_ep_collection(model, obs_rms, kwargs,
                                     seed=seed, num_episodes=train_episodes, random_actions=random_actions)
    stacked, grid_indexes = train_results['stacked'], train_results['grid_indexes']

    valid_results = perform_ep_collection(model, obs_rms, kwargs,
                                     seed=seed, num_episodes=valid_episodes, random_actions=random_actions)
    valid_stacked, valid_grid_indexes = train_results['stacked'], train_results['grid_indexes']
    
    # Train position decoders on activations
    classifiers = {}

    all_losses = []
    all_accuracies = []
    labels = []
    valid_accuracies = []
    
    valid_preds = []

    for key in stacked:
        # print(key)
        num_layers = stacked[key].shape[0]
        for i in range(num_layers):            
            x = stacked[key][i]
            valid_x = valid_stacked[key][i]
            
            linear, losses, accuracies, _ = train_classifier(x, grid_indexes, epochs=epochs)
            classifiers[f'{key}_{i}'] = linear
            
            labels.append(f'{key}_{i}')
            all_losses.append(losses)
            all_accuracies.append(accuracies)
    
            pred_valid = linear(valid_x).argmax(axis=1)
        
            valid_preds.append(pred_valid)
            
            accuracy_score(pred_valid, valid_grid_indexes)
            valid_accuracies.append(accuracy_score(pred_valid, valid_grid_indexes))
    return {
        'labels': labels,
        'losses': all_losses,
        'training_acc': all_accuracies,
        'final_valid_acc': valid_accuracies,
        'classifiers': classifiers,
        
        'valid_preds': valid_preds,
        'valid_grid_indexes': valid_grid_indexes
    }
    
    
def quick_vec_env(obs_rms, env_kwargs={}, env_name='NavEnv-v0', seed=0,
                 num_processes=1, eval_log_dir='/tmp/gym/_eval',
                 device=torch.device('cpu'), capture_video=False):
    eval_envs = make_vec_envs(env_name, seed + num_processes, num_processes,
                          None, eval_log_dir, device, True, 
                          capture_video=capture_video, 
                          env_kwargs=env_kwargs)

    vec_norm = utils.get_vec_normalize(eval_envs)
    if vec_norm is not None:
        vec_norm.eval()
        vec_norm.obs_rms = obs_rms
        
    return eval_envs

    
def perform_ep_collection(model, obs_rms=None, kwargs=None, model_num=None, 
                          num_episodes=1, seed=0, random_actions=False,
                         data_callback=nav_data_callback):
    '''
    Collect activations, positions, etc. for a number of episodes
    Used by train_position_classifier to collect activation and pos data
    Can also be used to generate single episodes of data to evaluate classifiers etc.
    
    Same model (str / Policy) parameters as train_position_classifier
    '''
    if model_num is not None and type(model) == str:
        model, obs_rms, kwargs = load_model_and_env(model, model_num)
    elif obs_rms is not None and kwargs is not None \
        and type(model) == Policy:
        pass
    else:
        raise ValueError('Must pass either model (str) + model_num or model (Policy) + obs_rms + kwargs')

    
    # Generate random episodes to train on
    if seed is not None:
        np.random.seed(seed)
    action_randomizer = lambda step: np.random.choice([0, 1, 2])
    
    if random_actions:
        results = forced_action_evaluate(model, obs_rms, forced_actions=action_randomizer, env_kwargs=kwargs, seed=seed,
                                        data_callback=data_callback, num_episodes=num_episodes, with_activations=True)
    else:
        results = forced_action_evaluate(model, obs_rms, env_kwargs=kwargs, seed=seed,
                                        data_callback=data_callback, num_episodes=num_episodes, with_activations=True)

    pos = np.vstack(results['data']['pos'])
    angle = torch.tensor(np.vstack(results['data']['angle']))
    stacked, listed_activ = stack_activations(results['activations'], also_ret_list=True)
    grid_indexes = torch.tensor([find_grid_index(p) for p in pos])
    
    results['pos'] = pos
    results['angle'] = angle
    results['stacked'], results['listed_activ'] = stacked, listed_activ
    results['grid_indexes'] = grid_indexes
    
    return results
    
    
    
def prune_fail_episodes(targets, dones):
    '''
    Remove episodes where the agent failed. Need to pass 'dones'
    from episode evaluations, which lets us determine when episode
    reached max length, as well as targets where the corresponding
    data points of failed episodes will be removed from
    
    E.g.
    
    '''
    max_ep_len = 202
    done_idxs = np.where(np.vstack(dones))[0]
    diffs = np.diff(done_idxs)
    incomplete_idxs = np.where(diffs == max_ep_len)[0]
    
    pruned_targets = []
    
    for i in range(len(done_idxs)):
        if i == 0:
            done_targets = targets[:done_idxs[i]]
        else:
            done_targets = targets[done_idxs[i-1]:done_idxs[i]]
        
        if i-1 not in incomplete_idxs:
            pruned_targets.append(done_targets)
    
    return pruned_targets



def split_by_ep(targets, dones):
    '''
    Given a collection of res data from an evalu() call, split up
    the data by episodes using res['dones']. Uses the True values from
    dones list to tell when a 
    
    For example, 
    res = evalu(...)
    ep_pos = split_by_ep(res['data']['pos'], res['dones'])
    '''
    done_idxs = np.where(np.vstack(dones))[0]
    split_targets = []
    for i in range(len(done_idxs)):
        if i == 0:
            done_targets = targets[:done_idxs[i]]
        else:
            done_targets = targets[done_idxs[i-1]:done_idxs[i]]

        split_targets.append(done_targets)
    return split_targets


def draw_box(env=None, corner=None, size=None, ax=None, c='y'):
    '''
    Draw a box to an axis. By default, use static platform location
    and size for platform
    '''
    if env is not None:
        for box in env.boxes:
            if box.is_goal:
                corner = box.corner
                size = box.size
                break
    else:
        if corner is None:
            corner = np.array([240, 60])
        if size == None:
            size = [20, 20]
    if type(size) != list:
        size = [size, size]
        
    box_array = np.vstack([
        corner,
        corner+np.array([0, size[1]]),
        corner+np.array([size[0], size[1]]),
        corner+np.array([size[0], 0]),
        corner
    ]).T
    
    if ax == None:
        plt.plot(box_array[0], box_array[1], c=c)
    else:
        ax.plot(box_array[0], box_array[1], c=c)
    
    
        
def dones_to_timesteps(dones):
    '''
    Given a set of dones, create episode-split timesteps that increment
    for each timestep (just used for plotting)
    '''
    i=1
    steps = []
    for j, done in enumerate(dones):
        if done[0] == True:
            i = 1
        steps.append(i)
        i += 1
    steps = np.array(steps)
    return steps



'''New representation analysis code for spatial and angle heatmaps'''

aux_tasks = ['none', 'wall0coef1', 'wall1coef1', 'wall01coef1', 'goaldistcoef1', 
             'terminalcoef1', 'catwall0coef1', 'catwall1coef1', 'catwall01coef1',
             'catquadcoef1', 'catfacewallcoef1', 'rewdistscale0015', 'rewexplore']
aux_labels = ['Control', 'AD (E)', 'AD (N)', 'AD (NE)', 'GD',
              'TP', 'LR (E)', 'LR (N)', 'LR (NE)', 'QP', 'FW', 'RD', 'RE']

def rasterize_pos(pos, sigma=1):
    '''
    Count up how many times points in space are visited
    Gives an overall idea of overall positions across trials
    
    sigma: sigma for gaussian filtering - 0 is no gaussian smoothing
    '''
    x_spaces = np.linspace(0, 300, 31)
    y_spaces = np.linspace(300, 0, 31)

    all_counts = []
    for i in range(len(y_spaces)-1):
        row_counts = []
        for j in range(len(x_spaces)-1):
            x_start = x_spaces[j]
            x_end = x_spaces[j+1]
            y_start = y_spaces[i+1]
            y_end = y_spaces[i]

            count = ((x_start <= pos[:, 0]) & (pos[:, 0] < x_end) & \
                     (y_start <= pos[:, 1]) & (pos[:, 1] < y_end)).sum()
            row_counts.append(count)

        all_counts.append(row_counts)

    return gaussian_filter(all_counts, sigma)


'''
Get mean heatmap similarities
'''
def mean_heatmap_similarity(hms1, hms2, true_mean=False):
    '''Get mean cosine similarity between heatmaps with hms1 as
    the base.
    This cosine similarity is measured as follows
        1. For each heatmap in hms1, find the corresponding heatmap in
            hms2 with the greatest cosine similarity
        2. Take the mean of these largest cosine similarities
    true_mean: 
    '''
    cs = cosine_similarity(hms1.reshape(16, -1), hms2.reshape(16, -1))
    max_sim = cs.max(axis=1)
    mean_sim = max_sim.mean()
    
    if true_mean:
        mean_sim = cs.mean()
    return mean_sim




def pairwise_heatmap_similarity(hm_res, hm_res2=None, typ='random', 
                                trials=range(10), chk=1500, 
                                activ_type='actor_activations',
                                layer_idx=1, drop_diagonal=True,
                                true_mean=False):
    '''Compute mean heatmap similarities'''
    all_sims = []
    for i in trials:
        row_sims = []
        for j in trials:            
            hms1 = hm_res[typ][i][chk][activ_type][layer_idx]
            
            if hm_res2 != None:
                hms2 = hm_res2[typ][j][chk][activ_type][layer_idx]
            else:
                hms2 = hm_res[typ][j][chk][activ_type][layer_idx]
                
            row_sims.append(mean_heatmap_similarity(hms1, hms2,
                                                   true_mean=true_mean))
        all_sims.append(row_sims)
    all_sims = np.array(all_sims)
    
    if drop_diagonal:
        mean_sim = np.mean(all_sims)
        for i in trials:
            all_sims[i, i] = mean_sim
    
    return all_sims



def internal_heatmap_similarity(hms):
    '''For a set of heatmaps, for each heatmap look for the most cosine similar
    heatmap. Take the means of these.
    
    This is used to calculate how much similarity there is within a single activation set'''
    
    if len(hms.shape) > 2:
        hms = hms.reshape(16, -1)
    cs = cosine_similarity(hms)
    cs[np.identity(16, dtype='bool')] = 0
    return np.mean(np.max(cs, axis=0))
    
    
def add_comb_hm(hm_res):
    '''
    Add heatmaps which combine heatmaps from policy and random initialization
    Which is a bit of a smoother representation
    '''
    activ_types = ['shared_activations', 'actor_activations', 'critic_activations']
    hm_res['comb'] = {}
    for trial in hm_res['policy']:
        hm_res['comb'][trial] = {}
        for chk in hm_res['policy'][trial]:
            hm_res['comb'][trial][chk] = {}
    
    for trial in hm_res['policy']:
        for chk in hm_res['policy'][trial]:
            for activ_type in activ_types:
                build_hms = []
                for layer_idx in range(hm_res['policy'][trial][chk][activ_type].shape[0]):
                    pol_hm = hm_res['policy'][trial][chk][activ_type][layer_idx]
                    ran_hm = hm_res['random'][trial][chk][activ_type][layer_idx]

                    comb_hm = np.stack([pol_hm, ran_hm]).mean(axis=0)
                    build_hms.append(comb_hm)
                hm_res['comb'][trial][chk][activ_type] = np.stack(build_hms)
                
            ran_pos = rasterize_pos(hm_res['random'][trial][chk]['pos'])
            pol_pos = rasterize_pos(hm_res['policy'][trial][chk]['pos'])
            comb_pos = np.stack([pol_pos, ran_pos]).mean(axis=0)
            hm_res['comb'][trial][chk]['pos'] = comb_pos
    
    
    
    
def get_pos_sim(hm_res, hm_res2=None, typ='comb', chk=1500,
                trials=range(10), rasterize=False):
    '''
    Get overall position frequency similarity between data.
    If no hm_res2 is given, only compare hm_res to itself.
    
    rasterize: Whether to rasterize the position to turn it into a heatmap
        Note that if typ == 'comb', then we are likely using positions generated
        from add_comb_hs, which are prerasterized so rasterize should be False.
        Otherwise if typ == 'policy' or 'random' rasterize should be True.
    '''
    
    if rasterize:
        rasterized_pos1 = [rasterize_pos(hm_res[typ][i][chk]['pos']) for i in trials]
    else:
        rasterized_pos1 = [np.array(hm_res[typ][i][chk]['pos']) for i in trials]
    rasterized_pos1 = np.array(rasterized_pos1).reshape(len(trials), -1)
    
    if hm_res2 == None:
        pos_cs = cosine_similarity(rasterized_pos1)
    else:
        if rasterize:
            rasterized_pos2 = [rasterize_pos(hm_res2[typ][i][chk]['pos']) for i in trials]
        else:
            rasterized_pos2 = [np.array(hm_res2[typ][i][chk]['pos']) for i in trials]
        rasterized_pos2 = np.array(rasterized_pos2).reshape(len(trials), -1)
        pos_cs = cosine_similarity(rasterized_pos1, rasterized_pos2)
        
    return pos_cs




def get_pos_sim_vs_hm_sim(hm_res, hm_res2=None, chk=1500,
                          typ='comb', activ_type='actor_activations',
                          layer_idx=1, trials=range(10),
                          triu_tril='both', true_hm_mean=False):
    '''
    Get both position cosine similarity (measured by cosine similarity
        between rasterized, smoothed position heatmaps) and 
        activation heatmaps.
    Note that activation heatmap cosine similarity is measured between
        the highest cosine similarity math
    triu_tril: 'triu', 'tril', or 'both'
    true_hm_mean: whether to use mean computation of heatmap cosine similarity
    '''
    hm_sims = pairwise_heatmap_similarity(hm_res, hm_res2, chk=chk,
                                          typ=typ, activ_type=activ_type,
                                          layer_idx=layer_idx, trials=trials,
                                          true_mean=true_hm_mean)
    
    # If combined policy and random data, position has been rasterized already
    if typ == 'comb':
        rasterize = False
    else:
        rasterize = True
        
    pos_sims = get_pos_sim(hm_res, hm_res2, typ=typ, chk=chk,
                           trials=trials, rasterize=rasterize)
    
    
    triu_pos_sims = np.triu(pos_sims, k=1)
    triu_hm_sims = np.triu(hm_sims, k=1)
    triu_pos_sims = triu_pos_sims[triu_pos_sims != 0]
    triu_hm_sims = triu_hm_sims[triu_hm_sims != 0]
    
    tril_pos_sims = np.tril(pos_sims, k=1)
    tril_hm_sims = np.tril(hm_sims, k=1)
    tril_pos_sims = tril_pos_sims[tril_pos_sims != 0]
    tril_hm_sims = tril_hm_sims[tril_hm_sims != 0]

    if triu_tril == 'triu':
        return triu_hm_sims, triu_pos_sims
    elif triu_tril == 'tril':
        return tril_hm_sims, tril_pos_sims
    elif triu_tril == 'both':
        return np.append(triu_hm_sims, tril_hm_sims), \
               np.append(triu_pos_sims, tril_pos_sims)
    
    
    
def get_pca_from_hms(hms, n_components=16, ret_pc_hms=False):
    '''
    Compute a principle component analysis from heatmaps
    ret_pc_hms: return the resulting PCs in a way that can be projected as heatmap as well
    '''
    if len(hms.shape) > 2:
        hms = hms.reshape(16, -1)
    pca = PCA(n_components=n_components)
    pca.fit(hms)
    
    if ret_pc_hms:
        return pca.components_.reshape(n_components, 30, 30)
    return pca



'''Internal Clustering'''

def remaining_idxs(idxs, remove):
    remaining = []
    for idx in idxs:
        if idx not in remove:
            remaining.append(idx)
            
    return np.array(remaining)


def internal_clusters(hms, similarity_cutoff=0.8, sort_clusters=True):
    '''
    Cluster heatmaps internally by cosine similarity to each other
    sort_clusters: sort each cluster from most central to least
    '''
    if len(hms.shape) > 2:
        hms = hms.reshape(16, -1)
    cs = np.abs(cosine_similarity(hms))
    
    idxs = np.arange(0, 16)
    clusters = []

    while len(idxs) > 0:
        remove_idxs = np.argwhere(cs[idxs[0], idxs] > similarity_cutoff).reshape(-1)
        cluster = idxs[remove_idxs]
        clusters.append(cluster)

        idxs = remaining_idxs(idxs, cluster)

        if len(clusters) > 16:
            break
    
    if sort_clusters:
        sorted_clusters = []
        for cluster in clusters:
            centrality = np.abs(cosine_similarity(hms[cluster])).mean(axis=0)
            sorted_clusters.append(cluster[np.argsort(centrality)])
        return sorted_clusters
    
    return clusters


def plot_hm_clusters(hms, similarity_cutoff=0.8, sort_clusters=True):
    '''Generate an array that fits clusters to plot with and plot
    '''
    clusters = internal_clusters(hms, similarity_cutoff, sort_clusters)
    
    cluster_lens = [len(cluster) for cluster in clusters]
    longest_cluster = np.max(cluster_lens)

    # Generate array based on cluster lengths
    array = []
    ax_idx = 1
    for l in cluster_lens:
        line = np.append(np.arange(ax_idx, ax_idx+l), np.zeros(longest_cluster - l)).astype('int')
        ax_idx = ax_idx + l
        array.append(line)
    array = np.array(array)

    # Plot out clusters
    fig, ax = pplt.subplots(array)
    ax_idx = 0
    for cluster in clusters:
        for hm_idx in cluster:
            ax[ax_idx].imshow(hms[hm_idx])
            ax_idx += 1

def num_internal_clusters(heatmaps):
    heatmaps_flat = np.array([h.reshape(-1) for h in heatmaps])
    clusters = internal_clusters(cosine_similarity(heatmaps_flat), 0.8)
    return len(clusters)



'''Angular Statistics to find heading representations'''

def compute_angle_statistics(angle, normalize_dists=True):
    '''
    !First attempt at finding heading representations - not used any more
    
    Compute 2 things from angles during episodes:
    At each step which direction is the agent closest to facing
    Angular distances to each direction
    '''
    east = (angle >= -np.pi/4) & (angle < np.pi/4)
    north = (angle >= np.pi/4) & (angle < 3*np.pi/4)
    south = (angle >= -3*np.pi/4) & (angle < -np.pi/4)
    west = ~north & ~east & ~south
    
    idxs = [east, north, west, south]

    dist_from_east = np.abs(angle)
    dist1 = np.abs(angle + 2*np.pi - np.pi/2)
    dist2 = np.abs(angle - np.pi/2)
    dist_from_north = np.min([dist1, dist2], axis=0)

    dist1 = np.abs(angle - 2*np.pi + np.pi/2)
    dist2 = np.abs(angle + np.pi/2)
    dist_from_south = np.min([dist1, dist2], axis=0)

    dist1 = np.abs(angle + 2*np.pi - np.pi)
    dist2 = np.abs(angle - np.pi)
    dist_from_west = np.min([dist1, dist2], axis=0)
            
    dists = [dist_from_east, dist_from_north, dist_from_west, dist_from_south]
    if normalize_dists:
        dists = [dist - np.pi/2 for dist in dists]
    
    return idxs, dists


def wrap_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def circular_gaussian_filter_fixed_angles(angles, weights, sigma=3, num_angles=100):
    '''Gaussian filter across angles'''
    # Sort angles and corresponding weights
    sorted_indices = np.argsort(angles)
    sorted_angles = angles[sorted_indices]
    sorted_weights = weights[sorted_indices]

    # Wrap the angles and weights around the circle to handle boundary cases
    wrapped_angles = np.concatenate((wrap_angle(sorted_angles - 2 * np.pi), sorted_angles, wrap_angle(sorted_angles + 2 * np.pi)))
    wrapped_weights = np.concatenate((sorted_weights, sorted_weights, sorted_weights))

    # Apply Gaussian filter
    filtered_weights = gaussian_filter1d(wrapped_weights, sigma)

    # Extract the filtered weights corresponding to the original angles
    filtered_weights = filtered_weights[len(angles): 2 * len(angles)]

    # Create uniformly spaced angles
    uniform_angles = np.linspace(-np.pi, np.pi, num_angles, endpoint=False)

    # Interpolate the filtered weights for the uniformly spaced angles
    uniform_filtered_weights = np.interp(uniform_angles, sorted_angles, filtered_weights)

    return uniform_angles, uniform_filtered_weights


def weighted_circular_statistics(angles, weights):
    '''Computed weighted circular statistics to compute coherence of angle activations'''
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



def activ_circular_statistics(angles, activs, ret=0, remove_mean=True):
    '''
    Compute general circular statistics for set of activations
    ret: what to return
        0: directions, resultant_lengths
        1: directions, resultant_lengths, filtered_angles, filtered_weights
    '''
    directions = []
    resultant_lengths = []
    weight_magnitudes = []
    filtered_weights = []
    for activ in activs:
        # Calculate where there is the most deviation from mean activations
        if remove_mean:
            uniform_angles, uniform_filtered_weights = circular_gaussian_filter_fixed_angles(angles, activ-activ.mean())
            mean_direction, mean_resultant_length = weighted_circular_statistics(uniform_angles, np.abs(uniform_filtered_weights))
        else:
            uniform_angles, uniform_filtered_weights = circular_gaussian_filter_fixed_angles(angles, activ)
            mean_direction, mean_resultant_length = weighted_circular_statistics(uniform_angles, uniform_filtered_weights)
            
        
        directions.append(mean_direction)
        resultant_lengths.append(mean_resultant_length)
        weight_magnitudes.append(np.abs(uniform_filtered_weights))
        filtered_weights.append(uniform_filtered_weights)
        
    if ret == 0:
        return directions, resultant_lengths
    elif ret == 1:
        return directions, resultant_lengths, uniform_angles, filtered_weights
    
        
def dist_from_angle(angles, target_angle):
    dist1 = np.abs(angles - target_angle)
    dist2 = np.abs(angles + 2*np.pi - target_angle)
    dist3 = np.abs(angles + 2*np.pi + target_angle)
    dist4 = np.abs(angles - 2*np.pi + target_angle)
    dist5 = np.abs(angles - 2*np.pi - target_angle)
    
    dist = np.min([dist1, dist2, dist3, dist4, dist5], axis=0)
    return dist



def positive_negative_resultants(node_weights, comb=False):
    '''
    Assume that weights are given for points from np.linspace(-np.pi, np.pi, 100)
    and compute positive vs negative resultants (resultant directions and lengths
    when only considering positive or negative activations)
    
    comb: whether to combine positive and negative directions for easy combined analysis
    '''
    uniform_angles = uniform_angles = np.linspace(-np.pi, np.pi, 100, endpoint=False)
    
    pos_directions = []
    pos_lengths = []
    neg_directions = []
    neg_lengths = []
    for i in range(16):
        weights = node_weights[i]
        pos_weights = weights.copy()
        neg_weights = weights.copy()
        pos_weights[pos_weights < 0] = 0
        neg_weights[neg_weights > 0] = 0

        direction, length = weighted_circular_statistics(uniform_angles, pos_weights)
        pos_directions.append(direction)
        pos_lengths.append(length)
        direction, length = weighted_circular_statistics(uniform_angles, neg_weights)
        neg_directions.append(direction)
        neg_lengths.append(length)
        
    if comb:
        directions = pos_directions + neg_directions
        lengths = pos_lengths + neg_lengths
        return directions, lengths
    else:
        return pos_directions, pos_lengths, neg_directions, neg_lengths
    
    
'''Spatial heatmap silhouette scores
In order to measure how "good" a spatial representation is, we consider
how well separated the positive and negative activations are
'''

# Mixed periodicity
def periodic_distance(a, b, box_size):
    diff = a[:, np.newaxis] - b
    dist = np.abs(diff)
    dist = np.minimum(dist, box_size - dist)
    return np.sqrt(np.sum(dist**2, axis=-1))

def non_periodic_distance(a, b):
    diff = a[:, np.newaxis] - b
    return np.sqrt(np.sum(diff**2, axis=-1))

def mixed_silhouette_score(data1, data2, weights1, weights2, box_size=300):
    # Combine the two sets of data into one array and create the combined weights
    combined_data = np.vstack((data1, data2))
    combined_weights = np.hstack((weights1, weights2))

    # Compute the pairwise distances between all data points using periodic and non-periodic distances
    periodic_distances = periodic_distance(combined_data, combined_data, box_size)
    non_periodic_distances = non_periodic_distance(combined_data, combined_data)
    distances = np.block([
        [periodic_distances[:len(data1), :len(data1)], non_periodic_distances[:len(data1), len(data1):]],
        [non_periodic_distances[len(data1):, :len(data1)], periodic_distances[len(data1):, len(data1):]]
    ])

    # Calculate the weighted average intra-cluster distances for each point
    intra_cluster_distances = np.hstack((
        np.sum(distances[:len(data1), :len(data1)] * weights1[:, np.newaxis], axis=1) / (weights1.sum() - weights1),
        np.sum(distances[len(data1):, len(data1):] * weights2[:, np.newaxis], axis=1) / (weights2.sum() - weights2)
    ))

    # Calculate the weighted average nearest-cluster distances for each point
    nearest_cluster_distances = np.hstack((
        np.sum(distances[:len(data1), len(data1):] * weights2[np.newaxis, :], axis=1) / weights2.sum(),
        np.sum(distances[len(data1):, :len(data1)] * weights1[np.newaxis, :], axis=1) / weights1.sum()
    ))

    # Compute the silhouette scores for each point
    silhouette_scores = (nearest_cluster_distances - intra_cluster_distances) / np.maximum(intra_cluster_distances, nearest_cluster_distances)

    # Compute the weighted silhouette score
    weighted_score = np.sum(silhouette_scores * combined_weights) / combined_weights.sum()

    return weighted_score


def compute_spatial_silhouette(hms, ret_dist=True, remove_mean=True):
    '''Given a set of spatial activation heatmaps (e.g. shape (16,30,30))
    compute the weighted silhouette score
    
    ret_dist: whether to also return the differences between heatmap and their means
        for plotting
    remove_mean: whether to consider activation - activation.mean as the heatmap
    
    '''
    grid = np.linspace(5, 295, 30)
    xs, ys = np.meshgrid(grid, grid)
    
    hm_dists = []
    sil_scores = []
    for i in range(hms.shape[0]):
        if remove_mean:
            hm_dist = hms[i] - hms[i].mean()
        else:
            hm_dist = hms[i]
        hm_pos = hm_dist.copy()
        hm_neg = hm_dist.copy()
        hm_pos[hm_pos < 0] = 0
        hm_neg[hm_neg > 0] = 0

        x, y = xs[hm_pos > 0], ys[hm_pos > 0]
        hm_pos_grids = np.array([x, y]).T
        hm_pos_weights = np.abs(hm_pos[hm_pos > 0])
        x, y = xs[hm_neg < 0], ys[hm_neg < 0]
        hm_neg_grids = np.array([x, y]).T
        hm_neg_weights = np.abs(hm_neg[hm_neg < 0])

        sil_score = mixed_silhouette_score(hm_pos_grids, hm_neg_grids, 
                                           hm_pos_weights, hm_neg_weights)
        
        hm_dists.append(hm_dist)
        sil_scores.append(sil_score)
    
    if ret_dist:
        return sil_scores, np.stack(hm_dists)
    else:
        return sil_scores
    
    
    
'''
Alternate computation of spatial representation score - summing exponentially distance
weighted activations
'''

def exponential_cluster_score(grid_points, weights, sigma=50):
    '''
    Compute the exponential cluster score of a set of points and weights.
    Pass in either positive or negative activation heatmaps
    
    Computes the distance between each point, performing exponential kernel
    Computes combined weights as the product of pairwise weights
    Sums the distance weighted combined weights and divides by weights squared for normalization
    '''
    diff = (grid_points[:, np.newaxis] - grid_points)
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    avg_weights = (weights[:, np.newaxis] * weights)
    exp_dist = np.exp(-dists / sigma)
    return np.sum(avg_weights * exp_dist), np.sum(weights**2)
    
def spatial_representation_exponential_score(hm, sigma=50):
    '''
    Compute spatial representation  score using exponential distance weighting
    Split into positive and negative (above and below average activations on heatmap)
        and compute exponential_cluster_score of each set
    '''
    extent = (5, 295)
    num_grid = 30
    grid = np.linspace(extent[0], extent[1], num_grid)
    xs, ys = np.meshgrid(grid, grid)
    
    hm_dist = hm - hm.mean()
    hm_pos = hm_dist.copy()
    hm_neg = hm_dist.copy()
    hm_pos[hm_pos < 0] = 0
    hm_neg[hm_neg > 0] = 0

    x, y = xs[hm_pos > 0], ys[hm_pos > 0]
    hm_pos_grids = np.array([x, y]).T
    hm_pos_weights = np.abs(hm_pos[hm_pos > 0])
    x, y = xs[hm_neg < 0], ys[hm_neg < 0]
    hm_neg_grids = np.array([x, y]).T
    hm_neg_weights = np.abs(hm_neg[hm_neg < 0])
    
    pos_score, pos_norm = exponential_cluster_score(hm_pos_grids, hm_pos_weights, sigma)
    neg_score, neg_norm = exponential_cluster_score(hm_neg_grids, hm_neg_weights, sigma)
    
    return (pos_score + neg_score) / (pos_norm + neg_norm) / 100

def compute_spatial_rep_scores(hms):
    '''
    Compute spatial representation score for a set of activation heatmaps
    '''
    scores = []
    for hm in hms:
        scores.append(spatial_representation_exponential_score(hm))
    return scores