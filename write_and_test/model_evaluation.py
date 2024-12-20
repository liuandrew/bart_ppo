import gymnasium as gym
import gym_bart
import torch
import sys
sys.path.insert(0, '..')
from evaluation import evaluate
from ppo.model import Policy
from ppo import utils
from ppo.envs import make_vec_env
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pickle
from pathlib import Path
import re
from matplotlib import animation
from IPython.display import HTML

from scipy.ndimage import gaussian_filter
# import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# Turn noff logging warnings for gymnasium on get_attr() being deprecated
import logging
logging.getLogger('gymnasium').setLevel(logging.CRITICAL)

device = torch.device("cpu")

save_folder = 'plots/proof_of_concept/'

'''
General operation of functions in this file:
print_trained_models(): Run this first to get a list of trained models
    in the trained_models folder.
model, obs_rms, env_kwargs = load_model_and_env(model_name):
    For model name, pass the entire path to the trained model from ppo folder
    e.g., model_name = 'nav_auxiliary_tasks/nav_aux_wall_1_t1.pt'

Now the model has been loaded and can be evaluated
results = evalu(model, obs_rms, n=10, env_name='NavEnv-v0', env_kwargs=env_kwargs)
    This will actually evaluate the model on the given environment
    Note that we evaluate with a single environment at a time to make data_callback
        easier to work with, but if no data_callback is passed we can parallelize.
        Note also that capture video only works on the first vectorized env
    The results will be a dictionary with information about the episodes
    We can run this with capture_video=True to record videos
    We can optionally pass a data_callback to collect extra information
        (see example data_callback functions)

To dig into the results of a single episode, use the following
get_ep(results['all_obs'], results['ep_ends'], ep_num=2)
The evalu() function includes the episode ending numbers for this purpose
'''

'''
================================================================
Evaluation function
================================================================
'''
def forced_action_evaluate(actor_critic, obs_rms=None, normalize=True, forced_actions=None,
             env_name='NavEnv-v0', seed=None, num_processes=1,
             device=torch.device('cpu'), ret_info=1, capture_video=False, env_kwargs={}, data_callback=None,
             num_episodes=10, verbose=0, with_activations=False, deterministic=True,
             aux_wrapper_kwargs={}, auxiliary_truth_sizes=[], auxiliary_tasks=[],
             auxiliary_task_args=[],
             eval_log_dir=None, video_folder='./video', with_aux=False,
             add_last_rnn=True, random_init_rnn_hxs=False):
    '''
    ret_info: level of info that should be tracked and returned
    capture_video: whether video should be captured for episodes
    env_kwargs: any kwargs to create environment with
    add_last_rnn: when True, add te final rnn_hx to the recorded rnn_hxs
    random_init_rnn_hxs: if True, randomize initial rnn_hxs with N(0, 0.2)
    data_callback: a function that should be called at each step to pull information
        from the environment if needed. The function will take arguments
            def callback(actor_critic, vec_envs, recurrent_hidden_states, data):
        actor_critic: the actor_critic network
        vec_envs: the vec envs (can call for example vec_envs.get_attr('objects') to pull data)
        recurrent_hidden_states: these are given in all data, but may want to use in computation
        obs: observation this step (after taking action) - 
            note that initial observation is never seen by data_callback
            also note that this observation will have the mean normalized
            so may instead want to call vec_envs.get_method('get_observation')
        action: actions this step
        reward: reward this step
        data: a data dictionary that will continuously be passed to be updated each step
            it will start as an empty dicionary, so keys must be initialized
        see below at example_data_callback in this file for an example
    '''

    if seed is None:
        seed = np.random.randint(0, 1e9)
    torch.manual_seed(seed)

    envs = make_vec_env(env_name, 
                         seed=seed, 
                         n_envs=num_processes,
                         env_kwargs=env_kwargs,
                         auxiliary_tasks=auxiliary_tasks,
                         auxiliary_task_args=auxiliary_task_args,
                         normalize=normalize,
                         dummy=True,)

    if normalize:
        envs.training = False
    if obs_rms is not None:
        envs.obs_rms = obs_rms

    eval_episode_rewards = []

    all_obs = []
    all_actions = []
    all_action_log_probs = []
    all_action_probs = []
    all_rewards = []
    all_rnn_hxs = []
    all_dones = []
    all_masks = []
    all_activations = []
    all_values = []
    all_actor_features = []
    all_auxiliary_preds = []
    all_auxiliary_truths = []
    data = {}
    
    ep_obs = []
    ep_actions = []
    ep_action_log_probs = []
    ep_action_probs = []
    ep_rewards = []
    ep_rnn_hxs = []
    ep_dones = []
    ep_values = []
    ep_masks = []
    ep_actor_features = []
    
    ep_auxiliary_preds = []
    ep_activations = []
    ep_auxiliary_truths = []
    

    obs = envs.reset()
    obs = torch.tensor(obs)
    rnn_hxs = torch.zeros(
        num_processes, actor_critic.recurrent_hidden_state_size, device=device)
    masks = torch.zeros(num_processes, 1, device=device)

    for ep in range(num_episodes):
        step = 0
        while True:
            ep_obs.append(obs)
            ep_rnn_hxs.append(rnn_hxs)
            if data_callback is not None and step == 0:
                data_inputs = {
                    'model': actor_critic,
                    'envs': envs,
                    'rnn_hxs': rnn_hxs,
                    'obs': obs,
                }
                data = data_callback(data_inputs, data, first=True)

            with torch.no_grad():
                
                if random_init_rnn_hxs:
                    for i, mask in enumerate(masks):
                        if mask == 0:
                            rnn_hxs[i] = torch.tensor(np.random.normal(0, 0.2, size=(64,))) 
                
                outputs = actor_critic.act(obs, rnn_hxs, 
                                        masks, deterministic=deterministic,
                                        with_activations=with_activations)
                if forced_actions is None:
                    action = outputs['action']
                elif type(forced_actions) in [int, float]:
                    action = torch.full((num_processes, 1), forced_actions)
                elif type(forced_actions) in [list, np.ndarray]:
                    # Can give partial episodes - after actions run out will use outputs
                    if step >= len(forced_actions):
                        action = outputs['action']
                    else:
                        action = torch.full((num_processes, 1), forced_actions[step])
                    
                elif type(forced_actions) == type(lambda:0):
                    actions = [torch.tensor(forced_actions(step)) for i in range(num_processes)]
                    action = torch.vstack(actions) 
                elif type(forced_actions) == dict:
                    # special case: assume a dict where each episode's actions are laid out
                    action = torch.full((num_processes, 1), forced_actions[ep][step])
                                                
                rnn_hxs = outputs['rnn_hxs']
                obs, reward, done, infos = envs.step(action)
                obs = torch.tensor(obs)
            
            masks = torch.tensor(
                [[0.0] if done_ else [1.0] for done_ in done],
                dtype=torch.float32,
                device=device)
            
            ep_actions.append(action)
            ep_action_log_probs.append(outputs['action_log_probs'])
            ep_action_probs.append(outputs['probs'])
            ep_rewards.append(reward)
            ep_dones.append(done)
            ep_values.append(outputs['value'])
            ep_masks.append(masks)
            ep_actor_features.append(outputs['actor_features'])
            
            if 'auxiliary_preds' in outputs:
                ep_auxiliary_preds.append(outputs['auxiliary_preds'])
            
            if with_activations:
                ep_activations.append(outputs['activations'])

            if data_callback is not None:
                data_inputs = {
                    'model': actor_critic,
                    'envs': envs,
                    'rnn_hxs': rnn_hxs,
                    'obs': obs,
                    'action': action,
                    'reward': reward,
                    'done': done,
                    'info': infos,
                    'step': step,
                }
                data = data_callback(data_inputs, data)
            
            auxiliary_truths = [[] for i in range(len(actor_critic.auxiliary_output_sizes))]
            if with_aux:
                for info in infos:
                    if 'auxiliary' in info and len(info['auxiliary']) > 0:
                        for i, aux in enumerate(info['auxiliary']):
                            auxiliary_truths[i].append(aux)
                if len(auxiliary_truths) > 0:
                    auxiliary_truths = [torch.tensor(np.vstack(aux)) for aux in auxiliary_truths]
            ep_auxiliary_truths.append(auxiliary_truths)
            
            step += 1
            
            if done[0]:
                if add_last_rnn:
                    ep_rnn_hxs.append(rnn_hxs)
                all_obs.append(np.vstack(ep_obs))
                all_actions.append(np.vstack(ep_actions))
                all_action_log_probs.append(np.vstack(ep_action_log_probs))
                all_action_probs.append(np.vstack(ep_action_probs))
                all_rewards.append(np.vstack(ep_rewards))
                all_rnn_hxs.append(np.vstack(ep_rnn_hxs))
                all_dones.append(np.vstack(ep_dones))
                all_masks.append(np.vstack(ep_masks))
                all_values.append(np.vstack(ep_values))
                all_actor_features.append(np.vstack(ep_actor_features))
                
                all_auxiliary_preds.append(ep_auxiliary_preds)
                all_activations.append(ep_activations)
                all_auxiliary_truths.append(ep_auxiliary_truths)

                if data_callback is not None:
                    data_inputs = {
                        'model': actor_critic,
                        'envs': envs
                    }
                    data = data_callback(data_inputs, data, stack=True)
                          
                if verbose >= 2:
                    print(f'ep {i}, rew {np.sum(ep_rewards)}' )
                    
                ep_obs = []
                ep_actions = []
                ep_action_log_probs = []
                ep_action_probs = []
                ep_rewards = []
                ep_rnn_hxs = []
                ep_dones = []
                ep_values = []
                ep_masks = []
                ep_actor_features = []
                
                ep_auxiliary_preds = []
                ep_activations = []
                ep_auxiliary_truths = []
                
                rnn_hxs = torch.zeros(
                    num_processes, actor_critic.recurrent_hidden_state_size, device=device)
                step = 0
                
                break
    envs.close()
    if verbose >= 1:
        print(" Evaluation using {} episodes: mean reward {:.5f}\n".format(
            len(eval_episode_rewards), np.mean(eval_episode_rewards)))

    return {
        'obs': all_obs,
        'actions': all_actions,
        'action_log_probs': all_action_log_probs,
        'action_probs': all_action_probs,
        'rewards': all_rewards,
        'rnn_hxs': all_rnn_hxs,
        'dones': all_dones,
        'masks': all_masks,
        'envs': envs,
        'data': data,
        'activations': all_activations,
        'values': all_values,
        'actor_features': all_actor_features,
        'auxiliary_preds': all_auxiliary_preds,
        'auxiliary_truths': all_auxiliary_truths,
    }



def forced_action_evaluate_multi(actor_critic, obs_rms=None, normalize=True, forced_actions=None,
             env_name='NavEnv-v0', seed=None, num_processes=1,
             device=torch.device('cpu'), ret_info=1, capture_video=False, env_kwargs={}, data_callback=None,
             num_episodes=10, verbose=0, with_activations=False, deterministic=True,
             aux_wrapper_kwargs={}, auxiliary_truth_sizes=[], auxiliary_tasks=[],
             auxiliary_task_args=[],
             eval_log_dir=None, video_folder='./video', with_aux=False):
    '''
    Rewritten forced_action_evaluate function for multiple parallel num_processes
    Useful for running multiple environments with different conditions (e.g. pass
     a list of env_kwargs and set num_processes)
    Note that data callback needs to be able to handle multiple processes
     See meta_bart_multi_callback for an example
    !Forced action has not been tested
    '''

    if seed is None:
        seed = np.random.randint(0, 1e9)

    envs = make_vec_env(env_name, 
                         seed=seed, 
                         n_envs=num_processes,
                         env_kwargs=env_kwargs,
                         auxiliary_tasks=auxiliary_tasks,
                         auxiliary_task_args=auxiliary_task_args,
                         normalize=normalize,
                         dummy=True,)
    torch.manual_seed(seed)

    if normalize:
        envs.training = False
    if obs_rms is not None:
        envs.obs_rms = obs_rms

    eval_episode_rewards = []

    all_obs = []
    all_actions = []
    all_action_log_probs = []
    all_action_probs = []
    all_rewards = []
    all_rnn_hxs = []
    all_dones = []
    all_masks = []
    all_activations = []
    all_values = []
    all_actor_features = []
    all_auxiliary_preds = []
    all_auxiliary_truths = []
    data = {}
    
    ep_obs = []
    ep_actions = []
    ep_action_log_probs = []
    ep_action_probs = []
    ep_rewards = []
    ep_rnn_hxs = []
    ep_dones = []
    ep_values = []
    ep_masks = []
    ep_actor_features = []
    
    ep_auxiliary_preds = []
    ep_activations = []
    ep_auxiliary_truths = []
    
    multi_dones = np.full(num_processes, False)

    obs = envs.reset()
    obs = torch.tensor(obs)
    rnn_hxs = torch.zeros(
        num_processes, actor_critic.recurrent_hidden_state_size, device=device)
    masks = torch.zeros(num_processes, 1, device=device)

    for ep in range(num_episodes):
        step = 0
        while True:
            ep_obs.append(obs)
            ep_rnn_hxs.append(rnn_hxs)
            if data_callback is not None and step == 0:
                data_inputs = {
                    'model': actor_critic,
                    'envs': envs,
                    'rnn_hxs': rnn_hxs,
                    'obs': obs,
                }
                data = data_callback(data_inputs, data, first=True,
                                     num_processes=num_processes)

            with torch.no_grad():
                outputs = actor_critic.act(obs, rnn_hxs, 
                                        masks, deterministic=deterministic,
                                        with_activations=with_activations)
                if forced_actions is None:
                    action = outputs['action']
                else:
                    action = torch.zeros((num_processes, 1), dtype=torch.int64)
                    for proc_num in range(num_processes):
                        if len(forced_actions[proc_num]) > step:
                            action[proc_num] = forced_actions[proc_num][step]
                        else:
                            action[proc_num] = 1
                rnn_hxs = outputs['rnn_hxs']
                obs, reward, done, infos = envs.step(action)
                obs = torch.tensor(obs)
            
            masks = torch.tensor(
                [[0.0] if done_ else [1.0] for done_ in done],
                dtype=torch.float32,
                device=device)
            
            ep_actions.append(action)
            ep_action_log_probs.append(outputs['action_log_probs'])
            ep_action_probs.append(outputs['probs'])
            ep_rewards.append(reward)
            ep_dones.append(done)
            ep_values.append(outputs['value'])
            ep_masks.append(masks)
            ep_actor_features.append(outputs['actor_features'])
            
            if 'auxiliary_preds' in outputs:
                ep_auxiliary_preds.append(outputs['auxiliary_preds'])
            
            if with_activations:
                ep_activations.append(outputs['activations'])

            if data_callback is not None:
                data_inputs = {
                    'model': actor_critic,
                    'envs': envs,
                    'rnn_hxs': rnn_hxs,
                    'obs': obs,
                    'action': action,
                    'reward': reward,
                    'done': done,
                    'info': infos,
                    'step': step,
                }
                data = data_callback(data_inputs, data, num_processes=num_processes)
            
            auxiliary_truths = [[] for i in range(len(actor_critic.auxiliary_output_sizes))]
            if with_aux:
                for info in infos:
                    if 'auxiliary' in info and len(info['auxiliary']) > 0:
                        for i, aux in enumerate(info['auxiliary']):
                            auxiliary_truths[i].append(aux)
                if len(auxiliary_truths) > 0:
                    auxiliary_truths = [torch.tensor(np.vstack(aux)) for aux in auxiliary_truths]
            ep_auxiliary_truths.append(auxiliary_truths)
            
            step += 1
            
            multi_dones = multi_dones | np.array(done)
            
            if multi_dones.all():
                all_obs.append(np.array(ep_obs).transpose(1, 0, 2))
                all_actions.append(np.array(ep_actions).transpose(1, 0, 2))
                all_action_log_probs.append(np.array(ep_action_log_probs).transpose(1, 0, 2))
                all_action_probs.append(np.array(ep_action_probs).transpose(1, 0, 2))
                all_rewards.append(np.array(ep_rewards).transpose(1, 0))
                all_rnn_hxs.append(np.array(ep_rnn_hxs).transpose(1, 0, 2))
                all_dones.append(np.array(ep_dones).transpose(1, 0))
                all_masks.append(np.array(ep_masks).transpose(1, 0, 2))
                all_values.append(np.array(ep_values).transpose(1, 0, 2))
                all_actor_features.append(np.array(ep_actor_features).transpose(1, 0, 2))
                
                all_auxiliary_preds.append(ep_auxiliary_preds)
                all_activations.append(ep_activations)
                all_auxiliary_truths.append(ep_auxiliary_truths)

                if data_callback is not None:
                    data_inputs = {
                        'model': actor_critic,
                        'envs': envs
                    }
                    data = data_callback(data_inputs, data, stack=True,
                                         num_processes=num_processes)
                          
                if verbose >= 2:
                    print(f'ep {i}, rew {np.sum(ep_rewards)}' )
                    
                ep_obs = []
                ep_actions = []
                ep_action_log_probs = []
                ep_action_probs = []
                ep_rewards = []
                ep_rnn_hxs = []
                ep_dones = []
                ep_values = []
                ep_masks = []
                ep_actor_features = []
                
                ep_auxiliary_preds = []
                ep_activations = []
                ep_auxiliary_truths = []
                
                step = 0
                
                rnn_hxs = torch.zeros(
                    num_processes, actor_critic.recurrent_hidden_state_size, device=device)
                break
    envs.close()
    if verbose >= 1:
        print(" Evaluation using {} episodes: mean reward {:.5f}\n".format(
            len(eval_episode_rewards), np.mean(eval_episode_rewards)))

    return {
        'obs': all_obs,
        'actions': all_actions,
        'action_log_probs': all_action_log_probs,
        'action_probs': all_action_probs,
        'rewards': all_rewards,
        'rnn_hxs': all_rnn_hxs,
        'dones': all_dones,
        'masks': all_masks,
        # 'envs': envs,
        'data': data,
        'activations': all_activations,
        'values': all_values,
        'actor_features': all_actor_features,
        'auxiliary_preds': all_auxiliary_preds,
        'auxiliary_truths': all_auxiliary_truths,
    }
'''
================================================================
Data Callbacks
================================================================
'''
def bart_toggle_data_callback(data_inputs={}, data={}, first=False, stack=False):
    '''
    Track some bart toggle task information
    
    data_inputs: may have model, envs, rnn_hxs, obs, action, reward, done, info
    data: currently tracked data dictionary, fill it up in this function with passed inputs
    first: called on the first step of every reset, only passses model, envs, 
        rnn_hxs (zeros) and first obs
    stack: final call after episode is done, only passes model and envs. If collecting data
        per step, this is the step where it can be stacked and stored as an episode of data
    '''
    if len(data) == 0:
        data['color'] = []
        data['end_size'] = []
        data['popped'] = []
        data['inflate_delay'] = []
        data['balloon_limit'] = []
        
    
    if 'done' in data_inputs and data_inputs['done']:
        info = data_inputs['info'][0]
        data['color'].append(info['current_color'])
        data['end_size'].append(info['last_size'])
        data['popped'].append(info['popped'])
        data['inflate_delay'].append(info['inflate_delay'])
        data['balloon_limit'].append(info['balloon_limit'])
        
    return data

def meta_bart_callback(data_inputs={}, data={}, first=False, stack=False):
    keys = ['current_color', 'last_size', 'balloon_limit', 'inflate_delay', 'popped']
    # print(data_inputs)
    
    if len(data) == 0:
        data['balloon_means'] = []
        data['ep_balloon_means'] = []
        for key in keys:
            data[key] = []
            data[f'ep_{key}'] = []    
        
    if 'info' in data_inputs:
        info = data_inputs['info'][0]
        if len(data['ep_balloon_means']) == 0:
            data['ep_balloon_means'].append(
                data_inputs['envs'].get_attr('balloon_mean_sizes')[0]
            )
        
        if 'bart_finished' in info and info['bart_finished']:
            for key in keys:
                data[f'ep_{key}'].append(info[key])
        
        if 'done' in data_inputs and data_inputs['done']:
            for key in keys:
                data[key].append(data[f'ep_{key}'])
                data[f'ep_{key}'] = []
            data['balloon_means'].append(
                data['ep_balloon_means'][0]
            )
            data['ep_balloon_means'] = []

    return data
    

def meta_bart_multi_callback(data_inputs={}, data={}, first=False, stack=False,
                             num_processes=1):
    keys = ['current_color', 'last_size', 'balloon_limit', 'inflate_delay', 
            'popped']
    # print(data_inputs)
    
    if len(data) == 0:
        data['balloon_means'] = []
        data['ep_balloon_means'] = []
        data['balloon_step'] = []
        data['ep_balloon_step'] = [[] for i in range(num_processes)]
        for key in keys:
            data[key] = []
            data[f'ep_{key}'] = [[] for i in range(num_processes)]
        
    if len(data['ep_balloon_means']) == 0:
        data['ep_balloon_means'].append(
            data_inputs['envs'].get_attr('balloon_mean_sizes')
        )

    if 'info' in data_inputs:
        infos = data_inputs['info']
        for i in range(len(infos)):
            info = infos[i]
            if 'bart_finished' in info and info['bart_finished']:
                for key in keys:
                    data[f'ep_{key}'][i].append(info[key])
                data['ep_balloon_step'][i].append(data_inputs['step'])
            
    if stack:
        for key in keys:
            data[key].append(data[f'ep_{key}'])
            data[f'ep_{key}'] = [[] for i in range(num_processes)]
        data['balloon_means'].append(
            data['ep_balloon_means'][0]
        )
        data['balloon_step'].append(data['ep_balloon_step'])
        data['ep_balloon_means'] = []
        data['ep_balloon_step'] = [[] for i in range(num_processes)]

    return data


def reshape_parallel_evalu_res(res, meta_balloons=None):
    '''Reshape parallel evaluation results to be episode-wise as we would
    expect running the experiment in sequence
    
    For now, just simply take the first episode from each process
    and don't worry about resets. This is made assuming that each process
    is run for one episode and we're just doing it for parallel
    
    meta_balloons: if passed, truncate all data to the number of balloons
        used specifically for meta bart trials
    '''

    keys = ['obs', 'actions', 'action_log_probs', 'action_probs',
            'rewards', 'rnn_hxs', 'dones', 'masks',
            'values']
    # 'activations'
    # 'auxiliary_truths', 'auxiliary_preds', 'actor_features'
    new_res = {
        k: [] for k in keys
    }
    for proc in range(len(res['dones'][0])):
        done = res['dones'][0][proc]
        idx = np.argmax(done)
        for k in keys:
            new_res[k].append(res[k][0][proc][:idx])
    
    if meta_balloons is not None:
        new_res['data'] = {}
        new_res['data']['balloon_means'] = res['data']['balloon_means'][0]
        new_res['data']['current_color'] = [d[:meta_balloons] for d in res['data']['current_color'][0]]
        new_res['data']['last_size'] = [d[:meta_balloons] for d in res['data']['last_size'][0]]
        new_res['data']['balloon_limit'] = [d[:meta_balloons] for d in res['data']['balloon_limit'][0]]
        new_res['data']['inflate_delay'] = [d[:meta_balloons] for d in res['data']['inflate_delay'][0]]
        new_res['data']['popped'] = [d[:meta_balloons] for d in res['data']['popped'][0]]
        new_res['data']['balloon_step'] = [d[:meta_balloons] for d in res['data']['balloon_step'][0]]
    else:
        new_res['data'] = res['data']

    if 'activations' in res:
        activ_types = ['shared', 'actor', 'critic']
        
        new_res['activations'] = {}
        for activ_type in activ_types:
            num_layers = len(res['activations'][0][0][f'{activ_type}_activations'])
            for layer in range(num_layers):
                activ = []                
                for proc in range(len(res['dones'][0])):
                    done = res['dones'][0][proc]
                    idx = np.argmax(done)
                    a = []
                    for step in range(idx):
                        a.append(res['activations'][0][step][f'{activ_type}_activations'][layer][proc])
                    activ.append(torch.vstack(a))
                    
                new_res['activations'][f'{activ_type}{layer}'] = activ
                    
    return new_res
            
        
        
def reshape_activations(res, inplace=True):
    activations = {}
    activ_types = ['shared', 'actor', 'critic']
    num_eps = len(res['dones'])
    
    for activ_type in activ_types:
        num_layers = len(res['activations'][0][0][f'{activ_type}_activations'])
        for layer in range(num_layers):
            for ep in range(num_eps):
                steps = len(res['dones'][ep])
                activ = []
                a = []
                for step in range(steps):
                    a.append(res['activations'][ep][step][f'{activ_type}_activations'][layer])
                activ.append(torch.vstack(a))
            activations[f'{activ_type}{layer}'] = activ
        
    if inplace:
        res['activations'] = activations
    else:
        return activations


def bart_color_n_callback(data_inputs={}, data={}, first=False, stack=False):
    keys = ['current_color', 'last_size', 'balloon_limit', 'inflate_delay', 'popped', 'passive']
    # print(data_inputs)
    if len(data) == 0:
        for key in keys:
            data[key] = []
            data[f'ep_{key}'] = []
        data['balloon_step'] = []
        data['ep_balloon_step'] = []
        
    if 'info' in data_inputs:
        info = data_inputs['info'][0]
        if 'bart_finished' in info and info['bart_finished']:
            for key in keys:
                data[f'ep_{key}'].append(info[key])
            data['ep_balloon_step'].append(data_inputs['step'])
        
        if 'done' in data_inputs and data_inputs['done']:
            for key in keys:
                data[key].append(data[f'ep_{key}'])
                data[f'ep_{key}'] = []
            data['balloon_step'].append(data['ep_balloon_step'])
            data['ep_balloon_step'] = []

    return data
'''
================================================================
Load checkpoint functions
================================================================
'''
# get checkpoints in folder
def get_chks(exp_name, trial=None, subdir='shortcut_resets', basedir='../saved_checkpoints'):
    base = Path(basedir)/subdir
    if trial is not None:
        chk_folder = base/f'{exp_name}_t{trial}'
    else:
        chk_folder = base/exp_name
    
    chks = []
    for i in chk_folder.iterdir():
        chks.append(int(i.name.split('.pt')[0]))
    chks = np.sort(chks)
    return chks

def load_chk(exp_name, chk, trial=None, subdir='shortcut_resets', basedir='../saved_checkpoints'):
    base = Path(basedir)/subdir
    if trial is not None:
        chk_folder = base/f'{exp_name}_t{trial}'
    else:
        chk_folder = base/exp_name

    available_chks = [f.name for f in chk_folder.iterdir()]
    if f'{chk}.pt' not in available_chks:
        print('chk {chk} is not available.')
        print('List of available checkpoints:')
        chks = np.array([int(f.name.split('.pt')[0]) for f in p.iterdir()])
        chks.sort()
        print(chks)
        return None, None, None
    
    model_path = chk_folder/f'{chk}.pt'
    model, (obs_rms, ret_rms) = torch.load(model_path)
    
    return model, obs_rms, ret_rms



def print_trained_models(folder='../saved_models/', ignore_non_pt=True,
                        ignore_non_dir_in_first=True, ret=False, exclude=1):
    '''
    Read the trained_models folder to see what models have been trained
    ignore_non_pt: don't print files without .pt extension
    ignore_non_dir_in_first: don't print files in the parent folder, skip straight to directories
    exclude:
        1: don't print any grid nav or visible platform trials
    '''
    
    space =  '    '
    branch = '│   '
    tee =    '├── '
    last =   '└── '
    
    ignore_dirs = []
    if exclude >= 1:
        ignore_dirs += ['basics']
    
    path = Path(folder)
    print(path.name)
        
    def inner_print(path, depth, ignore_non_pt=ignore_non_pt, ignore_non_dir_in_first=ignore_non_dir_in_first):
        directories = []
        unique_experiments = {}
        original_experiment_names = {}
        for d in path.iterdir():
            if d.is_dir() and d.name not in ignore_dirs:
                directories.append(d)
            elif d.suffix == '.pt':
                if not re.match('.*\d*\.pt', d.name) and not (ignore_non_dir_in_first and depth == 0):
                    #not a trial, simply print
                    print(branch*depth+tee+d.name)
                exp_name = '_'.join(d.name.split('_')[:-1])
                if exp_name in unique_experiments.keys():
                    unique_experiments[exp_name] += 1
                else:
                    unique_experiments[exp_name] = 1
                    original_experiment_names[exp_name] = d.name
            elif not ignore_non_pt:
                print(branch*depth+tee+d.name)
        for key, value in unique_experiments.items():
            if ignore_non_dir_in_first and depth == 0:
                break
            if value > 1:
                print(branch*depth + tee+'EXP', key + ':', value)
            else:
                print(branch*depth+tee+original_experiment_names[key])
        
        result_dict = unique_experiments.copy()
        for i, d in enumerate(directories):
            print(branch*depth + tee+d.name)
            sub_experiments = inner_print(d, depth+1, ignore_non_pt, ignore_non_dir_in_first)
            result_dict[d] = sub_experiments
        
        return result_dict
            
    directory_dict = inner_print(path, 0, ignore_non_pt, ignore_non_dir_in_first)    
    if ret:
        return directory_dict
    
        
    

def get_ep(data, ep_ends, ep_num=0):
    '''
    Pass data and a data block to grab data from this episode n alone
    E.g., call get_ep(hidden_states, 1) to get the hidden states for the 2nd episode
    '''
    if ep_num == 0:
        ep_start = 0
    else:
        ep_start = ep_ends[ep_num - 1]
    
    ep_end = ep_ends[ep_num]
    return data[ep_start:ep_end]

    
    
'''
================================================================
Old navigation functions, may be useful to adapt
================================================================
''' 
    
def animate_episode(ep_num=0, trajectory=False):
    #generate frames of episode
    rgb_array = []
    agent = get_ep(data['agent'], ep_num)
    goal = get_ep(data['goal'], ep_num)[0]
    for i in range(1, env.world_size[0]-1):
        for j in range(1, env.world_size[1]-1):
            env.objects[i, j] = 0
            env.visible[i, j] = 0
    env.objects[goal[0], goal[1]] = 1
    env.visible[goal[0], goal[1]] = 6

    for a in agent:
        env.agent[0][0] = a[0][0]
        env.agent[0][1] = a[0][1]
        env.agent[1] = a[1]
        rgb_array.append(env.render('rgb_array'))

    rgb_array = np.array(rgb_array)
    
    if trajectory:
        #generate trajectory
        trajectory = get_ep(all_trajectories, ep_num)
        scat_min = np.min(all_trajectories)
        scat_max = np.max(all_trajectories)


        fig, ax = plt.subplots(1, 2, figsize=(10,5))
        
        im = ax[0].imshow(rgb_array[0,:,:,:])
        x, y = np.full(trajectory.shape[0], -100.0), np.full(trajectory.shape[0], -100.0)
        scatter = ax[1].scatter(x, y)
        ax[1].set_xlim([scat_min, scat_max])
        ax[1].set_ylim([scat_min, scat_max])

        plt.close() # this is required to not display the generated image

        def init():
            im.set_data(rgb_array[0,:,:,:])
            scatter.set_offsets(np.c_[x, y])

        def animate(i):
            im.set_data(rgb_array[i,:,:,:])
            x[:i+1] = trajectory.T[0][:i+1]
            y[:i+1] = trajectory.T[1][:i+1]
            scatter.set_offsets(np.c_[x, y])
            # print(np.c_[x, y])
            return im, scatter,

        anim = animation.FuncAnimation(fig, animate, init_func=init, frames=rgb_array.shape[0],
                                       interval=500)
    else:
        fig = plt.figure(figsize=(6,6))
        
        im = plt.imshow(rgb_array[0,:,:,:])
        plt.close() # this is required to not display the generated image

        def init():
            im.set_data(rgb_array[0,:,:,:])

        def animate(i):
            im.set_data(rgb_array[i,:,:,:])
            return im

        anim = animation.FuncAnimation(fig, animate, init_func=init, frames=rgb_array.shape[0],
                                       interval=500)

    # fig = plt.figure()
    return HTML(anim.to_html5_video())

    
        
def get_activations(model, results):
    """
    OBSOLETE: evalu can now collect activations during episode
    
    After running evalu, pass the model and results
    to collect the layer activations at each timestep during experimentation
    Assuming the model is a FlexBase network

    Args:
        model (FlexBaseNN): loaded policy and model
        results (dict): dictionary from evalu() function

    Returns:
        activations: dictionary with activations from each layer
    """
    obs = torch.vstack(results['obs'][:])
    hidden_state = torch.vstack(results['hidden_states'][:])
    masks = torch.vstack(results['masks'][:])
    
    activations = model.base.forward_with_activations(obs, hidden_state, 
                                                    masks, deterministic=True)
    
    return activations


def activation_testing(model, env):
    vis_walls = env.vis_walls
    vis_wall_refs = env.vis_wall_refs
    
    env.character.pos = np.array([10, 10])
    env.character.angle = 0
    env.character.update_rays(vis_walls, vis_wall_refs)
    
    obs = torch.tensor(env.get_observation(), dtype=torch.float32).view(1, -1)
    rnn_hxs = torch.zeros(1, model.recurrent_hidden_state_size, dtype=torch.float32)
    masks = torch.zeros(1, 1, dtype=torch.float32)
    
    activations = model.base.forward_with_activations(obs, rnn_hxs, masks)
    
    
    
    
    
def determine_linear_fit(layer_activations, quantity):
    """Perform a linear regression between a layer's activations
    and some quantity to see if it could be linearly encoded
    within the weights

    Args:
        layer_activations (array: (samples, layer_size)): 
            activations collected over samples running
        quantity (array: (samples, quantity_size)): 
            quantity desired to check fit of
    """
    reg = LinearRegression()
    reg.fit(layer_activations, quantity)
    prediction = reg.predict(layer_activations)
    
    print(f'r2 score: {r2_score(prediction, quantity)}')
    print(f'MSE: {mean_squared_error(prediction, quantity)}')
    
    return reg


def plot_2d_activations(layer_activations, quantity, min_val=0, max_val=300,
                        grid_samples=300, sigma=10):
    """Attempt to fit a gradient boosted tree that takes a 2D
    quantity and fits it with layer activations
    Designed to see if we can predict a node activation based on position

    Args:
        layer_activations (array: (samples, layer_size)): 
            activations collected over samples running
        quantity (array: (samples, 2)): 
            2D quantity to fit and plot against
        min: min value to make grid
        max: max value to make grid
    """
    num_samples = layer_activations.shape[0]
    num_nodes = layer_activations.shape[1]
    plot_size = int(np.ceil(np.sqrt(num_nodes)))
    
    regressors = []
    
    fig, ax = plt.subplots(plot_size, plot_size, figsize=(20, 20),
                           sharex=True, sharey=True)
    grid = np.zeros((grid_samples * 2, grid_samples)).reshape(-1, 2)
    
    xs = np.linspace(min_val, max_val, grid_samples)
    ys = np.linspace(min_val, max_val, grid_samples)
    
    for i, y in enumerate(ys):
        grid[i*grid_samples:(i+1)*grid_samples] = np.concatenate(
            [xs.reshape((-1, 1)), np.full((grid_samples, 1), y)], axis=1
        )
        
    for i in range(num_nodes):
        ax_x = i % plot_size
        ax_y = i // plot_size
        
        model = xgb.XGBRegressor(max_depth=2)
        y = layer_activations[:, i]
        model.fit(quantity, y)
        regressors.append(model)
        
        grid_activations = model.predict(grid).reshape(grid_samples, grid_samples)
        smoothed = gaussian_filter(grid_activations, sigma=sigma)
        ax[ax_x, ax_y].imshow(smoothed)
        ax[ax_x, ax_y].set_xticks([])
        ax[ax_x, ax_y].set_yticks([])
    
    plt.tight_layout()
    
    return regressors
        
        
def generate_model_and_test(base='FlexBase', env='NavEnv-v0', env_kwargs={}):
    env = gym.make(env, **env_kwargs)
    obs = env.reset()
    action_space = env.action_space
    model = Policy(obs.shape, action_space, base=base)
    
    