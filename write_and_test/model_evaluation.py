import gym
import gym_nav
import torch
import sys
sys.path.insert(0, '..')
from evaluation import evaluate
from ppo.model import Policy
from ppo import utils
from ppo.envs import make_vec_envs
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pickle
from pathlib import Path
import re
from matplotlib import animation
from IPython.display import HTML

from scipy.ndimage.filters import gaussian_filter
import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

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


def print_trained_models(folder='../trained_models/ppo/', ignore_non_pt=True,
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
        ignore_dirs += ['invisible_poster', 'invisible_shared', 'invisible_wallcolors',
                        'nav_visible_reshaping', 'visible_reshaping', 'visible_wallcolors']
    
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
    


def load_model_and_env(experiment, trial_num=None, env_name=None,
                       base_folder='../trained_models/ppo/'):
    '''
    Args: 
        experiment (str): Name of experiment, including subfolder. E.g.
            'invisible_poster/invisible_poster_shape_0'
            If not passing a trial_num, can directly pass the exact file. E.g.
            'invisible_poster/invisible_poster_shape_0_0.pt'
        trial_num (int, optional): Trial number. If passed, find an appropriate file with
            the given trial number. The file may end with just the number (e.g. '_0.pt') or
            with the prefix t (e.g. '_t0.pt')
        env_name (str, optional): Name of environment. If passed, return the actual
            environment instead of env_kwargs. Defaults to 'NavEnv-v0' (continuous nav environment)
        base_folder (str, optional): Base folder where experiments are saved to
    
    Load a model along with its environment
    Use the fact that all saved models end with _i where i indicates the trial number
    
    For scenario name, include any folder directions starting from trained_models/ppo/
    E.g., for the poster scenario might use scenario_name = 'invisible_poster/invisible_poster_0_shape_0_0.pt'
    
    Note that after training has been completed, we need to manually go back to write_experiments.ipynb
    to add the appropriate env kwargs to the trained models folder
    
    Return: model, obs_rms, env_kwargs (or env)
    '''
    
    if '/' in experiment:
        sub_folders = '/'.join(experiment.split('/')[:-1])
        folder = Path(base_folder)/sub_folders
        experiment = experiment.split('/')[-1]
    else:
        folder = Path(base_folder)
    
    
    #Find the corresponding environment kwargs file
    if trial_num is None:
        env_file = '_'.join(experiment.split('_')[:-1]) + '_env'
    else:
        env_file = experiment + '_env'
        
    if trial_num is None:
        file = Path(folder)/experiment
    else:
        # try to find a file with either _0.pt or _t0.pt (if trial_num = 0 for example)
        file = Path(folder)/(f'{experiment}_{trial_num}.pt')
        if not file.exists():
            file = Path(folder)/(f'{experiment}_t{trial_num}.pt')
            if not file.exists():
                raise Exception(f'{experiment} with _{trial_num}.pt and _t{trial_num}.pt suffixes do not exist.')
        
        
    try:
        env_kwargs = pickle.load(open(folder/env_file, 'rb'))
    except:
        raise Exception(f'Error loading env_kwargs for {folder/env_file}, make sure that they have been saved from write_experiments.ipynb')
    # env = gym.make('Gridworld-v0', **env_kwargs)
    
    model, obs_rms = torch.load(file)
    
    
    if env_name is not None:
        env = gym.make(env_name, **env_kwargs)
        return model, obs_rms, env
    else:
        return model, obs_rms, env_kwargs


def load_checkpoint(experiment, trial=None, checkpoint=None, base_folder='../trained_models/checkpoint'):
    '''
    Load a saved model checkpoint, and if no checkpoint is given,
    print out possible checkpoint options
    '''
    folder = Path(base_folder)
    if trial != None:
        experiment = experiment + f'_t{trial}'
    path = folder/experiment
    
    checkpoints = [f.name.split('.pt')[0] for f in path.iterdir() if '.pt' in f.name]
    if len(checkpoints) == 0:
        raise Exception('No checkpoints found in the given folder')
    
    if checkpoint == None:
        print('Available checkpoints:', checkpoints)
    else:
        checkpoint = str(checkpoint)
        checkpoint = checkpoint.split('.pt')[0]
        if checkpoint not in checkpoints:
            raise Exception('Given checkpoint does not exist')
        else:
            return torch.load(path/f'{checkpoint}.pt')
        
        
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

    model_path = chk_folder/f'{chk}.pt'
    model, obs_rms = torch.load(model_path)
    
    return model, obs_rms
    


def evalu(model, obs_rms, n=100, env_name='NavEnv-v0', env_kwargs={},
          data_callback=None, capture_video=False, verbose=0, with_activations=False,
          seed=1, deterministic=True, normalize=True, aux_wrapper_kwargs={}, new_aux=False,
          auxiliary_truth_sizes=[], eval_log_dir='/tmp/gym/_eval'):
    '''
    Evaluate using the current global model, obs_rms, and env_kwargs
    Load ep_ends, ep_lens into global vars to be used by get_ep
        as well as all_obs, all_actions, all_rewards, all_hidden_states, all_dones, eval_envs, data
        
    capture_video: whether video should be captured
    verbose: 
        0: no summary prints
        1: print mean reward for all episodes
        2: print rewards for each episode
    '''
    
    results = evaluate(model, obs_rms, env_name, seed, 1, eval_log_dir, device, 
         env_kwargs=env_kwargs, data_callback=data_callback, num_episodes=n, capture_video=capture_video,
         verbose=verbose, with_activations=with_activations, deterministic=deterministic,
         normalize=normalize, aux_wrapper_kwargs=aux_wrapper_kwargs,
         new_aux=new_aux, auxiliary_truth_sizes=auxiliary_truth_sizes)

    #ep_ends and ep_lens used to easily pull data for single episodes
    results['ep_ends'] = np.where(np.array(results['dones']).flatten())[0]
    results['lens'] = np.diff(results['ep_ends'])
    
    # env = gym.make(env_name, **env_kwargs)

    #code for computing PCA of hidden_states over all evaluated episodes
    
    # global all_trajectories, hidden_states
    
    # hidden_states = np.zeros((len(all_hidden_states), all_hidden_states[0].shape[1]))
    # for i in range(len(all_hidden_states)):
    #     hidden_states[i] = all_hidden_states[i].numpy()[0]

    # pca = PCA(n_components=2)
    # pca.fit(hidden_states)

    # all_trajectories = pca.transform(hidden_states)

    # len(pca.transform(hidden_states))
    
    return results


    
    
def forced_action_evaluate(actor_critic, obs_rms=None, normalize=True, forced_actions=None,
            env_name='NavEnv-v0', seed=None, num_processes=1,
             device=torch.device('cpu'), ret_info=1, capture_video=False, env_kwargs={}, data_callback=None,
             num_episodes=10, verbose=0, with_activations=False, deterministic=True,
             aux_wrapper_kwargs={}, auxiliary_truth_sizes=[],
             eval_log_dir=None, video_folder='./video', with_aux=False):
    '''
    ret_info: level of info that should be tracked and returned
    capture_video: whether video should be captured for episodes
    env_kwargs: any kwargs to create environment with
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

    envs = make_vec_envs(env_name, seed + num_processes, num_processes,
                              None, eval_log_dir, device, True, 
                              capture_video=capture_video, 
                              env_kwargs=env_kwargs, normalize=normalize,
                              video_folder=video_folder,
                              **aux_wrapper_kwargs)

    vec_norm = utils.get_vec_normalize(envs)
    if vec_norm is not None:
        vec_norm.eval()
        vec_norm.obs_rms = obs_rms

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
    rnn_hxs = torch.zeros(
        num_processes, actor_critic.recurrent_hidden_state_size, device=device)
    masks = torch.zeros(num_processes, 1, device=device)

    for ep in range(num_episodes):
        step = 0
        while True:
            ep_obs.append(obs)
            ep_rnn_hxs.append(rnn_hxs)
            if data_callback is not None and step == 0:
                data = data_callback(None, envs, rnn_hxs,
                    obs, [], [], [False], data, first=True)

            with torch.no_grad():
                
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
                data = data_callback(None, envs, rnn_hxs,
                    obs, action, reward, done, data)
            else:
                data = {}
            
            auxiliary_truths = [[] for i in range(len(actor_critic.auxiliary_output_sizes))]
            if with_aux:
                for info in infos:
                    if 'auxiliary' in info and len(info['auxiliary']) > 0:
                        for i, aux in enumerate(info['auxiliary']):
                            auxiliary_truths[i].append(aux)
                if len(auxiliary_truths) > 0:
                    auxiliary_truths = [torch.tensor(np.vstack(aux)) for aux in auxiliary_truths]
            ep_auxiliary_truths.append(auxiliary_truths)
            
            
            # for info in infos:
            #     if 'episode' in info.keys():
            #         eval_episode_rewards.append(info['episode']['r'])
            #         #Andy: add verbosity option
            #         if verbose >= 2:
            #             print('ep ' + str(len(eval_episode_rewards)) + ' rew ' + \
            #                 str(info['episode']['r']))
            
            step += 1
            
            if done[0]:
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
                    data = data_callback(None, envs, rnn_hxs,
                        obs, action, reward, done, data, stack=True)
                          
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
    
    
    
def hallucination_evaluate(actor_critic, obs_rms=None, normalize=True, forced_actions=None,
            env_name='NavEnv-v0', seed=None, num_processes=1,
             device=torch.device('cpu'), ret_info=1, capture_video=False, env_kwargs={}, data_callback=None,
             num_episodes=10, verbose=0, with_activations=False, deterministic=True,
             aux_wrapper_kwargs={}, auxiliary_truth_sizes=[],
             eval_log_dir=None, video_folder='./video', hallucinator=None):
    '''
    ret_info: level of info that should be tracked and returned
    capture_video: whether video should be captured for episodes
    env_kwargs: any kwargs to create environment with
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
    hallucinator: an object with hallucination check and hallucinations passed
    '''

    if seed is None:
        seed = np.random.randint(0, 1e9)

    envs = make_vec_envs(env_name, seed + num_processes, num_processes,
                              None, eval_log_dir, device, True, 
                              capture_video=capture_video, 
                              env_kwargs=env_kwargs, normalize=normalize,
                              video_folder=video_folder,
                              **aux_wrapper_kwargs)

    vec_norm = utils.get_vec_normalize(envs)
    if vec_norm is not None:
        vec_norm.eval()
        vec_norm.obs_rms = obs_rms

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
    rnn_hxs = torch.zeros(
        num_processes, actor_critic.recurrent_hidden_state_size, device=device)
    masks = torch.zeros(num_processes, 1, device=device)

    for ep in range(num_episodes):
        step = 0
        while True:
            
            # Check for hallucination
            if hallucinator and hallucinator.check_for_hallucination(step, envs, data):
                h_obs = hallucinator.get_hallucination()
                ep_obs.append(h_obs)
                ep_rnn_hxs.append(rnn_hxs)
                if data_callback is not None and step == 0:
                    data = data_callback(None, envs, rnn_hxs,
                        obs, [], [], [False], data, first=True)

                with torch.no_grad():
                    outputs = actor_critic.act(obs, rnn_hxs, 
                                            masks, deterministic=deterministic,
                                            with_activations=with_activations)
                rnn_hxs = outputs['rnn_hxs']
                masks = torch.tensor([[1.0]], dtype=torch.float32, device=device)
                action = None
                reward = 0
                done = [False]
                infos = [{}]
                
                ep_actions.append(action)
                ep_action_log_probs.append(outputs['action_log_probs'])
                ep_action_probs.append(outputs['probs'])
                ep_rewards.append(reward)
                ep_dones.append(done)
                ep_values.append(outputs['value'])
                ep_masks.append(masks)
                ep_actor_features.append(outputs['actor_features'])

            else:            
                ep_obs.append(obs)
                ep_rnn_hxs.append(rnn_hxs)
                if data_callback is not None and step == 0:
                    data = data_callback(None, envs, rnn_hxs,
                        obs, [], [], [False], data, first=True)

                with torch.no_grad():
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
                data = data_callback(None, envs, rnn_hxs,
                    obs, action, reward, done, data)
            else:
                data = {}
                
            auxiliary_truths = [[] for i in range(len(actor_critic.auxiliary_output_sizes))]
            for info in infos:
                if 'auxiliary' in info and len(info['auxiliary']) > 0:
                    for i, aux in enumerate(info['auxiliary']):
                        auxiliary_truths[i].append(aux)
            if len(auxiliary_truths) > 0:
                auxiliary_truths = [torch.tensor(np.vstack(aux)) for aux in auxiliary_truths]
            ep_auxiliary_truths.append(auxiliary_truths)
            
            
            # for info in infos:
            #     if 'episode' in info.keys():
            #         eval_episode_rewards.append(info['episode']['r'])
            #         #Andy: add verbosity option
            #         if verbose >= 2:
            #             print('ep ' + str(len(eval_episode_rewards)) + ' rew ' + \
            #                 str(info['episode']['r']))
            
            step += 1
            
            if done[0]:
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
                    data = data_callback(None, envs, rnn_hxs,
                        obs, action, reward, done, data, stack=True)
                          
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
                
                if hallucinator:
                    hallucinator.reset()
                
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


    
    
# def nav_data_callback(actor_critic, vec_envs, recurrent_hidden_states,
#                                   obs, action, reward, done, data):
#     if data == {}:
#         data['pos'] = []
#         data['angle'] = []
    
#     pos = vec_envs.get_attr('character')[0].pos.copy()
#     angle = vec_envs.get_attr('character')[0].angle
#     data['pos'].append(pos)
#     data['angle'].append(angle)
    
#     return data

def nav_data_callback(actor_critic, vec_envs, recurrent_hidden_states,
                                  obs, action, reward, done, data):
    if data == {}:
        data['pos'] = []
        data['angle'] = []
    
    pos = [c.pos.copy() for c in vec_envs.get_attr('character')]
    angle = [c.angle for c in vec_envs.get_attr('character')]
    data['pos'].append(pos)
    data['angle'].append(angle)
    
    return data
        
def poster_data_callback(actor_critic, vec_envs, recurrent_hidden_states,
                                  obs, action, reward, done, data):
    if data == {}:
        data['pos'] = []
        data['angle'] = []
        data['poster_seen'] = []
        data['poster_in_view'] = []
    
    pos = vec_envs.get_attr('character')[0].pos.copy()
    angle = vec_envs.get_attr('character')[0].angle
    data['pos'].append(pos)
    data['angle'].append(angle)
    
    #Check whether poster has been seen yet
    done = done[0]
    true_obs = vec_envs.get_attr('env')[0].get_observation()
    poster_in_view = False
    if (true_obs == 4/6).any(): #Poster in view
        poster_in_view = True
        
    data['poster_in_view'].append(poster_in_view)
    
    if done == False:
        # print('continue ep')
        # print()
        if len(data['poster_seen']) > 0 and data['poster_seen'][-1] == True:
            # print(data['poster_seen'][-1])
            data['poster_seen'].append(True)
        else:
            data['poster_seen'].append(poster_in_view)
    else:
        # print('new ep')
        if poster_in_view:
            data['poster_seen'].append(True)
        else:
            data['poster_seen'].append(False)
        
        
        
    # if len(data['poster_seen']) > 0 and data['poster_seen'][-1] == True and done == False:
    #     data['poster_seen'].append(True)
    # else:
    #     true_obs = vec_envs.get_attr('env')[0].get_observation()
    #     if (true_obs == 4/6).any(): #Poster in view
    #         data['poster_seen'].append(True)
    #     else:
    #         data['poster_seen'].append(False)
    
    return data
    
    
    
    
def data_callback(actor_critic, vec_envs, recurrent_hidden_states, 
                  obs, action, reward, done, data):
    """Example of a data callback, this basic one is intended
    for 'GridNav-v0'. We take these arguments and can perform
    evaluations on them to collect data during episodes

    Args:
        These arguments will always be passed by the evaluation
        function (evaluation.py)
        data: we pass in the previous dictionary of data

    Returns:
        data: pass back updated dictionary of data
    """
    # Initialize the data dictionary if it has not been initialized yet
    if data == {}:
        data['goal'] = []
        data['agent'] = []
        data['facing_goal'] = []
        data['dist_from_goal'] = []
    
    agent = vec_envs.get_attr('character')[0]
    objects = vec_envs.get_attr('objects')[0]
    
    goal_y = (objects == 2).argmax(axis=0).max()
    goal_x = (objects == 2).argmax(axis=1).max()
    data['goal'].append([goal_y, goal_x])
    
    #check if agent is facing the goal
    correct_directions = np.array([False, False, False, False])
    if goal_y > agent[0][0]:
        correct_directions[1] = True
    if goal_y < agent[0][0]:
        correct_directions[3] = True
    if goal_x > agent[0][1]:
        correct_directions[0] = True
    if goal_x < agent[0][1]:
        correct_directions[2] = True
    
    if correct_directions[agent[1]]:
        data['facing_goal'].append(True)
    else:
        data['facing_goal'].append(False)
        
        
    dist_from_goal = np.abs(goal_y - agent[0][0]) + np.abs(goal_x - agent[0][1])
    data['dist_from_goal'].append(dist_from_goal)
    # vec_envs.env_method('render')
    
    data['agent'].append(agent.copy())
    
    return data

    
    
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
    
    