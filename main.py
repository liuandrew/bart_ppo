import copy
import glob
import os
import time
from collections import deque
from shutil import copyfile
from pathlib import Path

import gym
import gym_nav
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from ppo import algo, utils
from ppo.algo.ppo import PPOAux
from ppo.algo import gail
from ppo.arguments import get_args
from ppo.envs import make_vec_envs
from ppo.model import Policy, norm_scale_parameters
from ppo.storage import RolloutStorage, RolloutStorageAux
from evaluation import evaluate

from scheduler import archive_config_file


def main():
    args = get_args()

    #Andy: setup W&B and tensorboard
    if args.exp_name is not None:
        run_name = f"{args.exp_name}__{int(time.time())}"
    else:
        run_name = f"{args.env_name}__{args.seed}__{int(time.time())}"

    print(run_name) #print run name for logging
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
        logdir = wandb.run.dir
        writer = SummaryWriter(logdir)
    else:
        writer_path = Path('runs/' + args.save_dir)
        writer_path.mkdir(exist_ok=True, parents=True)
        writer_path = writer_path/run_name

        writer = SummaryWriter(writer_path)

    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    log_dir = os.path.expanduser(args.log_dir)
    eval_log_dir = log_dir + "_eval"
    utils.cleanup_log_dir(log_dir)
    utils.cleanup_log_dir(eval_log_dir)
    
    #Perform some checks for new auxiliary task trainers
    if args.use_new_aux == True:
        print('New Auxiliary training methods')
        print('auxiliary truth sizes: ' + str(args.auxiliary_truth_sizes))
        
        if 'auxiliary_heads' in args.nn_base_kwargs:
            print('auxiliary heads: ' + str(args.nn_base_kwargs['auxiliary_heads']))
            if len(args.auxiliary_truth_sizes) != len(args.nn_base_kwargs['auxiliary_heads']):
                raise Exception(f'number of auxiliary_truth_sizes {len(args.auxiliary_truth_sizes)} should be equivalent to number of auxiliary heads')
        else:
            if len(args.auxiliary_truth_sizes) > 0:
                raise Exception(f'number of auxiliary_truth_sizes {len(args.auxiliary_truth_sizes)} should be equivalent to number of auxiliary heads - no auxiliary heads given')

    if args.nn_base != 'FlexBaseAux':
        print('WARNING: nn_base should probably be FlexBaseAux for new aux methods')

    if args.save_name is not None:
        save_path = Path('saved_models/' + args.save_dir)
        save_path.mkdir(exist_ok=True, parents=True)
        save_path = save_path/f'{args.save_name}.pt'

    # Andy: generate path for saving checkpoints
    if args.checkpoint_interval > 0:
        chk_folder = Path('saved_checkpoints/' + args.save_dir)/args.save_name
        chk_folder.mkdir(exist_ok=True, parents=True)    


    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")
    print('initializing environments')
    #Andy: Add option to turn off environment normalization
    print('Normalize Env:', args.normalize_env)
    envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                         args.gamma, args.log_dir, device, False, capture_video=args.capture_video,
                         env_kwargs=args.env_kwargs,
                         **args.aux_wrapper_kwargs)
    if not args.normalize_env:
        envs.eval()
        
    loaded_model = False
    # print(args.cont)
    print('initializing model')
    print(args.save_interval)
    
    #Andy: for continuing an experiment, args.cont is True
    if args.cont:
        loaded_model = True
        if args.cont_file_name is not None:
            print('Loading', args.cont_file_name)
            actor_critic, obs_rms = torch.load(args.cont_file_name)
        else:
            print('Loading', save_path)
            actor_critic, obs_rms = torch.load(save_path)
    
    if not loaded_model:
        actor_critic = Policy(
            envs.observation_space.shape,
            envs.action_space,
            base=args.nn_base,
            base_kwargs={'recurrent': args.recurrent_policy, 
                         **args.nn_base_kwargs})
        
        #Andy: if cloning parameters, do it here. We are assuming
        #that the target cloning network has the appropriate parameter
        #sizes
        if args.clone_parameter_experiment:
            clone_args = args.clone_args
            clone_actor_critic, clone_obs_rms = torch.load(clone_args['clone_path'])
            
            # Named layers to clone
            clone_layers = clone_args['clone_layers'].split(',')
            # Named layers to freeze
            if 'freeze_layers' in clone_args:
                freeze_layers = clone_args['freeze_layers'].split(',')
            else:
                freeze_layers = []
                
            
            
            for name in clone_layers:
                copy_params = list(getattr(clone_actor_critic.base, name).parameters())
                paste_params = list(getattr(actor_critic.base, name).parameters())
                print(f'Cloning layer {name}')
                for i in range(len(copy_params)):
                    paste_params[i].data.copy_(copy_params[i].data)
                    
            if 'norm_layers' in clone_args and clone_args['norm_layers']:
                norm_scale_parameters(actor_critic.base, clone_layers)
                    
            for name in freeze_layers:
                print(f'Freezing layer {name}')                
                params = list(getattr(actor_critic.base, name).parameters())
                for i in range(len(params)):
                    params[i].requires_grad = False

            if clone_args['copy_obs_rms']:
                envs.obs_rms = clone_obs_rms
            if clone_args['freeze_obs_rms']:
                # envs.training = False
                envs.eval()
                
            print(f'envs.training {envs.training}')
            print(f'envs.obs_rms.mean {envs.obs_rms.mean}')
            
                
        actor_critic.to(device)

    print('initializing algo')
    if args.algo == 'a2c':
        agent = algo.A2C_ACKTR(
            actor_critic,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            alpha=args.alpha,
            max_grad_norm=args.max_grad_norm)
    elif args.algo == 'ppo':
            agent = PPOAux(
                actor_critic,
                args.clip_param,
                args.ppo_epoch,
                args.num_mini_batch,
                args.value_loss_coef,
                args.entropy_coef,
                args.auxiliary_loss_coef,
                lr=args.lr,
                eps=args.eps,
                max_grad_norm=args.max_grad_norm,
                remove_actor_grads_on_shared=args.remove_actor_grads_on_shared)
              
    rollouts = RolloutStorageAux(args.num_steps, args.num_processes,
                            envs.observation_space.shape, envs.action_space,
                            actor_critic.recurrent_hidden_state_size,
                            actor_critic.auxiliary_output_sizes, #note sizes not size here
                            auxiliary_truth_sizes=args.auxiliary_truth_sizes)

    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=10)
    ep_bonus_reward = [0]*args.num_processes
    
    universal_step_reset_point = 0
    start_update_step = 0
    
    start = time.time()
    #Andy: add global step
    if args.cont:
        global_step = int(obs_rms.count)
        start_update_step = int(obs_rms.count / (args.num_steps * args.num_processes))
        print('Continuing from update step', start_update_step)
    else:
        global_step = 0
    # print(global_step)
    num_updates = int(
        args.num_env_steps) // args.num_steps // args.num_processes
    for j in range(start_update_step, num_updates):
        
        # This parameter gives a way to control environment settings as
        #  a function of time steps trained.
        # To implement, simply add a def set_universal_step(self, global_step)
        #  function to the env, have it save it to a self.global_step variable
        #  and respond to the global step in some way
        # The paramemter added should be a dictionary containing 'on': True,
        #  'schedule': list of steps to send update to environment on
        # 
        if args.use_universal_step['on']:
            schedule = np.array(args.use_universal_step['schedule'])
            
            if 'step_args' in args.use_universal_step:
                # Method 1: pass arguments on the update that are defined in
                #  'args' of use_universal_step parameter, send to set_universal_arg
                #  These will be dictionary kw args
                step_args = args.use_universal_step['step_args']
                b = np.argwhere(global_step >= schedule).reshape(-1)
                if len(b) == 0:
                    idx = 0
                else:
                    idx = b[-1]
                if idx > universal_step_reset_point:
                    print(idx, step_args[idx])
                    universal_step_reset_point = idx
                    
                    if type(step_args[idx]) == dict:
                        envs.env_method('set_universal_arg', **step_args[idx])
                    else:
                        envs.env_method('set_universal_arg', step_args[idx])
                        
            else:
                # Method 2: just pass global_step and let env_kwargs handle it
                b = np.argwhere(global_step >= schedule).reshape(-1)
                if len(b) == 0:
                    idx = 0
                else:
                    idx = b[-1]
                if idx > universal_step_reset_point:
                    # perform reset 
                    universal_step_reset_point = idx
                    envs.env_method('set_universal_step', global_step)


        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            utils.update_linear_schedule(
                agent.optimizer, j, num_updates,
                agent.optimizer.lr if args.algo == "acktr" else args.lr)

        #1. Collect args.num_steps * args.num_processes number of experience steps
        #from the environment
        for step in range(args.num_steps):
            #Andy: add global step
            global_step += 1 * args.num_processes
            # Sample actions
            
            #Andy: Update to use outputs dict
            with torch.no_grad():
                # value, action, action_log_prob, recurrent_hidden_states, auxiliary_preds = \
                # actor_critic.act(rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                #     rollouts.masks[step])
                outputs = actor_critic.act(rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step])
                action = outputs['action']
                value = outputs['value']
                action_log_prob = outputs['action_log_probs']
                recurrent_hidden_states = outputs['rnn_hxs']
                auxiliary_preds = outputs['auxiliary_preds']



            # Obser reward and next obs
            # obs, reward, done, infos = envs.step(action)
            obs, reward, done, infos = envs.step(action)
            
            auxiliary_truths = [[] for i in range(len(actor_critic.auxiliary_output_sizes))]
            for n, info in enumerate(infos):
                if 'bonus_reward' in info:
                    ep_bonus_reward[n] += info['bonus_reward']
                if 'auxiliary' in info and len(info['auxiliary']) > 0:
                    for i, aux in enumerate(info['auxiliary']):
                        auxiliary_truths[i].append(aux)
            if len(auxiliary_truths) > 0:
                auxiliary_truths = [torch.tensor(np.vstack(aux)) for aux in auxiliary_truths]
            
            for n, info in enumerate(infos):
                if 'episode' in info.keys():
                    # This marks episode done, record to writer
                    episode_rewards.append(info['episode']['r'])
                    print(f'global_step={global_step}')
                    # Andy: add tensorboard writing episode returns
                    writer.add_scalar("charts/episodic_return", info["episode"]["r"], 
                        global_step)
                    writer.add_scalar("charts/episodic_length", info["episode"]["l"], 
                        global_step)
                    writer.add_scalar("charts/episodic_bonus_rewards", ep_bonus_reward[n])
                    
                    ep_bonus_reward[n] = 0



            # If done then clean the history of observations.
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos])
            rollouts.insert(obs, recurrent_hidden_states, action,
                            action_log_prob, value, reward, masks, bad_masks,
                            auxiliary_preds, auxiliary_truths)

        #2. Compute rewards and update parameters with policy improvement
        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()

        rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda, args.use_proper_time_limits)
        
        if args.algo == 'ppo':
            value_loss, action_loss, dist_entropy, approx_kl, clipfracs, \
                auxiliary_loss = agent.update(rollouts)
        else:
            value_loss, action_loss, dist_entropy = agent.update(rollouts)

        rollouts.after_update()

        #Andy: add tensorboard data tracking
        writer.add_scalar("charts/learning_rate", agent.optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", value_loss, global_step)
        writer.add_scalar("losses/policy_loss", action_loss, global_step)
        writer.add_scalar("losses/auxiliary_loss", auxiliary_loss, global_step)
        writer.add_scalar("losses/entropy", dist_entropy, global_step)

        if args.algo == 'ppo':
            writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        # writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start)), global_step)

        # save for every interval-th episode or for the last epoch
        if (j % args.save_interval == 0
                or j == num_updates - 1) and args.save_dir != "":

            # Andy: change save path to use more args
            torch.save([
                actor_critic,
                getattr(utils.get_vec_normalize(envs), 'obs_rms', None)
            ], save_path)

        # Andy: if checkpointing, save every interval-th episode
        # Note that 0th update is actually the 1st update, because this comes after update code
        if args.checkpoint_interval > 0 and (j % args.checkpoint_interval == 0 or j == num_updates - 1):
            chk_path = chk_folder/f'{j}.pt'
            torch.save([
                actor_critic,
                getattr(utils.get_vec_normalize(envs), 'obs_rms', None)
            ], chk_path)

        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            end = time.time()
            print(
                "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n"
                .format(j, total_num_steps,
                        int(total_num_steps / (end - start)),
                        len(episode_rewards), np.mean(episode_rewards),
                        np.median(episode_rewards), np.min(episode_rewards),
                        np.max(episode_rewards), dist_entropy, value_loss,
                        action_loss))



        if (args.eval_interval is not None and len(episode_rewards) > 1
                and j % args.eval_interval == 0):
            obs_rms = utils.get_vec_normalize(envs).obs_rms
            evaluate(actor_critic, obs_rms, args.env_name, args.seed,
                     args.num_processes, eval_log_dir, device)


    if args.config_file_name is not None:
        archive_config_file(args.config_file_name)
        print('Experiment completed, experiment log updated')


if __name__ == "__main__":
    main()
