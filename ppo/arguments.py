import argparse

import torch
import os
import pickle
import json
from ast import literal_eval



#Andy: code used for passing environmental kwargs through command line
def isfloat(str):
    try:
        float(str)
        return True
    except:
        return False

class ParseKwargs(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, dict())
        print(values)
        for value in values:
            key, value = value.split('=')

            if value.isnumeric():
                value = int(value)
            elif value == 'True':
                value = True
            elif value == 'False':
                value = False
            elif value == 'None':
                value = None
            elif isfloat(value):
                value = float(value)
            elif '[' in value:
                value = literal_eval(value)

            getattr(namespace, self.dest)[key] = value
            
class ParseList(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, list())
        print(values)
        # for value in values:
        #     if '[' in value:
        #         value = literal_eval(value)
        #         getattr(namespace, self.dest).append(value)
        for value in values:
            if '[' in value:
                value = literal_eval(value)
            setattr(namespace, self.dest, value)
            
            
def strtobool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')



def get_args():
    parser = argparse.ArgumentParser(description='RL')
    # Algo and a2c/gail params. Pretty much stick to ppo gail has been removed
    parser.add_argument('--algo', default='ppo', help='algorithm to use: a2c | ppo | acktr')
    parser.add_argument('--gail', action='store_true', default=False,
        help='do imitation learning with gail')
    parser.add_argument('--gail-experts-dir', default='./gail_experts', help='directory that contains expert demonstrations for gail')
    parser.add_argument('--gail-batch-size',type=int, default=128, help='gail batch size (default: 128)')
    parser.add_argument('--gail-epoch', type=int, default=5, help='gail epochs (default: 5)')
    parser.add_argument('--num-steps', type=int, default=5,
        help='number of forward steps in A2C (default: 5)')
    
    # RL Training parameters
    parser.add_argument('--lr', type=float, default=7e-4, help='learning rate (default: 7e-4)')
    parser.add_argument('--eps', type=float, default=1e-5, 
        help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument('--alpha', type=float, default=0.99,
        help='RMSprop optimizer apha (default: 0.99)')
    parser.add_argument('--gamma', type=float, default=0.99,
        help='discount factor for rewards (default: 0.99)')
    parser.add_argument('--use-gae', action='store_true', default=False,
        help='use generalized advantage estimation')
    parser.add_argument('--gae-lambda', type=float, default=0.95,
        help='gae lambda parameter (default: 0.95)')
    parser.add_argument('--entropy-coef', type=float, default=0.01,
        help='entropy term coefficient (default: 0.01)')
    parser.add_argument('--value-loss-coef', type=float, default=0.5,
        help='value loss coefficient (default: 0.5)')
    parser.add_argument('--max-grad-norm', type=float, default=0.5,
        help='max norm of gradients (default: 0.5)')
    parser.add_argument('--recurrent-policy', action='store_true', default=False,
        help='use a recurrent policy')
    parser.add_argument('--use-proper-time-limits', action='store_true', default=False,
        help='compute returns taking into account time limits')
    parser.add_argument('--use-linear-lr-decay', action='store_true', default=False,
        help='use a linear schedule on the learning rate')
    
    parser.add_argument('--cuda-deterministic', action='store_true', default=False,
        help="sets flags for determinism when using CUDA (potentially slow!)")
    parser.add_argument('--no-cuda', action='store_true', default=False,
        help='disables CUDA training')
    
    # PPO Parameters
    parser.add_argument('--ppo-epoch', type=int, default=4,
        help='number of ppo epochs (default: 4)')
    parser.add_argument('--num-mini-batch', type=int, default=32,
        help='number of minibatches to split PPO batch into. Should be less than num-processses, ideally a factor of it')
    parser.add_argument('--clip-param', type=float, default=0.2,
        help='ppo clip parameter (default: 0.2)')
    
    # File parameters
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--num-processes', type=int, default=16,
        help='how many training CPU processes to use (default: 16)')
    parser.add_argument('--log-interval', type=int, default=10,
        help='log interval, one log per n updates (default: 10)')
    parser.add_argument('--eval-interval', type=int, default=None,
        help='eval interval, one eval per n updates (default: None)')
    parser.add_argument('--log-dir', default='/tmp/gym/',
        help='directory to save agent logs (default: /tmp/gym)')
    
    # Env arguments
    parser.add_argument('--num-env-steps', type=int, default=10e6,
        help='number of environment steps to train (default: 10e6)')
    parser.add_argument('--env-name', default='PongNoFrameskip-v4',
        help='environment to train on (default: PongNoFrameskip-v4)')

    # Saving parameters
    parser.add_argument('--track', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
        help='if toggled, this experiment will be tracked with Weights and Biases')
    parser.add_argument('--wandb-project-name', type=str, default="cleanRL",
        help="the wandb's project name")
    parser.add_argument('--wandb-entity', type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument('--capture-video', type=int, default=0, 
        help='whether to capture videos of the agent performances (check out `videos` folder)' + 
        ' pass an int for how often (every n episodes) the videos should be recorded')
    parser.add_argument('--upload-video', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
        help='if toggled, upload videos to wandb (not working on chpc)')
    
    parser.add_argument('--exp-name', type=str, default=None,
        help='the name of this experiment')
    
    parser.add_argument('--save-interval', type=int, default=100,
        help='save interval, one save per n updates (default: 100)')
    parser.add_argument('--save-name', type=str, default=None,
        help='the name that will be saved for this experiment (default to env name)')
    parser.add_argument('--save-dir', type=str, default='',
        help='the subdirectory trained models saved in')
    
    parser.add_argument('--checkpoint-interval', type=int, default=100,
        help='number of updates before a checkpoint of the model should be saved, ' + \
        'if 0, then no checkpoints will be saved')
    parser.add_argument("--checkpoint-dir", type=str, default='',
        help="subdirectory of saved_checkpoints/ to save checkpoints into")

    parser.add_argument('--config-file-name', type=str, default=None,
        help='this is used specifically for automatic scheduler to determine ' + \
        'what experiment to write as being done once successfully completed')
    parser.add_argument('--env-kwargs', nargs='*', action=ParseKwargs, default=None)
    parser.add_argument('--normalize-env', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=False,
                        help='if toggled, turn off normalization in environment wrapper')
    parser.add_argument('--cont', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
        help='if toggled, attempt to load a model as named from save_path under the right folder to continue experiment')
    parser.add_argument('--cont-file-name', type=str, default=None,
        help='if given, try to load a specific file')
    
    #Andy: add options for using a custom NN base for policy
    parser.add_argument('--nn-base', type=str, default=None,
        help='pass a string to use a specific NNBase from model.py, e.g. FlexBase')
    parser.add_argument('--nn-base-kwargs', nargs='*', action=ParseKwargs, default={},
        help='pass kwargs for the NNBase, such as how many shared layers')

    #Andy: add options for AuxVecPyTorch wrapper that can add auxiliary task
    parser.add_argument('--aux-wrapper-kwargs', nargs='*', action=ParseKwargs, default={},
        help='pass kwargs for AuxVecPyTorch, mainly options for auxiliary tasks')
    
    #Andy: add auxiliary loss weight. In the future, we may want to adjust this
    #   so that we can use a different weighting per task
    parser.add_argument('--auxiliary-loss-coef', type=float, default=0.3,
        help='auxiliary loss coefficient (default: 0.3)')
    #Andy: add flag to use new auxiliary training methods that split each aux task into individually computed
    #losses so that we can specify different truth sizes and loss functions
    parser.add_argument('--use-new-aux', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=False,
        help='if toggled, use new auxiliary training algorithms PPOAux, RolloutStorageAux')
    parser.add_argument('--auxiliary-truth-sizes', nargs='*', action=ParseList, default=[])
    
    #Andy: add new experiment type - load and freeze parameters set from  an existing model
    parser.add_argument('--clone-parameter-experiment', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
        help='if toggled, attempt to load a model as named from clone-path parameter to use for a new model')
    parser.add_argument('--clone-args', nargs='*', action=ParseKwargs, default={},
        help='if clone-parameter-eperiment is true, pass params. Params expected: \n"clone_path": direct path to target clone newtork \n' + \
             '"clone_layers": list of layers that we want to clone or int of first n layers\n' + \
             '"freeze": boolean of whether to freeze the cloned layers. True/False or list for each layer in list')

    ''' Example
      clone_args = {
        'clone_layers': ['gru', 'shared0'], #named layers to copy over,
        'freeze_layers': ['gru', 'shared0'], #named layers to freeze,
        'copy_obs_rms': True #whether to copy obs_rms to envs,
        'freeze_obs_rms': True #whether to freeze obs_rms
        'norm_layers': True #whether to scale down weight sizes to be close to standard initial mean
    }
    '''

    #Andy: One speicifc experiment to see if we remove non-val grads from shared layer,
    # is there actually a reduction in performance
    parser.add_argument('--remove-actor-grads-on-shared', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
        help='if toggled, only use value grads for shared layers')

    parser.add_argument('--use-universal-step', nargs='*', action=ParseKwargs, default={'on': False},
        help='pass a dict with a [schedule] key to show when manual resets to update universal step should occur')

    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    assert args.algo in ['a2c', 'ppo', 'acktr']
    if args.recurrent_policy:
        assert args.algo in ['a2c', 'ppo'], \
            'Recurrent policy is not implemented for ACKTR'

    return args
