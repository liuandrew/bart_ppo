import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym

from ppo.distributions import Bernoulli, Categorical, DiagGaussian
from ppo.utils import init


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Policy(nn.Module):
    def __init__(self, obs_shape, action_space, base=None, base_kwargs=None):
        super(Policy, self).__init__()
        if base_kwargs is None:
            base_kwargs = {}
        if base is None:
            if len(obs_shape) == 3:
                base = CNNBase
            elif len(obs_shape) == 1:
                base = FlexBaseAux
            else:
                raise NotImplementedError
        elif type(base) == str:
            #Attempt to find a base class with the given string
            base = globals()[base]

        self.base = base(obs_shape[0], **base_kwargs)

        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
            self.dist = DiagGaussian(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "MultiBinary":
            num_outputs = action_space.shape[0]
            self.dist = Bernoulli(self.base.output_size, num_outputs)
        else:
            raise NotImplementedError

    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.base.recurrent_hidden_state_size

    #Andy: add auxiliary check
    @property
    def has_auxiliary(self):
        return self.base.has_auxiliary

    @property
    def auxiliary_output_size(self):
        if self.base.has_auxiliary:
            return self.base.auxiliary_output_size
        else:
            return 1
        
    @property
    def auxiliary_output_size(self):
        if self.base.has_auxiliary:
            return self.base.auxiliary_output_size
        else:
            return 1
    
    #Used for the new FlexBaseAux
    @property
    def auxiliary_output_sizes(self):
        if self.base.has_auxiliary:
            return self.base.auxiliary_output_sizes
        else:
            return []
    
    

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def act(self, inputs, rnn_hxs, masks, deterministic=False, with_activations=False):
        outputs = self.base(inputs, rnn_hxs, masks, deterministic, with_activations)
        value = outputs['value']
        actor_features = outputs['actor_features']
        rnn_hxs = outputs['rnn_hxs']
        if not self.base.has_auxiliary:
            outputs['auxiliary_preds'] = None

        dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()
        
        outputs['action'] = action
        outputs['action_log_probs'] = action_log_probs
        outputs['probs'] = dist.probs

        return outputs

    def get_value(self, inputs, rnn_hxs, masks):
        outputs = self.base(inputs, rnn_hxs, masks)
        value = outputs['value']
        return value

    def evaluate_actions(self, inputs, rnn_hxs, masks, action):
        
        # Andy: Should refactor this function to also return outputs dict
        outputs = self.base(inputs, rnn_hxs, masks)
        actor_features = outputs['actor_features']
        value = outputs['value']
        rnn_hxs = outputs['rnn_hxs']
        if self.base.has_auxiliary:
            auxiliary = outputs['auxiliary_preds']
        else:
            auxiliary = torch.zeros(value.shape)

        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hxs, auxiliary
    
    def get_rnn_hxs(self, num_processes=1):
        '''Get blank rnn_hxs to be used for a reset computation'''
        return torch.zeros(num_processes, self.recurrent_hidden_state_size)


class NNBase(nn.Module):
    def __init__(self, recurrent, recurrent_input_size, hidden_size):
        super(NNBase, self).__init__()

        self._hidden_size = hidden_size
        self._recurrent = recurrent
        #Andy: default the NNBase to having no auxiliary outputs
        self.has_auxiliary = False 
        self.auxiliary_output_size = 0

        if recurrent:
            self.gru = nn.GRU(recurrent_input_size, hidden_size)
            for name, param in self.gru.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0)
                elif 'weight' in name:
                    nn.init.orthogonal_(param)

    @property
    def is_recurrent(self):
        return self._recurrent

    @property
    def recurrent_hidden_state_size(self):
        if self._recurrent:
            return self._hidden_size
        return 1

    @property
    def output_size(self):
        return self._hidden_size

    def _forward_gru(self, x, hxs, masks):
        if x.size(0) == hxs.size(0):
            x, hxs = self.gru(x.unsqueeze(0), (hxs * masks).unsqueeze(0))
            x = x.squeeze(0)
            hxs = hxs.squeeze(0)
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)

            # unflatten
            x = x.view(T, N, x.size(1))

            # Same deal with masks
            masks = masks.view(T, N)

            # Let's figure out which steps in the sequence have a zero for any agent
            # We will always assume t=0 has a zero in it as that makes the logic cleaner
            has_zeros = ((masks[1:] == 0.0) \
                            .any(dim=-1)
                            .nonzero()
                            .squeeze()
                            .cpu())

            # +1 to correct the masks[1:]
            if has_zeros.dim() == 0:
                # Deal with scalar
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [T]

            hxs = hxs.unsqueeze(0)
            outputs = []
            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in masks together!
                # This is much faster
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]

                rnn_scores, hxs = self.gru(
                    x[start_idx:end_idx],
                    hxs * masks[start_idx].view(1, -1, 1))

                outputs.append(rnn_scores)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = torch.cat(outputs, dim=0)
            # flatten
            x = x.view(T * N, -1)
            hxs = hxs.squeeze(0)

        return x, hxs


class CNNBase(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=512):
        super(CNNBase, self).__init__(recurrent, hidden_size, hidden_size)

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('relu'))

        self.main = nn.Sequential(
            init_(nn.Conv2d(num_inputs, 32, 8, stride=4)), nn.ReLU(),
            init_(nn.Conv2d(32, 64, 4, stride=2)), nn.ReLU(),
            init_(nn.Conv2d(64, 32, 3, stride=1)), nn.ReLU(), Flatten(),
            init_(nn.Linear(32 * 7 * 7, hidden_size)), nn.ReLU())

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = self.main(inputs / 255.0)

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        return self.critic_linear(x), x, rnn_hxs


class MLPBase(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=64):
        super(MLPBase, self).__init__(recurrent, num_inputs, hidden_size)
        
        if recurrent:
            num_inputs = hidden_size

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        self.actor = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = inputs

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        hidden_critic = self.critic(x)
        hidden_actor = self.actor(x)

        return self.critic_linear(hidden_critic), hidden_actor, rnn_hxs






#Andy: Add FlexBase currently for allowing shared layers between
# actor and critic
class FlexBaseOld(NNBase):
    '''
    NN module that allows for shared actor and critic layers as well
    as varying number of output heads
    
    num_layers: how many hidden MLP layers from input (or from GRU) to output heads
    num_shared_layers: how many of these MLP layers should be shared between actor and critic 
        -1 means all layers should be shared
    '''
    def __init__(self, num_inputs, recurrent=True, hidden_size=64,
                num_layers=2, num_shared_layers=-1):
        super(FlexBaseOld, self).__init__(recurrent, num_inputs, hidden_size)
        
        print('Using FlexBase')
        print('num shared layers is ' + str(num_shared_layers))

        if recurrent:
            num_inputs = hidden_size

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))
        
        shared_layers = []
        critic_layers = []
        actor_layers = []
        
        # generate all the shared layers
        cur_shared_layers = 0
        in_dim = num_inputs
        for i in range(num_layers):
            if num_shared_layers == -1 or cur_shared_layers < num_shared_layers:
                shared_layers.append(init_(nn.Linear(in_dim, hidden_size)))
                shared_layers.append(nn.Tanh())
                in_dim = hidden_size # only first layer with have input size num_inputs
                cur_shared_layers += 1
        
        # generate the non-shared layers
        if num_shared_layers != -1:
            remaining_layers = num_layers - num_shared_layers
        else:
            remaining_layers = 0
        
        for i in range(remaining_layers):
            critic_layers.append(init_(nn.Linear(in_dim, hidden_size)))
            critic_layers.append(nn.Tanh())
            actor_layers.append(init_(nn.Linear(in_dim, hidden_size)))
            actor_layers.append(nn.Tanh())
            in_dim = hidden_size # only first layer with have input size num_inputs
            
        # finally create the critic linear output
        critic_layers.append(init_(nn.Linear(in_dim, 1)))
        
        if len(shared_layers) > 0:
            self.shared_layers = nn.Sequential(*shared_layers)
        else:
            self.shared_layers = None
        
        if len(actor_layers) > 0:
            self.actor_layers = nn.Sequential(*actor_layers)
        else:
            self.actor_layers = None
            
        if len(critic_layers) > 0:
            self.critic_layers = nn.Sequential(*critic_layers)
        else:
            self.critic_layers = None
            
        self.train()
            
    def forward(self, inputs, rnn_hxs, masks):
        x = inputs
        
        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)
        
        if self.shared_layers:
            x = self.shared_layers(x)
            
        if self.actor_layers:
            hidden_actor = self.actor_layers(x)
        else:
            # if all layers are shared between actor and critic,
            # the last output of shared layers will be x
            # which will be used by the dist function in Policy (model.py)
            hidden_actor = x
            
        if self.critic_layers:
            # this should always run since we will output the critic evaluation here
            critic_val = self.critic_layers(x)
        else:
            raise Exception('Something mysterious happened... there was no final critic head')
        
        return critic_val, hidden_actor, rnn_hxs





# To totally customize where auxiliary tasks are attached, lets split up the shared layers
# into individually activatable (self.shared_layers becomes a list of nn.Sequentials) ones
class FlexBase(NNBase):
    '''
    NN module that allows for shared actor and critic layers as well
    as varying number of output heads
    
    num_layers: how many hidden MLP layers from input (or from GRU) to output heads
    num_shared_layers: how many of these MLP layers should be shared between actor and critic 
        -1 means all layers should be shared
        
    Auxiliary Tasks:
    To add auxiliary heads, we will choose which layers each auxiliary head will be attached to
    For each auxiliary task we need:
        * Depth of layer to attach
        * Whether to attach to actor or critic side
        * Type of output (here we can use Gym spaces and distributions as the Policy does, or linear output)
    Thus each entry to auxiliary_heads should be
        [(depth: -1 is last, 1 is after first layer (1. recurrent layer, 2. hidden, etc.),
          side: 0:actor or 1:critic, -1: if we expect to be on shared layers
          output: gym.spaces.Discrete, gym.spaces.Box, or int representing output dimension for linear output)]
        
    conv1d_layers: how many of these conv1d layers we should use

    !IMPORTANT - to use Gym distributions and such, we will need to adjust code further
        specifically looking into allowing to pass the predicted outputs in 
        evaluate_actions in Policy from PPO algorithm, and getting log_probs
        to get loss from. Linear loss is easier to code in so this is what we
        will focus on for now
        In other words, Distributions are not ready to be used as auxiliary
        tasks yet
    '''
    def __init__(self, num_inputs, recurrent=True, hidden_size=64,
                num_layers=2, num_shared_layers=0, auxiliary_heads=[],
                conv1d_layers=0):
        super(FlexBase, self).__init__(recurrent, num_inputs, hidden_size)
        
        self.num_layers = num_layers
        self.auxiliary_heads = auxiliary_heads
        self.has_auxiliary = True

        if recurrent:
            num_inputs = hidden_size
            self.num_layers += 1

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))
        
        
        self.shared_layers = []
        self.critic_layers = []
        self.actor_layers = []
        self.conv1d_layers = []
        
        # generate all the shared layers
        cur_shared_layers = 0
        in_dim = num_inputs
        for i in range(num_layers):
            if num_shared_layers == -1 or cur_shared_layers < num_shared_layers:
                setattr(self, 'shared'+str(i), nn.Sequential(
                    init_(nn.Linear(in_dim, hidden_size)),
                    nn.Tanh()
                ))
                self.shared_layers.append(getattr(self, 'shared'+str(i)))
                in_dim = hidden_size # only first layer with have input size num_inputs
                cur_shared_layers += 1
        
        # generate the non-shared layers
        if num_shared_layers != -1:
            remaining_layers = num_layers - num_shared_layers
        else:
            remaining_layers = 0
        
        for i in range(remaining_layers):            
            setattr(self, 'critic'+str(i), nn.Sequential(
                init_(nn.Linear(in_dim, hidden_size)),
                nn.Tanh()
            ))
            setattr(self, 'actor'+str(i), nn.Sequential(
                init_(nn.Linear(in_dim, hidden_size)),
                nn.Tanh()
            ))
            
            self.critic_layers.append(getattr(self, 'critic'+str(i)))
            self.actor_layers.append(getattr(self, 'actor'+str(i)))

            in_dim = hidden_size # only first layer with have input size num_inputs
            
        # finally create the critic linear output
#         critic_layers.append(init_(nn.Linear(in_dim, 1)))
        self.critic_head = init_(nn.Linear(in_dim, 1))
        self.critic_layers.append(self.critic_head)
        
            
        self.auxiliary_layers = []
        self.auxiliary_output_idxs = [] # indexes for generating auxiliary outputs
        self.auxiliary_layer_types = [] # 0 linear, 1 distribution
        self.auxiliary_output_size = 0
        # generate auxiliary outputs
        current_auxiliary_output_idx = 0
        for i, head in enumerate(auxiliary_heads):
            depth = head[0]
            if depth == -1:
                depth = self.num_layers
            side = head[1]
            output_type = head[2]
            self.auxiliary_output_idxs.append(current_auxiliary_output_idx)
            if depth == 0:
                raise Exception('Auxiliary task requesting depth of 0')
            if depth > self.num_layers:
                raise Exception('Auxiliary task requesting depth greater than exists in network (head[0])')
            if side > 1:
                raise Exception('Auxiliary task requesting side that is not 0 (actor) or 1 (critic)')
            total_shared_layers = num_shared_layers
            if recurrent: 
                total_shared_layers += 1

            if side == -1:
                if depth > total_shared_layers:
                    raise Exception('Auxiliary task expects to be on shared layers, but is assigned to layers past shared')
            else:
                if depth <= total_shared_layers:
                    raise Exception('Auxiliary task expects to be on individual layers, but is assigned to shared depth')
            
            if type(output_type) == int:
                # linear output
                layer = init_(nn.Linear(hidden_size, output_type))
                self.auxiliary_output_size += output_type
                self.auxiliary_layer_types.append(0)
                current_auxiliary_output_idx += output_type
            elif hasattr(output_type, '__class__'):
                # output based on gym space
                # code taken from Policy to implement a dist function
                if output_type.__class__.__name__ == "Discrete":
                    num_outputs = output_type.n
                    layer = Categorical(hidden_size, num_outputs)
                    num_outputs = 1
                elif output_type.__class__.__name__ == "Box":
                    num_outputs = output_type.shape[0]
                    layer = DiagGaussian(hidden_size, num_outputs)
                elif output_type.__class__.__name__ == "MultiBinary":
                    num_outputs = output_type.shape[0]
                    layer = Bernoulli(hidden_size, num_outputs)
                else:
                    raise NotImplementedError
                self.auxiliary_output_size += num_outputs
                self.auxiliary_layer_types.append(1)
                current_auxiliary_output_idx += num_outputs
                
            setattr(self, 'auxiliary'+str(i), layer)
            self.auxiliary_layers.append(getattr(self, 'auxiliary'+str(i)))
        
        if self.auxiliary_output_size == 0:
            self.auxiliary_output_size = 1
            self.has_auxiliary = False
        self.train()
            
    # def forward(self, inputs, rnn_hxs, masks, deterministic=False):
    #     current_layer = 0
    #     shared_layer_idx = 0
    #     individual_layer_idx = 0
    #     on_shared_layers = True
    #     auxiliary_preds = torch.zeros((inputs.shape[0], self.auxiliary_output_size))
    #     x = inputs
        
    #     if self.is_recurrent:
    #         x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)
    #         current_layer += 1
        
    #     actor_x = x
    #     critic_x = x
        
    #     for i in range(current_layer, self.num_layers+1):
    #         # iterate through the layers whether shared or individual actor/critic
    #         # print(i)
    #         # first check if any auxiliary tasks have the current depth
    #         for j, head in enumerate(self.auxiliary_heads):
    #             # depth = head[0]
    #             # side = head[1]
    #             depth = head[0]
    #             if depth == -1:
    #                 depth = self.num_layers
    #             if depth == current_layer:
    #                 # print('Calling auxiliary head at depth {}'.format(i))
    #                 # figure out if we are on shared layer
    #                 if on_shared_layers:
    #                     auxiliary_input = x
    #                 elif head[1] == 0:
    #                     auxiliary_input = actor_x
    #                 elif head[1] == 1:
    #                     auxiliary_input = critic_x
                    
    #                 # convert to output of auxiliary head
    #                 auxiliary_output = self.auxiliary_layers[j](auxiliary_input)
    #                 if self.auxiliary_layer_types[j] == 1:
    #                     if deterministic:
    #                         auxiliary_output = auxiliary_output.mode()
    #                     else:
    #                         auxiliary_output = auxiliary_output.sample()
    #                 size = auxiliary_output.shape[1]
    #                 start_idx = self.auxiliary_output_idxs[j]
    #                 auxiliary_preds[:, start_idx:start_idx+size] = auxiliary_output
            
    #         # continue proceding through layers
    #         # check if we still have shared layers to complete
    #         if len(self.shared_layers) > 0 and shared_layer_idx < len(self.shared_layers):
    #             x = self.shared_layers[shared_layer_idx](x)
    #             # print('Calling shared layer {}'.format(shared_layer_idx))
    #             shared_layer_idx += 1
    #             # if shared layers are done, this will set actor_x and critic_x
    #             actor_x = x
    #             critic_x = x
    #         elif len(self.actor_layers) > 0 and individual_layer_idx < len(self.actor_layers):
    #             # no more shared layers - move to actor critic layers
    #             on_shared_layers = False
    #             # print('Calling actor critic layer {}'.format(individual_layer_idx))
    #             actor_x = self.actor_layers[individual_layer_idx](actor_x)
    #             critic_x = self.critic_layers[individual_layer_idx](critic_x)
    #             individual_layer_idx += 1
                
    #         current_layer += 1
                    
                    
    #     # Finally get critic value estimation
    #     critic_val = self.critic_layers[-1](critic_x)
                    
    #     if self.has_auxiliary:
    #         return critic_val, actor_x, rnn_hxs, auxiliary_preds
    #     else:
    #         return critic_val, actor_x, rnn_hxs
        
        
    def forward(self, inputs, rnn_hxs, masks, deterministic=False, with_activations=False):
        """Same as forward function but this will pass back all intermediate values

            _type_: _description_
        """
        current_layer = 0
        shared_layer_idx = 0
        individual_layer_idx = 0
        on_shared_layers = True
        auxiliary_preds = torch.zeros((inputs.shape[0], self.auxiliary_output_size))
        x = inputs

        shared_activations = []
        actor_activations = []
        critic_activations = []

        
        # 0. Compute activations for recurrent layer if we are recurrent
        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)
            shared_activations.append(x)
            current_layer += 1
        
        actor_x = x
        critic_x = x
        
        for i in range(current_layer, self.num_layers+1):
            # iterate through the layers whether shared or individual actor/critic
            # print(i)
            # 1. Auxiliary output computation
            # Check if any auxiliary tasks have the current depth
            # and depending on if they are on the shared, critic, or actor branch
            # evaluate the auxiliary task
            for j, head in enumerate(self.auxiliary_heads):
                # depth = head[0]
                # side = head[1]
                depth = head[0]
                if depth == -1:
                    depth = self.num_layers
                if depth == current_layer:
                    # print('Calling auxiliary head at depth {}'.format(i))
                    # figure out if we are on shared layer
                    if on_shared_layers:
                        auxiliary_input = x
                    elif head[1] == 0:
                        auxiliary_input = actor_x
                    elif head[1] == 1:
                        auxiliary_input = critic_x
                    
                    # convert to output of auxiliary head
                    auxiliary_output = self.auxiliary_layers[j](auxiliary_input)
                    if self.auxiliary_layer_types[j] == 1:
                        if deterministic:
                            auxiliary_output = auxiliary_output.mode()
                        else:
                            auxiliary_output = auxiliary_output.sample()
                    size = auxiliary_output.shape[1]
                    start_idx = self.auxiliary_output_idxs[j]
                    auxiliary_preds[:, start_idx:start_idx+size] = auxiliary_output
            
            # 2. Forward pass through the next layer

            # If we still have remaining shared layers, forward pass through the shared layers
            if len(self.shared_layers) > 0 and shared_layer_idx < len(self.shared_layers):
                x = self.shared_layers[shared_layer_idx](x)
                # print('Calling shared layer {}'.format(shared_layer_idx))
                shared_layer_idx += 1
                shared_activations.append(x)
                # if shared layers are done, this will set actor_x and critic_x
                actor_x = x
                critic_x = x
            
            # Otherwise, forward pass through actor and critic layers
            elif len(self.actor_layers) > 0 and individual_layer_idx < len(self.actor_layers):
                on_shared_layers = False
                # print('Calling actor critic layer {}'.format(individual_layer_idx))
                actor_x = self.actor_layers[individual_layer_idx](actor_x)
                critic_x = self.critic_layers[individual_layer_idx](critic_x)
                
                actor_activations.append(actor_x)
                critic_activations.append(critic_x)
                individual_layer_idx += 1
                
            current_layer += 1
                    
                    
        # Finally get critic value estimation
        critic_val = self.critic_layers[-1](critic_x)

        outputs = {
            'value': critic_val,
            'actor_features': actor_x,
            'rnn_hxs': rnn_hxs,
        }
        
        if self.has_auxiliary:
            outputs['auxiliary_preds'] = auxiliary_preds
        if with_activations:
            outputs['activations'] = {
                'shared_activations': shared_activations,
                'actor_activations': actor_activations,
                'critic_activations': critic_activations
            }        
        return outputs
    
    
    
    
    
    
    
    
    
    
    
    
    
# To totally customize where auxiliary tasks are attached, lets split up the shared layers
# into individually activatable (self.shared_layers becomes a list of nn.Sequentials) ones
class FlexBaseAux(NNBase):
    '''
    NN module that allows for shared actor and critic layers as well
    as varying number and types of output heads
    
    num_layers: how many hidden MLP layers from input (or from GRU) to output heads
    num_shared_layers: how many of these MLP layers should be shared between actor and critic 
        -1 means all layers should be shared
        
    Auxiliary Tasks:
    To add auxiliary heads, we will choose which layers each auxiliary head will be attached to
    For each auxiliary task we need:
        * Depth of layer to attach
        * Whether to attach to actor or critic side
        * Type of output (0: simple value output, 1: multi class output)
        * Size of output
    Thus each entry to auxiliary_heads should be
        [(depth: -1 is last, 1 is after first layer (1. recurrent layer, 2. hidden, etc.),
          side: 0:actor or 1:critic, -1: if we expect to be on shared layers
          output_type: 0:linear value, 1:multiclass distribution probabilities),
          output_size: int for how many output values, or how many classes]
        
    conv1d_layers: how many of these conv1d layers we should use

    !IMPORTANT - to use Gym distributions and such, we will need to adjust code further
        specifically looking into allowing to pass the predicted outputs in 
        evaluate_actions in Policy from PPO algorithm, and getting log_probs
        to get loss from. Linear loss is easier to code in so this is what we
        will focus on for now
        In other words, Distributions are not ready to be used as auxiliary
        tasks yet
    '''
    def __init__(self, num_inputs, recurrent=True, hidden_size=64,
                num_layers=2, num_shared_layers=0, auxiliary_heads=[],
                conv1d_layers=0):
        super(FlexBaseAux, self).__init__(recurrent, num_inputs, hidden_size)
        
        self.num_layers = num_layers
        self.auxiliary_heads = auxiliary_heads
        self.has_auxiliary = True

        if recurrent:
            num_inputs = hidden_size
            self.num_layers += 1

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))
        
        
        self.shared_layers = []
        self.critic_layers = []
        self.actor_layers = []
        self.conv1d_layers = []
        
        # generate all the shared layers
        cur_shared_layers = 0
        in_dim = num_inputs
        for i in range(num_layers):
            if num_shared_layers == -1 or cur_shared_layers < num_shared_layers:
                setattr(self, 'shared'+str(i), nn.Sequential(
                    init_(nn.Linear(in_dim, hidden_size)),
                    nn.Tanh()
                ))
                self.shared_layers.append(getattr(self, 'shared'+str(i)))
                in_dim = hidden_size # only first layer with have input size num_inputs
                cur_shared_layers += 1
        
        # generate the non-shared layers
        if num_shared_layers != -1:
            remaining_layers = num_layers - num_shared_layers
        else:
            remaining_layers = 0
        
        for i in range(remaining_layers):            
            setattr(self, 'critic'+str(i), nn.Sequential(
                init_(nn.Linear(in_dim, hidden_size)),
                nn.Tanh()
            ))
            setattr(self, 'actor'+str(i), nn.Sequential(
                init_(nn.Linear(in_dim, hidden_size)),
                nn.Tanh()
            ))
            
            self.critic_layers.append(getattr(self, 'critic'+str(i)))
            self.actor_layers.append(getattr(self, 'actor'+str(i)))

            in_dim = hidden_size # only first layer with have input size num_inputs
            
        # finally create the critic linear output
#         critic_layers.append(init_(nn.Linear(in_dim, 1)))
        self.critic_head = init_(nn.Linear(in_dim, 1))
        self.critic_layers.append(self.critic_head)
        
            
        self.auxiliary_layers = []
        self.auxiliary_output_idxs = [] # indexes for generating auxiliary outputs
        self.auxiliary_layer_types = [] # 0 linear, 1 distribution
        self.auxiliary_output_sizes = []
        # generate auxiliary outputs
        current_auxiliary_output_idx = 0
        for i, head in enumerate(auxiliary_heads):
            depth = head[0]
            if depth == -1:
                depth = self.num_layers
            side = head[1]
            output_type = head[2]
            output_size = head[3]
            self.auxiliary_output_idxs.append(current_auxiliary_output_idx)
            if depth == 0:
                raise Exception('Auxiliary task requesting depth of 0')
            if depth > self.num_layers:
                raise Exception('Auxiliary task requesting depth greater than exists in network (head[0])')
            if side > 1:
                raise Exception('Auxiliary task requesting side that is not 0 (actor) or 1 (critic)')
            total_shared_layers = num_shared_layers
            if recurrent: 
                total_shared_layers += 1

            if side == -1:
                if depth > total_shared_layers:
                    raise Exception('Auxiliary task expects to be on shared layers, but is assigned to layers past shared')
            else:
                if depth <= total_shared_layers:
                    raise Exception('Auxiliary task expects to be on individual layers, but is assigned to shared depth')
            
            if output_type == 0:
                # linear output
                layer = init_(nn.Linear(hidden_size, output_size))
                self.auxiliary_output_sizes.append(output_size)
                self.auxiliary_layer_types.append(0)
            elif output_type == 1:
                # output based on gym space
                # code taken from Policy to implement a dist function
                layer = Categorical(hidden_size, output_size)
                self.auxiliary_output_sizes.append(output_size)
                self.auxiliary_layer_types.append(1)
            else:
                raise NotImplementedError
                
            setattr(self, 'auxiliary'+str(i), layer)
            self.auxiliary_layers.append(getattr(self, 'auxiliary'+str(i)))
        
        if len(self.auxiliary_output_sizes) == 0:
            self.has_auxiliary = False
        self.train()
        
        
    def forward(self, inputs, rnn_hxs, masks, deterministic=False, with_activations=False):
        """Same as forward function but this will pass back all intermediate values

            _type_: _description_
        """
        current_layer = 0
        shared_layer_idx = 0
        individual_layer_idx = 0
        on_shared_layers = True
        # auxiliary_preds = torch.zeros((inputs.shape[0], self.auxiliary_output_size))
        auxiliary_preds = [None for i in range(len(self.auxiliary_output_sizes))]
        x = inputs

        shared_activations = []
        actor_activations = []
        critic_activations = []

        
        # 0. Compute activations for recurrent layer if we are recurrent
        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)
            shared_activations.append(x)
            current_layer += 1
        
        actor_x = x
        critic_x = x
        
        for i in range(current_layer, self.num_layers+1):
            # iterate through the layers whether shared or individual actor/critic
            # print(i)
            # 1. Auxiliary output computation
            # Check if any auxiliary tasks have the current depth
            # and depending on if they are on the shared, critic, or actor branch
            # evaluate the auxiliary task
            for j, head in enumerate(self.auxiliary_heads):
                # depth = head[0]
                # side = head[1]
                depth = head[0]
                if depth == -1:
                    depth = self.num_layers
                if depth == current_layer:
                    # print('Calling auxiliary head at depth {}'.format(i))
                    # figure out if we are on shared layer
                    if on_shared_layers:
                        auxiliary_input = x
                    elif head[1] == 0:
                        auxiliary_input = actor_x
                    elif head[1] == 1:
                        auxiliary_input = critic_x
                    
                    # convert to output of auxiliary head
                    auxiliary_output = self.auxiliary_layers[j](auxiliary_input)
                    if self.auxiliary_layer_types[j] == 1:
                        auxiliary_output = auxiliary_output.probs
                    auxiliary_preds[j] = auxiliary_output
            
            # 2. Forward pass through the next layer

            # If we still have remaining shared layers, forward pass through the shared layers
            if len(self.shared_layers) > 0 and shared_layer_idx < len(self.shared_layers):
                x = self.shared_layers[shared_layer_idx](x)
                # print('Calling shared layer {}'.format(shared_layer_idx))
                shared_layer_idx += 1
                shared_activations.append(x)
                # if shared layers are done, this will set actor_x and critic_x
                actor_x = x
                critic_x = x
            
            # Otherwise, forward pass through actor and critic layers
            elif len(self.actor_layers) > 0 and individual_layer_idx < len(self.actor_layers):
                on_shared_layers = False
                # print('Calling actor critic layer {}'.format(individual_layer_idx))
                actor_x = self.actor_layers[individual_layer_idx](actor_x)
                critic_x = self.critic_layers[individual_layer_idx](critic_x)
                
                actor_activations.append(actor_x)
                critic_activations.append(critic_x)
                individual_layer_idx += 1
                
            current_layer += 1
                    
                    
        # Finally get critic value estimation
        critic_val = self.critic_layers[-1](critic_x)

        outputs = {
            'value': critic_val,
            'actor_features': actor_x,
            'rnn_hxs': rnn_hxs,
        }
        
        if self.has_auxiliary:
            outputs['auxiliary_preds'] = auxiliary_preds
        if with_activations:
            outputs['activations'] = {
                'shared_activations': shared_activations,
                'actor_activations': actor_activations,
                'critic_activations': critic_activations
            }        
        return outputs
    
    
class DelayedRNNPPO(NNBase):
    '''
    Quick and simple static RNN network with a FC followed by RNN followed by
    2 layers of actor critic split
    '''
    def __init__(self, num_inputs, hidden_size=64,
                auxiliary_heads=[], recurrent=True):
        super(DelayedRNNPPO, self).__init__(True, hidden_size, hidden_size)
        # parameters create self.GRU with hidden_size as recurrent_input_size and
        #  hidden_size as recurrent_hidden_size
        
        self.auxiliary_heads = auxiliary_heads
        self.has_auxiliary = True

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))
                
        self.shared_layers = []
        self.critic_layers = []
        self.actor_layers = []
        self.conv1d_layers = []
        
        # generate all the shared layers        
        self.shared0 = nn.Sequential(init_(nn.Linear(num_inputs, hidden_size)),
                                nn.Tanh())
        self.critic0 = nn.Sequential(init_(nn.Linear(hidden_size, hidden_size)),
                                nn.Tanh())
        self.critic1 = nn.Sequential(init_(nn.Linear(hidden_size, hidden_size)),
                                nn.Tanh())
        self.actor0 = nn.Sequential(init_(nn.Linear(hidden_size, hidden_size)),
                                nn.Tanh())
        self.actor1 = nn.Sequential(init_(nn.Linear(hidden_size, hidden_size)),
                                nn.Tanh())        
        self.critic_head = init_(nn.Linear(hidden_size, 1))
        
            
        self.auxiliary_layers = []
        self.auxiliary_output_idxs = [] # indexes for generating auxiliary outputs
        self.auxiliary_layer_types = [] # 0 linear, 1 distribution
        self.auxiliary_output_sizes = []
        # generate auxiliary outputs - assume that all heads are attached at the final actor layer
        current_auxiliary_output_idx = 0
        for i, head in enumerate(auxiliary_heads):
            # depth = head[0]
            # if depth == -1:
            #     depth = self.num_layers
            # side = head[1]
            output_type = head[2]
            output_size = head[3]
            self.auxiliary_output_idxs.append(current_auxiliary_output_idx)

            if output_type == 0:
                # linear output
                layer = init_(nn.Linear(hidden_size, output_size))
                self.auxiliary_output_sizes.append(output_size)
                self.auxiliary_layer_types.append(0)
            elif output_type == 1:
                # output based on gym space
                # code taken from Policy to implement a dist function
                layer = Categorical(hidden_size, output_size)
                self.auxiliary_output_sizes.append(output_size)
                self.auxiliary_layer_types.append(1)
            else:
                raise NotImplementedError
                
            setattr(self, 'auxiliary'+str(i), layer)
            self.auxiliary_layers.append(getattr(self, 'auxiliary'+str(i)))
        
        if len(self.auxiliary_output_sizes) == 0:
            self.has_auxiliary = False
        else:
            self.has_auxiliary = True
            
        self.train()
        
        
    def forward(self, inputs, rnn_hxs, masks, deterministic=False, with_activations=False):
        """Same as forward function but this will pass back all intermediate values

        Generally this is called with a batch of 1 step inputs from N processes with size
        inputs: [N, obs_dim]
        rnn_hxs: [N, hidden_dim]
        masks: [N, 1]
            (masks are 1 to indicate episode is continuing or 0 to indicate
             an episode completion and reset of rnn state)
        
        If trying to evaluate multiple steps (which most commonly happens during update step)
        inputs: [T*N, obs_dim]
        rnn_hxs: [N, hidden_dim]
        masks: [T*N, 1]
        """
        auxiliary_preds = [None for i in range(len(self.auxiliary_output_sizes))]
        x = inputs

        shared_activations = []
        actor_activations = []
        critic_activations = []

        x = self.shared0(x)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        shared_activations.append(x)
        x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)
        shared_activations.append(x)
        
        actor_x = self.actor0(x)
        actor_activations.append(actor_x)
        actor_x = self.actor1(actor_x)
        actor_activations.append(actor_x)
        
        critic_x = self.critic0(x)
        critic_activations.append(critic_x)
        critic_x = self.critic1(x)
        critic_activations.append(critic_x)
                    
        # Finally get critic value estimation
        critic_val = self.critic_head(critic_x)

        # Get auxliary outputs
        if self.has_auxiliary:
            for j, layer in enumerate(self.auxiliary_layers):
                auxiliary_output = layer(actor_x)
                if self.auxiliary_layer_types[j] == 1:
                    auxiliary_output = auxiliary_output.probs
                auxiliary_preds[j] = auxiliary_output

        outputs = {
            'value': critic_val,
            'actor_features': actor_x,
            'rnn_hxs': rnn_hxs,
        }
        
        if self.has_auxiliary:
            outputs['auxiliary_preds'] = auxiliary_preds
        if with_activations:
            outputs['activations'] = {
                'shared_activations': shared_activations,
                'actor_activations': actor_activations,
                'critic_activations': critic_activations
            }        
        return outputs
    
    
def norm_scale_parameters(model, names=[]):
    '''Rescale layers of a model to be as if they were normally initiated'''
    
    for name in names:
        reduce_params = []
        for param in list(model.named_parameters()):
            if name in param[0]:
                reduce_params.append(param)    
        for param in reduce_params:
            if 'weight' in param[0]:
                mean = param[1].abs().mean().item()
                p_clone = param[1].clone()

                if name == 'gru':
                    nn.init.orthogonal_(p_clone)
                else:
                    nn.init.orthogonal_(p_clone, np.sqrt(2))
                target_mean = p_clone.abs().mean().item()
                scale = target_mean / mean
                print(scale)
                param[1].data.copy_(param[1] * scale)
            elif 'bias' in param[0]:
                nn.init.constant_(param[1], 0)
        