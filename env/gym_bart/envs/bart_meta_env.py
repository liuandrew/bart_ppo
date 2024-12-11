import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch

import random

class BartMetaEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "video.frames_per_second": 24}
    def __init__(self, colors_used=3, toggle_task=True,
                 give_last_action=True, give_size=True, give_rew=False,
                 inflate_speed=0.05, inflate_noise=0.02, rew_on_pop=None,
                 pop_noise=0.05, max_steps=2000, meta_setup=0, rew_structure=0,
                 fix_sizes=None, fix_sizes_per_balloon=False, num_balloons=None, rew_p=1,
                 fix_prev_action_bug=False):
        """
        Action space: 3 actions
            toggle_task: if True, action 1 inflates, action 0 lets go
                             if False, action 1 to start/stop, action 0 to wait
                                Currently using same action for start/stop, could consider
                                adding action 2 to stop
        Observation space: 
            give_last_action: if True, give last action in observation
            give_size: if True, give current size in observation
            
        Tweak parameters:
            inflate_speed: how quickly balloon inflates per time step
            inflate_noise: std of Gaussian noise added to inflation
            pop_noise: std of Gaussian noise added to pop time
            rew_on_pop: reward given if balloon pops (set to negative for punishment)
            
            meta_setup:
                0: set balloon mean sizes to be ordered red/orange
                1: set balloon mean sizes to be anything between [0.1, 1]

            rew_structure:
                0: points given for balloon size
                1: points given while inflating, negative on pop
                2: points given for balloon size, x^1.3
                3: points given while inflating, x^1.3, negative on pop
                4: points given for balloon size, x^2
                5: points given for balloon size, x^2, negative on pop
                6: points given for balloon size, x^p, p defined by rew_p
            fix_sizes:
                Can set to dict or list, or even list of lists
            fix_sizes_per_balloon:
                if True, expect that the fix_sizes will be list of lists and
                define the balloon size per balloon, rather than per episode

            fix_prev_action_bug:
                There was a bug that made inner_reset reset the prev_action making
                it useless to decode whether a reward was received
            
            num_balloons:
                If set, fix the number of balloons. Note that max_steps will still be
                    respected
                
        """
        super(BartMetaEnv, self).__init__()

        self.colors = {
            "red": {"mean": 0.2},
            "orange": {"mean": 0.5},
            "yellow": {"mean": 0.8},
            "gray": {"fixed_reward": 0},
            "purple": {"fixed_reward": 1}
        }
        self.color_to_idx = {"red": 0, "orange": 1, "yellow": 2,
                             "gray": 3, "purple": 4}
        self.idx_to_color = {0: "red", 1: "orange", 2: "yellow",
                             3: "gray", 4: "purple"}
        # Env setup parameters
        self.colors_used = colors_used
        self.toggle_task = toggle_task
        self.give_last_action = give_last_action
        self.give_size = give_size
        self.give_rew = give_rew
        self.max_steps = max_steps

        # Tweak parameters
        self.inflate_speed = inflate_speed
        self.inflate_noise = inflate_noise
        if rew_on_pop is None:
            if rew_structure in [0, 2, 4, 6]:
                rew_on_pop = 0
            elif rew_structure in [1, 3, 5]:
                rew_on_pop = -1
        self.rew_on_pop = rew_on_pop
        self.rew_p = rew_p
        self.pop_noise = pop_noise
        self.rew_structure = rew_structure
        self.fix_sizes = fix_sizes
        self.fix_sizes_per_balloon = fix_sizes_per_balloon
        self.fix_prev_action_bug = fix_prev_action_bug
        self.num_balloons = num_balloons
        if self.num_balloons is None:
            self.num_balloons = 1e8
        
        self.inflate_delay = 0
        self.current_step = 0
        self.balloon_count = 0
        self.current_ep = 0
        self.prev_reward = 0.
        self.prev_action = 0
        # self.observation_space = spaces.Tuple((
        #     spaces.Discrete(len(self.colors)),  # Color index
        #     spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),  # Current size
        #     spaces.Discrete(2)  # Previous action
        #     spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),  # Last reward
        # ))
        if self.give_rew:
            self.observation_space = spaces.Box(low=0, high=1, shape=(5+1+2+1,))
        else:
            self.observation_space = spaces.Box(low=0, high=1, shape=(5+1+2,))
        self.action_space = spaces.Discrete(2)  # 0: do nothing, 1: hold inflate button
        # meta parameters
        self.meta_setup = meta_setup
        self.balloon_mean_sizes = {
            0: None,
            1: None,
            2: None,
            3: None,
            4: None,
        }

    def reset(self, seed=None, options={}):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        self.current_step = 0
        self.balloon_count = 0
        # generate meta params
        if self.meta_setup == 0:
            self.balloon_mean_sizes = {
                0: np.random.uniform(0.1, 0.4),
                1: np.random.uniform(0.4, 0.7),
                2: np.random.uniform(0.7, 1)
            }
        elif self.meta_setup == 1:
            self.balloon_mean_sizes = {
                c: np.random.uniform(0.1, 1)
                for c in range(5)
            }
        
        if self.fix_sizes is not None:
            if type(self.fix_sizes) == list:
                for i, size in enumerate(self.fix_sizes):
                    if type(size) == list:
                        self.balloon_mean_sizes[i] = size[self.current_ep]
                    else:
                        self.balloon_mean_sizes[i] = size
            elif type(self.fix_sizes) == dict:
                for i, size in self.fix_sizes.items():
                    if type(size) == list:
                        self.balloon_mean_sizes[i] = size[self.current_ep]
                    else:
                        self.balloon_mean_sizes[i] = size

        self.current_ep += 1
        obs = self.inner_reset()
        return obs, {}

    def inner_reset(self):
        '''
        generate new balloon under current meta conditions
        note that reset() changes to new meta conditions
        '''
        if self.colors_used <= 1:
            self.current_color_idx = 1
            self.current_color = "orange"
        else:
            self.current_color_idx = random.choice(range(self.colors_used))
            self.current_color = self.idx_to_color[self.current_color_idx]
        
        if self.fix_sizes_per_balloon:
            # update the balloon size each balloon
            for i, size in enumerate(self.fix_sizes):
                self.balloon_mean_sizes[i] = size[self.balloon_count]

        self.current_size = 0.0
        if not self.fix_prev_action_bug:
            self.prev_action = 0
        self.inflate_delay = 0
        self.balloon_count += 1
        
        self.currently_inflating = False # used for stop/start version
        
        # Pick a pop size
        mean = self.balloon_mean_sizes[self.current_color_idx]
        self.current_balloon_limit = random.gauss(mean, self.pop_noise)
        
        return self.get_observation()

    def step(self, action):
        terminated = False
        truncated = False
        popped = False
        finished = False
        end_size = 0
        reward = 0
        inflate = 0

        if not self.toggle_task:
            if action == 1:  # Hold inflate button
                inflate = random.gauss(self.inflate_speed, self.inflate_noise)
                self.current_size += inflate

                if self.current_color in ["red", "yellow", "orange"] and \
                    self.current_size > self.current_balloon_limit:
                    end_size = self.current_size
                    self.current_size = 0  # Balloon pops
                    finished = True
                    popped = True
            else:  # Action 0: stop inflating
                if self.current_color in ["red", "yellow", "orange"]:
                    finished = True
                    reward = self.current_size
        else:
            if self.currently_inflating:
                if action == 1:
                    if self.current_color in ["red", "yellow", "orange"]:
                        reward = self.current_size
                        self.currently_inflating = False
                        finished = True
                if action == 0:
                    inflate = random.gauss(self.inflate_speed, self.inflate_noise)
                    self.current_size += inflate
                    if self.current_color in ["red", "yellow", "orange"] and \
                        self.current_size > self.current_balloon_limit:
                        self.current_size = 0  # Balloon pops
                        end_size = self.current_size
                        self.currently_inflating = False
                        finished = True
                        popped = True
            else:
                if action == 1:
                    self.currently_inflating = True
                    inflate = random.gauss(self.inflate_speed, self.inflate_noise)
                    self.current_size += inflate
                else:
                    self.inflate_delay += 1
        
        # Compute rewards
        if popped:
            reward = self.rew_on_pop
        elif self.rew_structure in [0, 2, 4, 5, 6]:
            if finished:
                if self.rew_structure == 0:
                    reward = self.current_size
                elif self.rew_structure == 2:
                    reward = (self.current_size)**1.3
                elif self.rew_structure in [4, 5]:
                    reward = (self.current_size)**2
                elif self.rew_structure == 6:
                    reward = (self.current_size)**(self.rew_p)
        elif self.rew_structure in [1, 3]:
            # Calculate the amount of points gained based on the balloon size increase
            prev_size = self.current_size - inflate
            if self.rew_structure == 1:
                next_rew = self.current_size
                prev_rew = prev_size
            elif self.rew_structure == 3:
                next_rew = (self.current_size)**1.3
                prev_rew = (prev_size)**1.3
            reward = next_rew - prev_rew 
               
        # print(popped, reward, type(reward), self.current_size)
        
        # Note on max step termination, we will still give current size worth of points
        #   since this might better allow for reaction time in meta environment
        self.current_step += 1
        if self.current_step >= self.max_steps: 
            if not popped:
                reward = self.current_size
            terminated = True
        elif self.balloon_count > self.num_balloons:
            terminated = True

        self.prev_action = action
        self.prev_reward = reward

        if popped:
            last_size = end_size
        else:
            last_size = self.current_size
            
        info = {
            'current_color': self.color_to_idx[self.current_color],
            'last_size': last_size,
            'popped': popped,
            'bart_finished': finished,
            'inflate_delay': self.inflate_delay,
            'balloon_limit': self.current_balloon_limit
        }

        if finished:
            next_obs = self.inner_reset()
        else:
            next_obs = self.get_observation()

        return next_obs, reward, terminated, truncated, info
    
    def get_observation(self):
        
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        # obs = torch.zeros(8, dtype=torch.float)
        obs[self.color_to_idx[self.current_color]] = 1
        if self.give_size:
            obs[len(self.colors)] = self.current_size
        if self.give_last_action:
            obs[len(self.colors)+1+self.prev_action] = 1
        if self.give_rew:
            obs[len(self.colors)+3] = self.prev_reward
            
        return obs
    
    def render(self, mode="rgb_array"):
        return np.zeros((64, 64))
    
    