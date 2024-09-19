import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch

import random

class BartEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "video.frames_per_second": 24}
    def __init__(self, colors_used=3, toggle_task=True,
                 give_last_action=True, give_size=True,
                 passive_trial_prob=0.2,
                 fixed_reward_prob=0.2, random_start_wait=False,
                 inflate_speed=0.05, inflate_noise=0, rew_on_pop=0,
                 pop_noise=0.05, max_steps=200, fix_conditions=[],
                 punish_passive=0):
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
            passive_trial_prob: how often to have passive trials
            random_start_wait: whether to add an initial waiting period of 0-5 timesteps
            punish_passive: punishment for hitting the button on passive trials
            
        fixed_conditions: pass conditions to force resets with in the form of a list
            each fixed condition should itself be a dict with optional entries of
            'color': string or int
            'size': float (size limit of balloon)
            'passive': bool (only for red/orange/yellow balloon)
            'delay': int (number  of steps to delay before showing balloon)
            
        """
        super(BartEnv, self).__init__()

        self.colors = {
            "red": {"mean": 0.2},
            "orange": {"mean": 0.5},
            "yellow": {"mean": 0.8},
            "gray": {"fixed_reward": 0, "fixed_size": 0.4},
            "pink": {"fixed_reward": 0.7, "fixed_size": 0.7}
        }
        self.color_to_idx = {"red": 0, "orange": 1, "yellow": 2,
                             "gray": 3, "pink": 4}
        self.idx_to_color = {0: "red", 1: "orange", 2: "yellow",
                             3: "gray", 4: "pink"}
        # Env setup parameters
        self.colors_used = colors_used
        self.toggle_task = toggle_task
        self.give_last_action = give_last_action
        self.give_size = give_size
        self.max_steps = max_steps
        
        self.passive_trial_prob = passive_trial_prob
        self.fixed_reward_prob = fixed_reward_prob
        self.random_start_wait = random_start_wait
        self.punish_passive = punish_passive
        self.fix_conditions = fix_conditions

        # Tweak parameters
        self.inflate_speed = inflate_speed
        self.inflate_noise = inflate_noise
        self.rew_on_pop = rew_on_pop
        self.pop_noise = pop_noise

        self.inflate_delay = 0
        self.current_step = 0
        self.start_wait_length = 0
        self.balloon_count = 0
        # self.observation_space = spaces.Tuple((
        #     spaces.Discrete(len(self.colors)),  # Color index
        #     spaces.Discrete(1), # Passive trial flag
        #     spaces.Discrete(2)  # Previous action
        #     spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),  # Current size
        # ))
        self.observation_space = spaces.Box(low=0, high=1, shape=(5+1+2+1,))
        self.action_space = spaces.Discrete(2)  # 0: do nothing, 1: hold inflate button


    def reset(self, seed=None, options={}):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
            
        self.passive_trial = False
        
        if random.random() < self.fixed_reward_prob:
            self.current_color = self.idx_to_color[random.choice([3, 4])] #gray or pink
        else:
            if self.colors_used <= 1:
                self.current_color = "orange"
            else:
                self.current_color = self.idx_to_color[random.choice(range(self.colors_used))]
                
            if random.random() < self.passive_trial_prob:
                self.passive_trial = True
        if self.balloon_count < len(self.fix_conditions) and \
            'color' in self.fix_conditions[self.balloon_count]:
            c = self.fix_conditions[self.balloon_count]['color']
            if type(c) == int:
                c = self.idx_to_color[c]
            self.current_color = c
        if self.balloon_count < len(self.fix_conditions) and \
            'passive' in self.fix_conditions[self.balloon_count]:
            self.passive_trial = self.fix_conditions[self.balloon_count]['passive']
        self.current_size = 0.0
        self.prev_action = 0
        self.inflate_delay = 0
        self.current_step = 0
        self.start_wait_length = 0

        self.currently_inflating = False # used for stop/start version
        
        if self.random_start_wait:
            self.start_wait_length = random.choice(range(6))
        if self.balloon_count < len(self.fix_conditions) and \
            'delay' in self.fix_conditions[self.balloon_count]:
            self.start_wait_length = self.fix_conditions[self.balloon_count]['delay']
        
        # Pick a pop size
        if self.current_color in ["red", "orange", "yellow"]:
            mean = self.colors[self.current_color]["mean"]
            if self.passive_trial:
                self.current_balloon_limit = mean
            else:
                self.current_balloon_limit = random.gauss(mean, self.pop_noise)
        else:
            self.current_balloon_limit = self.colors[self.current_color]["fixed_size"]
        if self.balloon_count < len(self.fix_conditions) and \
            'size' in self.fix_conditions[self.balloon_count]:
            self.current_balloon_limit = self.fix_conditions[self.balloon_count]['size']

            
            
        self.balloon_count += 1
            

        return self.get_observation(), {}

    def step(self, action):
        terminated = False
        truncated = False
        popped = False
        end_size = 0
        reward = 0

        if self.current_step < self.start_wait_length:
            # Ignore all actions
            pass
        elif not self.toggle_task:
            if action == 1:  # Hold inflate button
                inflate = random.gauss(self.inflate_speed, self.inflate_noise)
                self.current_size += inflate

                if self.current_color in ["red", "yellow", "orange"] and \
                    self.current_size > self.current_balloon_limit:
                    end_size = self.current_size
                    self.current_size = 0  # Balloon pops
                    reward = self.rew_on_pop
                    terminated = True
                    popped = True
                # elif self.current_color in ["gray", "pink"]:
                #     if self.current_size >= 20:  # Fixed size for passive trials
                #         self.current_size = 20

            else:  # Action 0: stop inflating
                if self.current_color in ["red", "yellow", "orange"]:
                    reward = self.current_size
                # elif self.current_color in ["gray", "pink"]:
                #     reward = self.colors[self.current_color]["fixed_reward"]
                terminated = True
        else:
            if self.currently_inflating and self.current_color in ["red", "yellow", "orange"] \
                and not self.passive_trial:
                if action == 1:
                    reward = self.current_size
                    self.currently_inflating = False
                    terminated = True
                if action == 0:
                    inflate = random.gauss(self.inflate_speed, self.inflate_noise)
                    self.current_size += inflate

                    if self.current_size > self.current_balloon_limit:
                        end_size = self.current_size
                        reward = self.rew_on_pop
                        self.current_size = 0  # Balloon pops
                        self.currently_inflating = False
                        terminated = True
                        popped = True
                        
            if self.currently_inflating and (self.current_color in ["gray", "pink"] or 
                                             self.passive_trial) and action == 1:
                reward = self.punish_passive
                        
            
            elif self.currently_inflating and self.current_color in ["red", "yellow", "orange"]:
                inflate = random.gauss(self.inflate_speed, self.inflate_noise)
                self.current_size += inflate
                if self.current_size > self.current_balloon_limit:
                    end_size = self.current_size
                    self.current_size = 0
                    reward = self.current_balloon_limit
                    self.currently_inflating = False
                    terminated = True
                    popped = False
                        
            elif self.currently_inflating and self.current_color in ["gray", "pink"]:
                inflate = random.gauss(self.inflate_speed, self.inflate_noise)
                self.current_size += inflate
                if self.current_size > self.current_balloon_limit:
                    end_size = self.current_size
                    self.current_size = 0
                    reward = self.colors[self.current_color]["fixed_reward"]
                    self.currently_inflating = False
                    terminated = True
                    popped = False

            else:
                if action == 1:
                    self.currently_inflating = True
                    inflate = random.gauss(self.inflate_speed, self.inflate_noise)
                    self.current_size += inflate
                else:
                    self.inflate_delay += 1
                    
            

        self.prev_action = action
        if popped:
            last_size = end_size
        else:
            last_size = self.current_size
            
        # Note on max step termination, we will still give current size worth of points
        #   since this might better allow for reaction time in meta environment
        self.current_step += 1
        if self.current_step >= self.max_steps:
            if not popped:
                reward = self.current_size
            terminated = True
            
        info = {
            'current_color': self.color_to_idx[self.current_color],
            'last_size': last_size,
            'popped': popped,
            'inflate_delay': self.inflate_delay,
            'balloon_limit': self.current_balloon_limit
        }
        return self.get_observation(), reward, terminated, truncated, info
    
    def get_observation(self):
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        # obs = torch.zeros(8, dtype=torch.float)
        if self.current_step >= self.start_wait_length:
            obs[self.color_to_idx[self.current_color]] = 1

        if self.passive_trial:
            obs[len(self.colors)] = 1
        if self.give_size:
            obs[len(self.colors)+1] = self.current_size
        if self.give_last_action:
            obs[len(self.colors)+2+self.prev_action] = 1
            
        return obs
    
    def render(self, mode="rgb_array"):
        return np.zeros((64, 64))
    
    