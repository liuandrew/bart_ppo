import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch

import random

class BartEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "video.frames_per_second": 24}
    def __init__(self, colors_used=1, hold_to_inflate=True, 
                 give_last_action=True, give_size=True,
                 inflate_speed=0.05, inflate_noise=0.02, rew_on_pop=0,
                 pop_noise=0.05):
        """
        Action space: 3 actions
            hold_to_inflate: if True, action 1 inflates, action 0 lets go
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
        """
        super(BartEnv, self).__init__()

        self.colors = {
            "red": {"mean": 0.2},
            "yellow": {"mean": 0.5},
            "orange": {"mean": 0.8},
            "gray": {"fixed_reward": 0},
            "purple": {"fixed_reward": 1}
        }
        self.color_to_idx = {"red": 0, "yellow": 1, "orange": 2,
                             "gray": 3, "purple": 4}
        self.idx_to_color = {0: "red", 1: "yellow", 2: "orange",
                             3: "gray", 4: "purple"}
        # Env setup parameters
        self.colors_used = colors_used
        self.hold_to_inflate = hold_to_inflate
        self.give_last_action = give_last_action
        self.give_size = give_size
        
        # Tweak parameters
        self.inflate_speed = inflate_speed
        self.inflate_noise = inflate_noise
        self.rew_on_pop = rew_on_pop
        self.pop_noise = pop_noise

        # self.observation_space = spaces.Tuple((
        #     spaces.Discrete(len(self.colors)),  # Color index
        #     spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),  # Current size
        #     spaces.Discrete(2)  # Previous action
        # ))
        self.observation_space = spaces.Box(low=0, high=1, shape=(5+1+2,))
        self.action_space = spaces.Discrete(2)  # 0: do nothing, 1: hold inflate button


    def reset(self, seed=None, options={}):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        if self.colors_used <= 1:
            self.current_color = "yellow"
        else:
            self.current_color = self.idx_to_color[random.choice(range(self.colors_used))]
        self.current_size = 0.0
        self.prev_action = 0

        self.currently_inflating = False # used for stop/start version
        
        # Pick a pop size
        mean = self.colors[self.current_color]["mean"]
        self.current_balloon_limit = random.gauss(mean, self.pop_noise)

        return self.get_observation(), {}

    def step(self, action):
        terminated = False
        truncated = False
        reward = 0

        if self.hold_to_inflate:
            if action == 1:  # Hold inflate button
                inflate = random.gauss(self.inflate_speed, self.inflate_noise)
                self.current_size += inflate

                if self.current_color in ["red", "yellow", "orange"] and \
                    self.current_size > self.current_balloon_limit:
                    self.current_size = 0  # Balloon pops
                    reward = self.rew_on_pop
                    terminated = True
                # elif self.current_color in ["gray", "purple"]:
                #     if self.current_size >= 20:  # Fixed size for passive trials
                #         self.current_size = 20

            else:  # Action 0: stop inflating
                if self.current_color in ["red", "yellow", "orange"]:
                    reward = self.current_size
                # elif self.current_color in ["gray", "purple"]:
                #     reward = self.colors[self.current_color]["fixed_reward"]
                terminated = True
        else:
            if self.currently_inflating:
                if action == 1:
                    if self.current_color in ["red", "yellow", "orange"]:
                        reward = self.current_size
                        self.currently_inflating = False
                        terminated = True
                if action == 0:
                    inflate = random.gauss(self.inflate_speed, self.inflate_noise)
                    self.current_size += inflate
                    if self.current_color in ["red", "yellow", "orange"] and \
                        self.current_size > self.current_balloon_limit:
                        self.current_size = 0  # Balloon pops
                        reward = self.rew_on_pop
                        self.currently_inflating = False
                        terminated = True
            else:
                if action == 1:
                    self.currently_inflating = True
                    inflate = random.gauss(self.inflate_speed, self.inflate_noise)
                    self.current_size += inflate
                
        self.prev_action = action
        return self.get_observation(), reward, terminated, truncated, {}
    
    def get_observation(self):
        obs = np.zeros(8, dtype=np.float32)
        # obs = torch.zeros(8, dtype=torch.float)
        obs[self.color_to_idx[self.current_color]] = 1
        if self.give_size:
            obs[len(self.colors)] = self.current_size
        if self.give_last_action:
            obs[len(self.colors)+1+self.prev_action] = 1
            
        return obs
    
    def render(self, mode="rgb_array"):
        return np.zeros((64, 64))
    
    