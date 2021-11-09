'''This script serves as opengym environment wrapper for our purposes'''
import gym
import numpy
import torch

class EnvWrapper():
    def __init__(self, opt):
        # Parameters
        self.state_size = opt['env']['state_size']
        self.action_size = opt['env']['action_size']
        self.skip_frames = opt['env']['skip_frames']

        # Load environimetn
        self.env = gym.make(opt['env']['env_name'])

    def step(self, action):
        """Take an action and returns the next observation, reward and so on."""
        total_reward = 0
        for _ in range(self.skip_frames+1):
            state, reward, is_done, _ = self.env.step(action)
            total_reward += reward
            if is_done:
                break
        return state, total_reward, is_done 

    def render(self):
        self.env.render()

    def reset(self):
        return self.env.reset()
        