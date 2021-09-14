import gym
import pybulletgym

import numpy as np
import torch

from gym.spaces import Box
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy
from PIL import Image

from hydra.utils import get_original_cwd, to_absolute_path


class ReacherDaggerEnv(gym.Env):
    def __init__(self, frame_height=60, frame_width=80, resize_visual=True):
        self._base_env = gym.make('ReacherPyBulletEnv-v0')

        self.expert_model = SAC.load(to_absolute_path('sac_reacher_expert_longer'), env=self._base_env)
        mean_reward, std_reward = evaluate_policy(self.expert_model, self._base_env, n_eval_episodes=10, deterministic=True)
        print(f"Loaded expert with mean reward={mean_reward:.2f} +/- {std_reward}")

        self.action_space = self._base_env.action_space
        self._height = frame_height
        self._width = frame_width
        self._resize_visual = resize_visual
        self.observation_space = Box(low=0, high=255, 
                                     shape=[3, self._height, self._width],
                                     dtype=np.uint8)

        # Wrapper necessities
        self.reward_range = self._base_env.reward_range
        self.metadata = self._base_env.metadata

        # Tracking expert help provided.
        self.expert_calls = 0

    def reset(self):
        self.prop_states = np.copy(self._base_env.reset())
        visual_obs = self._base_env.render(mode='rgb_array')
        if self._resize_visual:
            visual_obs = self._resize_image_obs(visual_obs)
        return visual_obs

    def step(self, action):
        prop_obs, reward, done, info = self._base_env.step(action)
        self.prop_states = np.copy(prop_obs)
        visual_obs = self._base_env.render(mode='rgb_array')
        if self._resize_visual:
            visual_obs = self._resize_image_obs(visual_obs)
        return visual_obs, reward, done, info

    def get_expert_action(self):
        self.expert_calls += 1
        with torch.no_grad():
            action, _ = self.expert_model.predict(
                self.prop_states, deterministic=True
            )
        return action.clip(-1., 1.)

    def _resize_image_obs(self, obs, resize=True):
        if resize:
            obs_image = Image.fromarray(obs)
            resized_image = obs_image.resize(size=(self._width, self._height), resample=0)
            im = np.asarray(resized_image)
        else:
            im = obs
        # Convert from B x H x W x C to B x C x H x W
        im = im.transpose((2, 0, 1))
        return im
