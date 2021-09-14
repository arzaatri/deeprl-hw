import gym  # open ai gym
import pybulletgym  # register PyBullet enviroments with open ai gym
import time
import numpy as np
import torch

from gym.spaces import Box
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy

from hydra.utils import get_original_cwd, to_absolute_path

env = gym.make('ReacherPyBulletEnv-v0')
print(env.reset())
print(env.reset())
expert_model = SAC.load(to_absolute_path('sac_reacher_expert_longer'), env=env)
print(env.action_space)
prop_obs, reward, done, info = env.step(env.action_space.sample())
with torch.no_grad():
            action, _ = expert_model.predict(
                prop_states, deterministic=True
            )
print(action.clip(-1., 1.))