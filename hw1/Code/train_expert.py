import gym  # open ai gym
import pybulletgym  # register PyBullet enviroments with open ai gym
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy

env = gym.make('ReacherPyBulletEnv-v0')

model = SAC("MlpPolicy", env, verbose=1)

# Random Agent, before training
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, deterministic=True)
print(f"Before training mean_reward={mean_reward:.2f} +/- {std_reward}")

model.learn(total_timesteps=300_000, log_interval=10)

mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, deterministic=True)
print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")

model.save("sac_reacher_expert_longer_trained")

model = SAC.load("sac_reacher_expert_longer_trained")
# Trained Agent, after training
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, deterministic=True)
print(f"After training mean_reward={mean_reward:.2f} +/- {std_reward}")