import gym  # open ai gym
import pybulletgym  # register PyBullet enviroments with open ai gym
import time

env = gym.make('ReacherPyBulletEnv-v0')
env.render()
env.reset()
for _ in range(500):
    env.render() # call this before env.reset, if you want a window showing the environment
    time.sleep(0.017)
    env.step(env.action_space.sample())
    if _ == 2:
        print(env.action_space)
        print(env.observation_space)
env.close()  # should return a state vector if everything worked