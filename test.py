import gym
import Box2D as b2
from gym_lock.envs import LockEnv #import LockEnv

env = gym.make('lock-v0')
env.reset()
for _ in range(1000):
    env.render()
    env.step((0,-1)) # take a random action
