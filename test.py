import gym
import Box2D as b2
from gym_lock.envs import PointMassLockEnv #import LockEnv
import time
env = gym.make('arm_lock-v0')
env.reset()
for i in range(10000000):
    env.render()
    time.sleep(0.1)
    env.step((0,-1)) # take a random action
    if i == 1:
        time.sleep(4)
