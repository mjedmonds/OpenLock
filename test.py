import gym
import Box2D as b2
from gym_lock.envs import PointMassLockEnv #import LockEnv
import time
env = gym.make('arm_lock-v0')
env.reset()
for i in range(10000000):
    env.render()
    if i > 300:
        env.step(True)
    else:
        env.step(False)

