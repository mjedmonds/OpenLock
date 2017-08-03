import gym
import Box2D as b2
from gym_lock.envs import PointMassLockEnv #import LockEnv
import time
from gym_lock.types import TwoDConfig

env = gym.make('arm_lock-v0')
env.reset()



for i in range(10000000):
    env.render()
    if i > 50 and i % 5  == 0:
        print 'hello'
        env.step(TwoDConfig([15, 0], 0))
    else:
        time.sleep(0.025)
        env.step(False)

