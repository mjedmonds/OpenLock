import gym
import Box2D as b2
from gym_lock.envs import PointMassLockEnv #import LockEnv
import time
env = gym.make('arm_lock-v0')
env.reset()

import numpy as np

def generate_valid_config(t1, t2, t3):
    joint_config = [{'name' : '0-0'},
                    {'name' : '0+1-', 'theta' : t1, 'screw' : [0, 0, 0, 0, 0, 1]},
                    {'name' : '1-1+', 'x' : 8},
                    {'name' : '1+2-', 'theta' : t2, 'screw' : [0, 0, 0, 0, 0, 1]}, 
                    {'name' : '2-2+', 'x' : 8},
                    {'name' : '2+3-', 'theta' : t3, 'screw' : [0, 0, 0, 0, 0, 1]},
                    {'name' : '3-3+', 'x' : 8}]
    return joint_config

c1 = generate_valid_config(0, np.pi / 2, 0)
c2 = generate_valid_config(0, 0, np.pi / 2)
c3 = generate_valid_config(0, np.pi / 2, np.pi / 2)
c4 = generate_valid_config(np.pi / 2, 0, 0)

c5 = generate_valid_config(np.pi / 2, np.pi / 2, 0)

for i in range(10000000):
    print i
    env.render()
    if i % 10 == 0:
        env.step(c1)
        #if i > 750:
        #    env.step(c2)
        #if i > 1250:
        #    env.step(c3)
        #if i > 1750:
        #    env.step(c4)
    else:
        env.step(False)

