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
                    {'name' : '1-1+', 'x' : 5},
                    {'name' : '1+2-', 'theta' : t2, 'screw' : [0, 0, 0, 0, 0, 1]}, 
                    {'name' : '2-2+', 'x' : 5},
                    {'name' : '2+3-', 'theta' : t3, 'screw' : [0, 0, 0, 0, 0, 1]},
                    {'name' : '3-3+', 'x' : 5}]
    return joint_config

c1 = generate_valid_config(0, np.pi / 2, 0)
c2 = generate_valid_config(0, 0, np.pi / 2)
c3 = generate_valid_config(0, np.pi / 2, np.pi / 2)
c4 = generate_valid_config(-np.pi/2, np.pi/2, -np.pi/2)

c5 = generate_valid_config(np.pi / 2, 0, 0)

s = [1000, 2000, 3000 , 4000, 5000, 6000]


for i in range(10000000):
    env.render()
    if 0 == 0:
        if i < s[0]:
            env.step(c1)
        if i > s[0] and i < s[1]:
            env.step(c2)
        if i > s[1] and i < s[2]:
            env.step(c3)
        if i > s[2] and i < s[3]:
            env.step(c4)

    else:
        env.step(False)

