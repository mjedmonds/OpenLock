import gym
import Box2D as b2
from gym_lock.envs import PointMassLockEnv #import LockEnv
import time
env = gym.make('arm_lock-v0')
env.reset()

import numpy as np
from numpy import pi

from gym_lock.kine import KinematicChain, generate_four_arm, TwoDKinematicTransform

def gen_theta():
    return (np.random.ranf() - 0.5) * 2 * np.pi

# configs = []
# for i in range(0, 10):
#     configs.append(generate_valid_config(gen_theta(), gen_theta(), gen_theta()))

base=TwoDKinematicTransform()
targ = KinematicChain(base, generate_four_arm(pi/2, 0, 0, pi/2))
idx = 0

done = False

for i in range(10000000):
    if env.world_def.clock % 10 == 0:
        env.render()
    env.step(False)
    # if done:
    #     env.step(False)
    # else:
    #     obs, rew, done, info = env.step(targ)
    #     print done




        # if done:
    #     idx = idx + 1
    #     targ = KinematicChain(configs[idx])
    #     print 'new target'
    #     print configs[idx]

    # if 0 == 0:
    #     if i < s[0]:
    #         env.step(c1)
    #     if i > s[0] and i < s[1]:
    #         env.step(c2)
    #     if i > s[1] and i < s[2]:
    #         env.step(c3)
    #     if i > s[2] and i < s[3]:
    #         env.step(c4)
    # else:
    #     env.step(False)

