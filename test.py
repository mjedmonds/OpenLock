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
targ = KinematicChain(base, generate_four_arm(-np.pi, 0, 0, 0))
idx = 0

done = False
import time
for i in range(10000000):
    # if env.world_def.clock % 10 == 0:
    #     env.render()
    obs, rew, done, info = env.step(KinematicChain(base, generate_four_arm(np.pi/4, -np.pi/4, 0, 0)))
    obs, rew, done, info = env.step(KinematicChain(base, generate_four_arm(3*np.pi/8, -6*np.pi/8, 6*np.pi/8, -6*np.pi/8)))
    obs, rew, done, info = env.step(KinematicChain(base, generate_four_arm(-np.pi/4, np.pi/4, 0, 0)))
    obs, rew, done, info = env.step(KinematicChain(base, generate_four_arm(3*np.pi/8, -6*np.pi/8, 6*np.pi/8, -6*np.pi/8)))

    # env.step(False)
    if done:
        exit()
    env.render()


