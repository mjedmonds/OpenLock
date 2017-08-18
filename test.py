import gym
import Box2D as b2
from gym_lock.envs import ArmLockEnv
import time
env = gym.make('arm_lock-v0')
env.reset()

import numpy as np
import random as ran
from numpy import pi

from gym_lock.kine import KinematicChain, generate_four_arm, TwoDKinematicTransform, KinematicLink

def gen_theta():
    return (np.random.ranf() - 0.5) * 2 * np.pi

# configs = []
# for i in range(0, 10):
#     configs.append(generate_valid_config(gen_theta(), gen_theta(), gen_theta()))

base=TwoDKinematicTransform()

def generate_random_config():
    t1, t2, t3, t4, t5 = [(ran.random() - 0.5) * 2 * np.pi for i in range(0, 5)]
    print t1, t2, t3, t4
    return generate_four_arm(t1, t2, t3, t4, t5)


done = False
import time

desired_config = None

while(True):
    # if env.world_def.clock % 10 == 0:
    #     env.render()

    # env.step(KinematicChain(base, generate_random_config()))
    if desired_config:
        env.step(KinematicChain(base, [KinematicLink(TwoDKinematicTransform(),
                                                     TwoDKinematicTransform(),
                                                     TwoDKinematicTransform(name='1-1+',
                                                                            x=desired_config.x,
                                                                            y=desired_config.y,
                                                                            theta=desired_config.theta),
                                                     None)]))
        desired_config = None
        env.viewer.desired_config = None

    else:
        env.step(False)
        if env.viewer.desired_config:
            desired_config = env.viewer.desired_config

