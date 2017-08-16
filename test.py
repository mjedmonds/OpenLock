import gym
import Box2D as b2
from gym_lock.envs import PointMassLockEnv #import LockEnv
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
while(True):
    # if env.world_def.clock % 10 == 0:
    #     env.render()

    # env.step(KinematicChain(base, generate_random_config()))
    env.step(KinematicChain(base, [KinematicLink(TwoDKinematicTransform(),
                                                 TwoDKinematicTransform(),
                                                 TwoDKinematicTransform(name='1-1+', x=16, y=5, theta=0),
                                                 None)]))
    # exit()
    env.step(KinematicChain(base, [KinematicLink(TwoDKinematicTransform(),
                                                 TwoDKinematicTransform(),
                                                 TwoDKinematicTransform(name='1-1+', x=5, y=0, theta=0),
                                                 None)]))
    env.step(KinematicChain(base, [KinematicLink(TwoDKinematicTransform(),
                                                 TwoDKinematicTransform(),
                                                 TwoDKinematicTransform(name='1-1+', x=15, y=-7.5, theta=0),
                                                 None)]))
    env.step(KinematicChain(base, [KinematicLink(TwoDKinematicTransform(),
                                                 TwoDKinematicTransform(),
                                                 TwoDKinematicTransform(name='1-1+', x=16, y=-7.5, theta=0),
                                                 None)]))
    env.step(KinematicChain(base, [KinematicLink(TwoDKinematicTransform(),
                                                 TwoDKinematicTransform(),
                                                 TwoDKinematicTransform(name='1-1+', x=17, y=-7.5, theta=0),

                                                 None)]))
    print env.world_def.lock_joint.translation
    import time
    time.sleep(3)
    env.step(KinematicChain(base, [KinematicLink(TwoDKinematicTransform(),
                                                 TwoDKinematicTransform(),
                                                 TwoDKinematicTransform(name='1-1+', x=5, y=0, theta=0),
                                                 None)]))
    env.render()

