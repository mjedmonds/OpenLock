import gym
import numpy as np
from gym_lock.envs import ArmLockEnv
from gym_lock.common import Action

env = gym.make('arm_lock-v0')

while (True):
    # if env.viewer.desired_config:
    #     # TODO: messy interface for stepping, add adapter or simplyify
    #     # TODO: add ability to move base
    #     obs, rew, done, info = env.step(Action('goto', env.viewer.desired_config))
    #     # TODO: abstraction
    #     env.viewer.desired_config = None
    #     print obs['FSM_STATE']
    # else:
    #     env.step(False)


    obs, rew, done, info = env.step(Action('push_perp', ('l0', 4))) # unlock l0
    print obs['FSM_STATE']

    obs, rew, done, info = env.step(Action('push_perp', ('l1', 4))) # unlock l1
    print obs['FSM_STATE']

    obs, rew, done, info = env.step(Action('push_perp', ('l2', 4))) # try to unlock l2, but it doesn't budge!
    print obs['FSM_STATE']

    obs, rew, done, info = env.step(Action('pull_perp', ('l2', 4))) # try pulling l2 instead
    print obs['FSM_STATE']

    obs, rew, done, info = env.step(Action('push_perp', ('door', 4))) # open the door
    print obs['FSM_STATE']



    # env.step(Action('rest', None))
    # for i in range(0, 500):
    #     env.step(None)
