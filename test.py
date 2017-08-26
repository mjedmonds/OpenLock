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
    env.step(Action('push_perp', (env.world_def.lock, 2)))
    env.step(Action('move_end_frame', (-5, 0, 0)))
    # env.step(Action('rest', None))
    # env.step(Action('push_perp', (env.world_def.door, 4)))
    # env.step(Action('move_end_frame', (-10, 0, 0)))
    env.step(Action('pull_perp', (env.world_def.lock, 2)))



    env.step(Action('rest', None))
    for i in range(0, 500):
        env.step(None)
