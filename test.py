import gym
from gym_lock.envs import ArmLockEnv

env = gym.make('arm_lock-v0')

while (True):
    if env.viewer.desired_config:
        # TODO: messy interface for stepping, add adapter or simplyify
        # TODO: add ability to move base
        obs, rew, done, info = env.step(env.viewer.desired_config)
        # TODO: abstraction
        env.viewer.desired_config = None
    else:
        env.step(False)
