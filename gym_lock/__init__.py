from gym.envs.registration import register

register(
    id='arm_lock-v0',
    entry_point='gym_lock.envs:ArmLockEnv',
)
