
from gym.envs.registration import register

register(
    id='lock-v0',
    entry_point='gym_lock.envs:LockEnv',
)
