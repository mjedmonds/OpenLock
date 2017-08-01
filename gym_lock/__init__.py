
from gym.envs.registration import register

register(
    id='point_mass_lock-v0',
    entry_point='gym_lock.envs:PointMassLockEnv',
)

register(
    id='arm_lock-v0',
    entry_point='gym_lock.envs:ArmLockEnv',
)
