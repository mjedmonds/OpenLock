from gym.envs.registration import register

register(id="openlock-v1", entry_point="openlock.envs:OpenLockEnv")
