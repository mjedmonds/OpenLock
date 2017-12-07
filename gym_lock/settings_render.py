import numpy as np

from gym_lock.common import TwoDConfig, Color
from gym_lock.scenarios.multi_lock import MultiLockScenario
from gym_lock.scenarios.CE3 import CommonEffect3Scenario
from gym_lock.scenarios.CC3 import CommonCause3Scenario
from gym_lock.scenarios.CE4 import CommonEffect4Scenario
from gym_lock.scenarios.CC4 import CommonCause4Scenario

# this must be a global variable because main.py and arm_lock_env.py need access to the same object
# CURRENT_SCENARIO = None

# used as the current scenario for the environment and world_def setup
# SCENARIO = 'CE3'
# SCENARIO = 'CC3'
# SCENARIO = 'CE4'
# SCENARIO = 'CC4'
# SCENARIO = 'multi-lock'


def select_scenario(scenario):
    CURRENT_SCENARIO = None
    if scenario == 'CE3':
        CURRENT_SCENARIO = CommonEffect3Scenario()
    elif scenario == 'CC3':
        CURRENT_SCENARIO = CommonCause3Scenario()
    elif scenario == 'CE4':
        CURRENT_SCENARIO = CommonEffect4Scenario()
    elif scenario == 'CC4':
        CURRENT_SCENARIO = CommonCause4Scenario()
    elif scenario == 'multi-lock':
        CURRENT_SCENARIO = MultiLockScenario()
    else:
        raise ValueError('Invalid scenario chosen in settings_render.py: %s' % scenario)
    return CURRENT_SCENARIO


RENDER_SETTINGS = {
    "RENDER_CLK_DIV": 25,
    "VIEWPORT_W": 800,
    "VIEWPORT_H": 800,
    "SCALE": 15.0,
    "DRAW_SHAPES": True,
    "DRAW_JOINTS": True,
    "DRAW_AABB": False, # TODO
    "DRAW_MARKERS" : True,
    "COLORS" : {
        'active': Color(0.6, 0.6, 0.6),
        'static': Color(0.5, 0.9, 0.5),
        'kinematic': Color(0.5, 0.5, 0.9),
        'asleep': Color(0.6, 0.6, 0.6),
        'default': Color(0.6, 0.6, 0.6),
        'rev_joint' : Color(1.0, 0, 0),
        'pris_joint' : Color(0.35, 0.35, 0.35),
        'dist_joint' : Color(0.0, 0.0, 1.0),
        'weld_joint' : Color(0, 0, 0)
    }
}

BOX2D_SETTINGS = {
    "FPS": 500,
    "VEL_ITERS": 10,
    "POS_ITERS": 10,
    "POS_PID_CLK_DIV": 10,
    "STATE_MACHINE_CLK_DIV" : 10,

    # Makes physics results more accurate (see Box2D wiki)
    "ENABLE_WARM_START": True,
    "ENABLE_CONTINUOUS": True,  # Calculate time of impact
    "ENABLE_SUBSTEP": False,

    "ARM_LENGTH": 5.0,
    "ARM_WIDTH": 1.0,
    "INITIAL_BASE_CONFIG": TwoDConfig(0, 0, 0),
    "INITIAL_THETA_VECTOR": [np.pi, np.pi / 2, np.pi / 2, 0, np.pi / 2],
}

ENV_SETTINGS = {
    "PID_POS_CONV_TOL": 0.005,
    "PID_VEL_CONV_TOL": 0.05,
    "PID_CONV_MAX_STEPS": 500,
    "INVK_CONV_MAX_STEPS": 2000,
    "INVK_DLS_LAMBDA": 0.75,
    "INVK_CONV_TOL": 0.001,
    "PATH_INTERP_STEP_DELTA": 0.05
}

