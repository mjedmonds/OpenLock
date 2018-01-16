import numpy as np

from gym_lock.common import TwoDConfig, Color


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
    "VEL_ITERS": 3,
    "POS_ITERS": 3,
    "POS_PID_CLK_DIV": 10,
    "VEL_PID_CLK_DIV": 1,
    "STATE_MACHINE_CLK_DIV": 5,

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

