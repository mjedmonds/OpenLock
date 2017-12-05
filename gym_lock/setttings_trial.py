import numpy as np

from gym_lock.common import TwoDConfig

TRIAL = 'trial6'

UPPER = TwoDConfig(0, 15, 0)
LEFT = TwoDConfig(-15, 0, np.pi / 2)
LOWER = TwoDConfig(0, -15, -np.pi)
UPPERLEFT = TwoDConfig(-11, 11, np.pi/4)
UPPERRIGHT = TwoDConfig(11, 11, -np.pi/4)
LOWERLEFT = TwoDConfig(-11, -11, 3*np.pi / 4)
LOWERRIGHT = TwoDConfig(11, -11, 5*np.pi/4)


LEVER_CONFIGS = {
    'trial1' : [UPPERLEFT, LOWERLEFT, UPPERRIGHT],
    'trial2' : [UPPER, LOWER, LEFT],
    'trial3' : [UPPERLEFT, LOWERLEFT, LOWERRIGHT],
    'trial4' : [UPPER, UPPERLEFT, UPPERRIGHT],
    'trial5' : [UPPERLEFT, LOWERLEFT, LEFT],
    'trial6' : [LOWERLEFT, LOWER, LOWERRIGHT],
    'trial7' : [UPPERLEFT, UPPERRIGHT, LOWERRIGHT, LOWERLEFT],
    'full'   : [UPPERRIGHT, UPPER, UPPERLEFT, LEFT, LOWERLEFT, LOWER, LOWERRIGHT]
}

LEVER_OPT_PARAMS = {
    'trial1' : [None, None, None],
    'trial2' : [None, None, None],
    'trial3' : [None, None, None],
    'trial4' : [None, None, None],
    'trial5' : [None, None, None],
    'trial6' : [None, None, None],
    'trial7' : [None, None, None, None],
    'full'   : [None, None, None, None, None, None],
    'multi-lock' : [None, None, {'lower_lim': 0.0, 'upper_lim': 2.0}]
}

CURRENT_LEVER_CONFIG = LEVER_CONFIGS[TRIAL]
CURRENT_LEVER_OPT_PARAMS = LEVER_OPT_PARAMS[TRIAL]