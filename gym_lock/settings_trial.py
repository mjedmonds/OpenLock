import numpy as np

from gym_lock.common import TwoDConfig, LeverConfig, LeverRole

UPPER = TwoDConfig(0, 15, 0)
LEFT = TwoDConfig(-15, 0, np.pi / 2)
LOWER = TwoDConfig(0, -15, -np.pi)
UPPERLEFT = TwoDConfig(-11, 11, np.pi/4)
UPPERRIGHT = TwoDConfig(11, 11, -np.pi/4)
LOWERLEFT = TwoDConfig(-11, -11, 3*np.pi / 4)
LOWERRIGHT = TwoDConfig(11, -11, 5*np.pi/4)

OLD_LEVER_CONFIGS = {
    'trial1' : [UPPERLEFT, LOWERLEFT, UPPERRIGHT],
    'trial2' : [UPPER, LOWER, LEFT],
    'trial3' : [UPPERLEFT, LOWERLEFT, LOWERRIGHT],
    'trial4' : [UPPER, UPPERLEFT, UPPERRIGHT],
    'trial5' : [UPPERLEFT, LOWERLEFT, LEFT],
    'trial6' : [LOWERLEFT, LOWER, LOWERRIGHT],
    'trial7' : [UPPERLEFT, UPPERRIGHT, LOWERRIGHT, LOWERLEFT],
    'multi-lock' : [UPPER, LOWER, LEFT],
    'full'   : [UPPERRIGHT, UPPER, UPPERLEFT, LEFT, LOWERLEFT, LOWER, LOWERRIGHT]
}

LEVER_CONFIGS = {
    # Trial 1. l0=UPPERLEFT, l1=LOWERLEFT, l2=UPPERRIGHT,
    'trial1'   : [LeverConfig(UPPERRIGHT,   LeverRole.l2,       None),
                  LeverConfig(UPPER,        LeverRole.inactive, None),
                  LeverConfig(UPPERLEFT,    LeverRole.l0,       None),
                  LeverConfig(LEFT,         LeverRole.inactive, None),
                  LeverConfig(LOWERLEFT,    LeverRole.l1,       None),
                  LeverConfig(LOWER,        LeverRole.inactive, None),
                  LeverConfig(LOWERRIGHT,   LeverRole.inactive, None)],
    # Trial 2. l0=UPPER, l1=LOWER, l2=LEFT,
    'trial2'   : [LeverConfig(UPPERRIGHT,   LeverRole.inactive, None),
                  LeverConfig(UPPER,        LeverRole.l0,       None),
                  LeverConfig(UPPERLEFT,    LeverRole.inactive, None),
                  LeverConfig(LEFT,         LeverRole.l2,       None),
                  LeverConfig(LOWERLEFT,    LeverRole.inactive, None),
                  LeverConfig(LOWER,        LeverRole.l1,       None),
                  LeverConfig(LOWERRIGHT,   LeverRole.inactive, None)],
    # Trial 3. l0=UPPERLEFT , l1=LOWERLEFT, l2=LOWERRIGHT,
    'trial3'   : [LeverConfig(UPPERRIGHT,   LeverRole.inactive, None),
                  LeverConfig(UPPER,        LeverRole.inactive, None),
                  LeverConfig(UPPERLEFT,    LeverRole.l0,       None),
                  LeverConfig(LEFT,         LeverRole.inactive, None),
                  LeverConfig(LOWERLEFT,    LeverRole.l1,       None),
                  LeverConfig(LOWER,        LeverRole.inactive, None),
                  LeverConfig(LOWERRIGHT,   LeverRole.l2,       None)],
    # Trial 4. l0=UPPER, l1=UPPERLEFT, l2=UPPERRIGHT,
    'trial4'   : [LeverConfig(UPPERRIGHT,   LeverRole.l2,       None),
                  LeverConfig(UPPER,        LeverRole.l0,       None),
                  LeverConfig(UPPERLEFT,    LeverRole.l1,       None),
                  LeverConfig(LEFT,         LeverRole.inactive, None),
                  LeverConfig(LOWERLEFT,    LeverRole.inactive, None),
                  LeverConfig(LOWER,        LeverRole.inactive, None),
                  LeverConfig(LOWERRIGHT,   LeverRole.inactive, None)],
    # Trial 5. l0=UPPERLEFT, l1=LOWERLEFT, l2=LEFT,
    'trial5'   : [LeverConfig(UPPERRIGHT,   LeverRole.inactive, None),
                  LeverConfig(UPPER,        LeverRole.inactive, None),
                  LeverConfig(UPPERLEFT,    LeverRole.l0,       None),
                  LeverConfig(LEFT,         LeverRole.l2,       None),
                  LeverConfig(LOWERLEFT,    LeverRole.l1,       None),
                  LeverConfig(LOWER,        LeverRole.inactive, None),
                  LeverConfig(LOWERRIGHT,   LeverRole.inactive, None)],
    # Trial 6. l0=LOWERLEFT, l1=LOWER, l2=LOWERRIGHT,
    'trial6'   : [LeverConfig(UPPERRIGHT,   LeverRole.inactive, None),
                  LeverConfig(UPPER,        LeverRole.inactive, None),
                  LeverConfig(UPPERLEFT,    LeverRole.inactive, None),
                  LeverConfig(LEFT,         LeverRole.inactive, None),
                  LeverConfig(LOWERLEFT,    LeverRole.l0,       None),
                  LeverConfig(LOWER,        LeverRole.l1,       None),
                  LeverConfig(LOWERRIGHT,   LeverRole.l2,       None)],
    # Trial 7. l0=UPPERLEFT, l1=UPPERRIGHT, l2=LOWERRIGHT, l3=LOWERLEFT
    'trial7'   : [LeverConfig(UPPERRIGHT,   LeverRole.l1,       None),
                  LeverConfig(UPPER,        LeverRole.inactive, None),
                  LeverConfig(UPPERLEFT,    LeverRole.l0,       None),
                  LeverConfig(LEFT,         LeverRole.inactive, None),
                  LeverConfig(LOWERLEFT,    LeverRole.l3,       None),
                  LeverConfig(LOWER,        LeverRole.inactive, None),
                  LeverConfig(LOWERRIGHT,   LeverRole.l2, None)],
    # multi-lock. l0=UPPER, l1=LOWER, l2=LEFT,
    'multi-lock': [LeverConfig(UPPERRIGHT,  LeverRole.inactive, None),
                  LeverConfig(UPPER,        LeverRole.l0,       None),
                  LeverConfig(UPPERLEFT,    LeverRole.inactive, None),
                  LeverConfig(LEFT,         LeverRole.l2,       {'lower_lim': 0.0, 'upper_lim': 2.0}),
                  LeverConfig(LOWERLEFT,    LeverRole.inactive, None),
                  LeverConfig(LOWER,        LeverRole.l1,       None),
                  LeverConfig(LOWERRIGHT,   LeverRole.inactive, None)],
    # full.
    'full'     : [LeverConfig(UPPERRIGHT,   LeverRole.l0,       None),
                  LeverConfig(UPPER,        LeverRole.l1,       None),
                  LeverConfig(UPPERLEFT,    LeverRole.l2,       None),
                  LeverConfig(LEFT,         LeverRole.l3,       None),
                  LeverConfig(LOWERLEFT,    LeverRole.l4,       None),
                  LeverConfig(LOWER,        LeverRole.l5,       None),
                  LeverConfig(LOWERRIGHT,   LeverRole.l6,       None)],
}

OLD_LEVER_OPT_PARAMS = {
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


def select_trial(trial):
    return trial, LEVER_CONFIGS[trial]


def select_random_trial(completed_trials, min_idx, max_idx):
    '''
    sets a new random
    :param completed_trials: list of trials already selected
    :param min: min value of trial index
    :param max: max value of trial index
    :return:
    '''
    trial_base = 'trial'
    rand = np.random.randint(min_idx, max_idx+1)
    trial = trial_base + str(rand)
    while trial in completed_trials:
        rand = np.random.randint(min_idx, max_idx+1)
        trial = trial_base + str(rand)

    return select_trial(trial)

