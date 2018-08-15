import numpy as np

from openlock.common import TwoDConfig, LeverConfig, LeverRoleEnum, LeverPositionEnum

NUM_LEVERS = 7

ATTEMPT_LIMIT = 30
ACTION_LIMIT = 3

THREE_LEVER_TRIALS = ['trial1', 'trial2', 'trial3', 'trial4', 'trial5', 'trial6']
FOUR_LEVER_TRIALS = ['trial7', 'trial8', 'trial9', 'trial10', 'trial11']

PARAMS = {
    'CE3-CE4': {
        'train_num_trials': 6,
        'train_scenario_name': 'CE3',
        'train_attempt_limit': ATTEMPT_LIMIT,
        'train_action_limit': ACTION_LIMIT,
        'test_num_trials': 1,
        'test_scenario_name': 'CE4',
        'test_attempt_limit': ATTEMPT_LIMIT,
        'test_action_limit': ACTION_LIMIT
    },
    'CE3-CC4': {
        'train_num_trials': 6,
        'train_scenario_name': 'CE3',
        'train_attempt_limit': ATTEMPT_LIMIT,
        'train_action_limit': ACTION_LIMIT,
        'test_num_trials': 1,
        'test_scenario_name': 'CC4',
        'test_attempt_limit': ATTEMPT_LIMIT,
        'test_action_limit': ACTION_LIMIT
    },
    'CC3-CE4': {
        'train_num_trials': 6,
        'train_scenario_name': 'CC3',
        'train_attempt_limit': ATTEMPT_LIMIT,
        'train_action_limit': ACTION_LIMIT,
        'test_num_trials': 1,
        'test_scenario_name': 'CE4',
        'test_attempt_limit': ATTEMPT_LIMIT,
        'test_action_limit': ACTION_LIMIT
    },
    'CC3-CC4': {
        'train_num_trials': 6,
        'train_scenario_name': 'CC3',
        'train_attempt_limit': ATTEMPT_LIMIT,
        'train_action_limit': ACTION_LIMIT,
        'test_num_trials': 1,
        'test_scenario_name': 'CC4',
        'test_attempt_limit': ATTEMPT_LIMIT,
        'test_action_limit': ACTION_LIMIT
    },
    'CC4': {
        'train_num_trials': 5,
        'train_scenario_name': 'CC4',
        'train_attempt_limit': ATTEMPT_LIMIT,
        'train_action_limit': ACTION_LIMIT,
        'test_num_trials': 0,
        'test_scenario_name': None,
        'test_attempt_limit': ATTEMPT_LIMIT,
        'test_action_limit': ACTION_LIMIT
    },
    'CE4': {
        'train_num_trials': 5,
        'train_scenario_name': 'CE4',
        'train_attempt_limit': ATTEMPT_LIMIT,
        'train_action_limit': ACTION_LIMIT,
        'test_num_trials': 0,
        'test_scenario_name': None,
        'test_attempt_limit': ATTEMPT_LIMIT,
        'test_action_limit': ACTION_LIMIT
    },
    'testing': {
        'train_num_trials': 1,
        'train_scenario_name': 'CC3',
        'train_attempt_limit': ATTEMPT_LIMIT,
        'train_action_limit': ACTION_LIMIT,
        'test_scenario_name': None,
        'test_attempt_limit': ATTEMPT_LIMIT,
        'test_action_limit': ACTION_LIMIT
    }
}

# maps arbitrary indices to parameter settings strings
IDX_TO_PARAMS = [
    'CE3-CE4',
    'CE3-CC4',
    'CC3-CE4',
    'CC3-CC4',
    'CE4',
    'CC4'
]

# mapping from 2dconfigs to position indices
CONFIG_TO_IDX = {
    LeverPositionEnum.UPPERRIGHT.config: 0,
    LeverPositionEnum.UPPER.config:      1,
    LeverPositionEnum.UPPERLEFT.config:  2,
    LeverPositionEnum.LEFT.config:       3,
    LeverPositionEnum.LOWERLEFT.config:  4,
    LeverPositionEnum.LOWER.config:      5,
    LeverPositionEnum.LOWERRIGHT.config: 6
}

# mapping from position indices to position names
IDX_TO_POSITION = {
    0: 'UPPERRIGHT',
    1: 'UPPER',
    2: 'UPPERLEFT',
    3: 'LEFT',
    4: 'LOWERLEFT',
    5: 'LOWER',
    6: 'LOWERRIGHT',
}

# mapping from position names to position indices
POSITION_TO_IDX = {
    'UPPERRIGHT':   0,
    'UPPER':        1,
    'UPPERLEFT':    2,
    'LEFT':         3,
    'LOWERLEFT':    4,
    'LOWER':        5,
    'LOWERRIGHT':   6,
}

LEVER_CONFIGS = {
    # Trial 1. l0=LeverPositionEnum.UPPERLEFT, l1=LeverPositionEnum.LOWERLEFT, l2=LeverPositionEnum.UPPERRIGHT,
    'trial1'   : [LeverConfig(LeverPositionEnum.UPPERRIGHT,   LeverRoleEnum.l0,       None),
                  LeverConfig(LeverPositionEnum.UPPER,        LeverRoleEnum.inactive, None),
                  LeverConfig(LeverPositionEnum.UPPERLEFT,    LeverRoleEnum.l2,       None),
                  LeverConfig(LeverPositionEnum.LEFT,         LeverRoleEnum.inactive, None),
                  LeverConfig(LeverPositionEnum.LOWERLEFT,    LeverRoleEnum.l1,       None),
                  LeverConfig(LeverPositionEnum.LOWER,        LeverRoleEnum.inactive, None),
                  LeverConfig(LeverPositionEnum.LOWERRIGHT,   LeverRoleEnum.inactive, None)],
    # Trial 2. l0=LeverPositionEnum.UPPER, l1=LeverPositionEnum.LOWER, l2=LEFT,
    'trial2'   : [LeverConfig(LeverPositionEnum.UPPERRIGHT,   LeverRoleEnum.inactive, None),
                  LeverConfig(LeverPositionEnum.UPPER,        LeverRoleEnum.l2,       None),
                  LeverConfig(LeverPositionEnum.UPPERLEFT,    LeverRoleEnum.inactive, None),
                  LeverConfig(LeverPositionEnum.LEFT,         LeverRoleEnum.l0,       None),
                  LeverConfig(LeverPositionEnum.LOWERLEFT,    LeverRoleEnum.inactive, None),
                  LeverConfig(LeverPositionEnum.LOWER,        LeverRoleEnum.l1,       None),
                  LeverConfig(LeverPositionEnum.LOWERRIGHT,   LeverRoleEnum.inactive, None)],
    # Trial 3. l0=LeverPositionEnum.UPPERLEFT , l1=LeverPositionEnum.LOWERLEFT, l2=LeverPositionEnum.LOWERRIGHT,
    'trial3'   : [LeverConfig(LeverPositionEnum.UPPERRIGHT,   LeverRoleEnum.inactive, None),
                  LeverConfig(LeverPositionEnum.UPPER,        LeverRoleEnum.inactive, None),
                  LeverConfig(LeverPositionEnum.UPPERLEFT,    LeverRoleEnum.l1,       None),
                  LeverConfig(LeverPositionEnum.LEFT,         LeverRoleEnum.inactive, None),
                  LeverConfig(LeverPositionEnum.LOWERLEFT,    LeverRoleEnum.l2,       None),
                  LeverConfig(LeverPositionEnum.LOWER,        LeverRoleEnum.inactive, None),
                  LeverConfig(LeverPositionEnum.LOWERRIGHT,   LeverRoleEnum.l0,       None)],
    # Trial 4. l0=LeverPositionEnum.UPPER, l1=LeverPositionEnum.UPPERLEFT, l2=LeverPositionEnum.UPPERRIGHT,
    'trial4'   : [LeverConfig(LeverPositionEnum.UPPERRIGHT,   LeverRoleEnum.l0,       None),
                  LeverConfig(LeverPositionEnum.UPPER,        LeverRoleEnum.l2,       None),
                  LeverConfig(LeverPositionEnum.UPPERLEFT,    LeverRoleEnum.l1,       None),
                  LeverConfig(LeverPositionEnum.LEFT,         LeverRoleEnum.inactive, None),
                  LeverConfig(LeverPositionEnum.LOWERLEFT,    LeverRoleEnum.inactive, None),
                  LeverConfig(LeverPositionEnum.LOWER,        LeverRoleEnum.inactive, None),
                  LeverConfig(LeverPositionEnum.LOWERRIGHT,   LeverRoleEnum.inactive, None)],
    # Trial 5. l0=LeverPositionEnum.UPPERLEFT, l1=LeverPositionEnum.LOWERLEFT, l2=LEFT,
    'trial5'   : [LeverConfig(LeverPositionEnum.UPPERRIGHT,   LeverRoleEnum.inactive, None),
                  LeverConfig(LeverPositionEnum.UPPER,        LeverRoleEnum.inactive, None),
                  LeverConfig(LeverPositionEnum.UPPERLEFT,    LeverRoleEnum.l2,       None),
                  LeverConfig(LeverPositionEnum.LEFT,         LeverRoleEnum.l0,       None),
                  LeverConfig(LeverPositionEnum.LOWERLEFT,    LeverRoleEnum.l1,       None),
                  LeverConfig(LeverPositionEnum.LOWER,        LeverRoleEnum.inactive, None),
                  LeverConfig(LeverPositionEnum.LOWERRIGHT,   LeverRoleEnum.inactive, None)],
    # Trial 6. l0=LeverPositionEnum.LOWERLEFT, l1=LeverPositionEnum.LOWER, l2=LeverPositionEnum.LOWERRIGHT,
    'trial6'   : [LeverConfig(LeverPositionEnum.UPPERRIGHT,   LeverRoleEnum.inactive, None),
                  LeverConfig(LeverPositionEnum.UPPER,        LeverRoleEnum.inactive, None),
                  LeverConfig(LeverPositionEnum.UPPERLEFT,    LeverRoleEnum.inactive, None),
                  LeverConfig(LeverPositionEnum.LEFT,         LeverRoleEnum.inactive, None),
                  LeverConfig(LeverPositionEnum.LOWERLEFT,    LeverRoleEnum.l2,       None),
                  LeverConfig(LeverPositionEnum.LOWER,        LeverRoleEnum.l1,       None),
                  LeverConfig(LeverPositionEnum.LOWERRIGHT,   LeverRoleEnum.l0,       None)],
    # Trial 7. l0=LeverPositionEnum.LOWERLEFT, l1=LeverPositionEnum.UPPERRIGHT, l2=LeverPositionEnum.LOWERRIGHT, l3=LeverPositionEnum.UPPERLEFT
    'trial7'   : [LeverConfig(LeverPositionEnum.UPPERRIGHT,   LeverRoleEnum.l1,       None),
                  LeverConfig(LeverPositionEnum.UPPER,        LeverRoleEnum.inactive, None),
                  LeverConfig(LeverPositionEnum.UPPERLEFT,    LeverRoleEnum.l3,       None),
                  LeverConfig(LeverPositionEnum.LEFT,         LeverRoleEnum.inactive, None),
                  LeverConfig(LeverPositionEnum.LOWERLEFT,    LeverRoleEnum.l0,       None),
                  LeverConfig(LeverPositionEnum.LOWER,        LeverRoleEnum.inactive, None),
                  LeverConfig(LeverPositionEnum.LOWERRIGHT,   LeverRoleEnum.l2,       None)],
    # Trial 8. l0=LeverPositionEnum.UPPERRIGHT, l1=LeverPositionEnum.UPPER, l2=LeverPositionEnum.UPPERLEFT, l3=LEFT
    'trial8'   : [LeverConfig(LeverPositionEnum.UPPERRIGHT,   LeverRoleEnum.l0,       None),
                  LeverConfig(LeverPositionEnum.UPPER,        LeverRoleEnum.l1,       None),
                  LeverConfig(LeverPositionEnum.UPPERLEFT,    LeverRoleEnum.l2,       None),
                  LeverConfig(LeverPositionEnum.LEFT,         LeverRoleEnum.l3,       None),
                  LeverConfig(LeverPositionEnum.LOWERLEFT,    LeverRoleEnum.inactive, None),
                  LeverConfig(LeverPositionEnum.LOWER,        LeverRoleEnum.inactive, None),
                  LeverConfig(LeverPositionEnum.LOWERRIGHT,   LeverRoleEnum.inactive, None)],
    # Trial 9. l0=LeverPositionEnum.UPPERLEFT, l1=LeverPositionEnum.UPPER, l2=LEFT, l3=LeverPositionEnum.LOWERLEFT
    'trial9'   : [LeverConfig(LeverPositionEnum.UPPERRIGHT,   LeverRoleEnum.inactive, None),
                  LeverConfig(LeverPositionEnum.UPPER,        LeverRoleEnum.l1,       None),
                  LeverConfig(LeverPositionEnum.UPPERLEFT,    LeverRoleEnum.l0,       None),
                  LeverConfig(LeverPositionEnum.LEFT,         LeverRoleEnum.l2,       None),
                  LeverConfig(LeverPositionEnum.LOWERLEFT,    LeverRoleEnum.l3,       None),
                  LeverConfig(LeverPositionEnum.LOWER,        LeverRoleEnum.inactive, None),
                  LeverConfig(LeverPositionEnum.LOWERRIGHT,   LeverRoleEnum.inactive, None)],
    # Trial 10. l0=LeverPositionEnum.LOWERLEFT, l1=LeverPositionEnum.UPPERLEFT, l2=LEFT, l3=LeverPositionEnum.LOWER
    'trial10'  : [LeverConfig(LeverPositionEnum.UPPERRIGHT,   LeverRoleEnum.inactive, None),
                  LeverConfig(LeverPositionEnum.UPPER,        LeverRoleEnum.inactive, None),
                  LeverConfig(LeverPositionEnum.UPPERLEFT,    LeverRoleEnum.l1,       None),
                  LeverConfig(LeverPositionEnum.LEFT,         LeverRoleEnum.l2,       None),
                  LeverConfig(LeverPositionEnum.LOWERLEFT,    LeverRoleEnum.l0,       None),
                  LeverConfig(LeverPositionEnum.LOWER,        LeverRoleEnum.l3,       None),
                  LeverConfig(LeverPositionEnum.LOWERRIGHT,   LeverRoleEnum.inactive, None)],
    # Trial 11. l0=LeverPositionEnum.LOWERRIGHT, l1=LEFT, l2=LeverPositionEnum.LOWERLEFT, l3=LeverPositionEnum.LOWER
    'trial11'  : [LeverConfig(LeverPositionEnum.UPPERRIGHT,   LeverRoleEnum.inactive, None),
                  LeverConfig(LeverPositionEnum.UPPER,        LeverRoleEnum.inactive, None),
                  LeverConfig(LeverPositionEnum.UPPERLEFT,    LeverRoleEnum.inactive, None),
                  LeverConfig(LeverPositionEnum.LEFT,         LeverRoleEnum.l1,       None),
                  LeverConfig(LeverPositionEnum.LOWERLEFT,    LeverRoleEnum.l2,       None),
                  LeverConfig(LeverPositionEnum.LOWER,        LeverRoleEnum.l3,       None),
                  LeverConfig(LeverPositionEnum.LOWERRIGHT,   LeverRoleEnum.l0,       None)],

    # multi-lock. l0=LeverPositionEnum.UPPER, l1=LeverPositionEnum.LOWER, l2=LEFT,
    'multi-lock': [LeverConfig(LeverPositionEnum.UPPERRIGHT,  LeverRoleEnum.inactive, None),
                  LeverConfig(LeverPositionEnum.UPPER,        LeverRoleEnum.l2,       None),
                  LeverConfig(LeverPositionEnum.UPPERLEFT,    LeverRoleEnum.inactive, None),
                  LeverConfig(LeverPositionEnum.LEFT,         LeverRoleEnum.l0,       {'lower_lim': 0.0, 'upper_lim': 2.0}),
                  LeverConfig(LeverPositionEnum.LOWERLEFT,    LeverRoleEnum.inactive, None),
                  LeverConfig(LeverPositionEnum.LOWER,        LeverRoleEnum.l1,       None),
                  LeverConfig(LeverPositionEnum.LOWERRIGHT,   LeverRoleEnum.inactive, None)],
    # full.
    'full'     : [LeverConfig(LeverPositionEnum.UPPERRIGHT,   LeverRoleEnum.l0,       None),
                  LeverConfig(LeverPositionEnum.UPPER,        LeverRoleEnum.l1,       None),
                  LeverConfig(LeverPositionEnum.UPPERLEFT,    LeverRoleEnum.l2,       None),
                  LeverConfig(LeverPositionEnum.LEFT,         LeverRoleEnum.l3,       None),
                  LeverConfig(LeverPositionEnum.LOWERLEFT,    LeverRoleEnum.l4,       None),
                  LeverConfig(LeverPositionEnum.LOWER,        LeverRoleEnum.l5,       None),
                  LeverConfig(LeverPositionEnum.LOWERRIGHT,   LeverRoleEnum.l6,       None)],
}


def select_trial(trial):
    return trial, LEVER_CONFIGS[trial]


def get_trial(name, completed_trials=None):
    """
    Apply specific rules for selecting random trials.
    Namely, For CE4 & CC4, only selects from trials 7-11, otherwise only selects from trials 1-6.

    :param name: Name of trial
    :param completed_trials:
    :return: trial and configs
    """
    # select a random trial and add it to the scenario
    if name != 'CE4' and name != 'CC4':
        # trials 1-6 have 3 levers for CC3/CE3
        trial, configs = select_random_trial(completed_trials, THREE_LEVER_TRIALS)
    else:
        # trials 7-11 have 4 levers for CC4/CE4
        trial, configs = select_random_trial(completed_trials, FOUR_LEVER_TRIALS)

    return trial, configs


def select_random_trial(completed_trials, possible_trials):
    '''
    sets a new random trial
    :param completed_trials: list of trials already selected
    :param possible_trials: list of all trials possible
    :return:
    '''
    if len(completed_trials) == len(possible_trials):
        return None, None

    incomplete_trials = np.setdiff1d(possible_trials, completed_trials)
    rand_trial_idx = np.random.randint(0, len(incomplete_trials))
    trial = incomplete_trials[rand_trial_idx]

    return select_trial(trial)

