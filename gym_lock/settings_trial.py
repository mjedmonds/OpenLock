import numpy as np

from gym_lock.common import TwoDConfig, LeverConfig, LeverRole

NUM_LEVERS = 7

UPPER = TwoDConfig(0, 15, 0)
LEFT = TwoDConfig(-15, 0, np.pi / 2)
LOWER = TwoDConfig(0, -15, -np.pi)
UPPERLEFT = TwoDConfig(-11, 11, np.pi/4)
UPPERRIGHT = TwoDConfig(11, 11, -np.pi/4)
LOWERLEFT = TwoDConfig(-11, -11, 3*np.pi / 4)
LOWERRIGHT = TwoDConfig(11, -11, 5*np.pi/4)

ATTEMPT_LIMIT = 30
ACTION_LIMIT = 3
DATA_DIR = '../OpenLockResults/subjects'


PARAMS = {
    'CE3-CE4': {
        'data_dir': DATA_DIR,
        'num_train_trials': 6,
        'train_scenario_name': 'CE3',
        'train_attempt_limit': ATTEMPT_LIMIT,
        'train_action_limit': ACTION_LIMIT,
        'num_test_trials': 1,
        'test_scenario_name': 'CE4',
        'test_attempt_limit': ATTEMPT_LIMIT,
        'test_action_limit': ACTION_LIMIT
    },
    'CE3-CC4': {
        'data_dir': DATA_DIR,
        'num_train_trials': 6,
        'train_scenario_name': 'CE3',
        'train_attempt_limit': ATTEMPT_LIMIT,
        'train_action_limit': ACTION_LIMIT,
        'num_test_trials': 1,
        'test_scenario_name': 'CC4',
        'test_attempt_limit': ATTEMPT_LIMIT,
        'test_action_limit': ACTION_LIMIT
    },
    'CC3-CE4': {
        'data_dir': DATA_DIR,
        'num_train_trials': 6,
        'train_scenario_name': 'CC3',
        'train_attempt_limit': ATTEMPT_LIMIT,
        'train_action_limit': ACTION_LIMIT,
        'num_test_trials': 1,
        'test_scenario_name': 'CE4',
        'test_attempt_limit': ATTEMPT_LIMIT,
        'test_action_limit': ACTION_LIMIT
    },
    'CC3-CC4': {
        'data_dir': DATA_DIR,
        'num_train_trials': 6,
        'train_scenario_name': 'CC3',
        'train_attempt_limit': ATTEMPT_LIMIT,
        'train_action_limit': ACTION_LIMIT,
        'num_test_trials': 1,
        'test_scenario_name': 'CC4',
        'test_attempt_limit': ATTEMPT_LIMIT,
        'test_action_limit': ACTION_LIMIT
    },
    'CC4': {
        'data_dir': DATA_DIR,
        'num_train_trials': 5,
        'train_scenario_name': 'CC4',
        'train_attempt_limit': ATTEMPT_LIMIT,
        'train_action_limit': ACTION_LIMIT,
        'num_test_trials': 0,
        'test_scenario_name': None,
        'test_attempt_limit': ATTEMPT_LIMIT,
        'test_action_limit': ACTION_LIMIT
    },
    'CE4': {
        'data_dir': DATA_DIR,
        'num_train_trials': 5,
        'train_scenario_name': 'CE4',
        'train_attempt_limit': ATTEMPT_LIMIT,
        'train_action_limit': ACTION_LIMIT,
        'num_test_trials': 0,
        'test_scenario_name': None,
        'test_attempt_limit': ATTEMPT_LIMIT,
        'test_action_limit': ACTION_LIMIT
    },
    'testing': {
        'data_dir': DATA_DIR,
        'num_train_trials': 1,
        'train_scenario_name': 'CC3',
        'train_attempt_limit': ATTEMPT_LIMIT,
        'train_action_limit': ACTION_LIMIT,
        'test_scenario_name': None,
        'test_attempt_limit': ATTEMPT_LIMIT,
        'test_action_limit': ACTION_LIMIT
    }
}

IDX_TO_PARAMS = [
    'CE3-CE4',
    'CE3-CC4',
    'CC3-CE4',
    'CC3-CC4',
    'CE4',
    'CC4'
]

CONFIG_TO_IDX = {
    UPPERRIGHT: 0,
    UPPER: 1,
    UPPERLEFT: 2,
    LEFT: 3,
    LOWERLEFT: 4,
    LOWER: 5,
    LOWERRIGHT: 6
}

LEVER_CONFIGS = {
    # Trial 1. l0=UPPERLEFT, l1=LOWERLEFT, l2=UPPERRIGHT,
    'trial1'   : [LeverConfig(UPPERRIGHT,   LeverRole.l0,       None),
                  LeverConfig(UPPER,        LeverRole.inactive, None),
                  LeverConfig(UPPERLEFT,    LeverRole.l2,       None),
                  LeverConfig(LEFT,         LeverRole.inactive, None),
                  LeverConfig(LOWERLEFT,    LeverRole.l1,       None),
                  LeverConfig(LOWER,        LeverRole.inactive, None),
                  LeverConfig(LOWERRIGHT,   LeverRole.inactive, None)],
    # Trial 2. l0=UPPER, l1=LOWER, l2=LEFT,
    'trial2'   : [LeverConfig(UPPERRIGHT,   LeverRole.inactive, None),
                  LeverConfig(UPPER,        LeverRole.l2,       None),
                  LeverConfig(UPPERLEFT,    LeverRole.inactive, None),
                  LeverConfig(LEFT,         LeverRole.l0,       None),
                  LeverConfig(LOWERLEFT,    LeverRole.inactive, None),
                  LeverConfig(LOWER,        LeverRole.l1,       None),
                  LeverConfig(LOWERRIGHT,   LeverRole.inactive, None)],
    # Trial 3. l0=UPPERLEFT , l1=LOWERLEFT, l2=LOWERRIGHT,
    'trial3'   : [LeverConfig(UPPERRIGHT,   LeverRole.inactive, None),
                  LeverConfig(UPPER,        LeverRole.inactive, None),
                  LeverConfig(UPPERLEFT,    LeverRole.l1,       None),
                  LeverConfig(LEFT,         LeverRole.inactive, None),
                  LeverConfig(LOWERLEFT,    LeverRole.l2,       None),
                  LeverConfig(LOWER,        LeverRole.inactive, None),
                  LeverConfig(LOWERRIGHT,   LeverRole.l0,       None)],
    # Trial 4. l0=UPPER, l1=UPPERLEFT, l2=UPPERRIGHT,
    'trial4'   : [LeverConfig(UPPERRIGHT,   LeverRole.l0,       None),
                  LeverConfig(UPPER,        LeverRole.l2,       None),
                  LeverConfig(UPPERLEFT,    LeverRole.l1,       None),
                  LeverConfig(LEFT,         LeverRole.inactive, None),
                  LeverConfig(LOWERLEFT,    LeverRole.inactive, None),
                  LeverConfig(LOWER,        LeverRole.inactive, None),
                  LeverConfig(LOWERRIGHT,   LeverRole.inactive, None)],
    # Trial 5. l0=UPPERLEFT, l1=LOWERLEFT, l2=LEFT,
    'trial5'   : [LeverConfig(UPPERRIGHT,   LeverRole.inactive, None),
                  LeverConfig(UPPER,        LeverRole.inactive, None),
                  LeverConfig(UPPERLEFT,    LeverRole.l2,       None),
                  LeverConfig(LEFT,         LeverRole.l0,       None),
                  LeverConfig(LOWERLEFT,    LeverRole.l1,       None),
                  LeverConfig(LOWER,        LeverRole.inactive, None),
                  LeverConfig(LOWERRIGHT,   LeverRole.inactive, None)],
    # Trial 6. l0=LOWERLEFT, l1=LOWER, l2=LOWERRIGHT,
    'trial6'   : [LeverConfig(UPPERRIGHT,   LeverRole.inactive, None),
                  LeverConfig(UPPER,        LeverRole.inactive, None),
                  LeverConfig(UPPERLEFT,    LeverRole.inactive, None),
                  LeverConfig(LEFT,         LeverRole.inactive, None),
                  LeverConfig(LOWERLEFT,    LeverRole.l2,       None),
                  LeverConfig(LOWER,        LeverRole.l1,       None),
                  LeverConfig(LOWERRIGHT,   LeverRole.l0,       None)],
    # Trial 7. l0=LOWERLEFT, l1=UPPERRIGHT, l2=LOWERRIGHT, l3=UPPERLEFT
    'trial7'   : [LeverConfig(UPPERRIGHT,   LeverRole.l1,       None),
                  LeverConfig(UPPER,        LeverRole.inactive, None),
                  LeverConfig(UPPERLEFT,    LeverRole.l3,       None),
                  LeverConfig(LEFT,         LeverRole.inactive, None),
                  LeverConfig(LOWERLEFT,    LeverRole.l0,       None),
                  LeverConfig(LOWER,        LeverRole.inactive, None),
                  LeverConfig(LOWERRIGHT,   LeverRole.l2,       None)],
    # Trial 8. l0=UPPERRIGHT, l1=UPPER, l2=UPPERLEFT, l3=LEFT
    'trial8'   : [LeverConfig(UPPERRIGHT,   LeverRole.l0,       None),
                  LeverConfig(UPPER,        LeverRole.l1,       None),
                  LeverConfig(UPPERLEFT,    LeverRole.l2,       None),
                  LeverConfig(LEFT,         LeverRole.l3,       None),
                  LeverConfig(LOWERLEFT,    LeverRole.inactive, None),
                  LeverConfig(LOWER,        LeverRole.inactive, None),
                  LeverConfig(LOWERRIGHT,   LeverRole.inactive, None)],
    # Trial 9. l0=UPPERLEFT, l1=UPPER, l2=LEFT, l3=LOWERLEFT
    'trial9'   : [LeverConfig(UPPERRIGHT,   LeverRole.inactive, None),
                  LeverConfig(UPPER,        LeverRole.l1,       None),
                  LeverConfig(UPPERLEFT,    LeverRole.l0,       None),
                  LeverConfig(LEFT,         LeverRole.l2,       None),
                  LeverConfig(LOWERLEFT,    LeverRole.l3,       None),
                  LeverConfig(LOWER,        LeverRole.inactive, None),
                  LeverConfig(LOWERRIGHT,   LeverRole.inactive, None)],
    # Trial 10. l0=LOWERLEFT, l1=UPPERLEFT, l2=LEFT, l3=LOWER
    'trial10'  : [LeverConfig(UPPERRIGHT,   LeverRole.inactive, None),
                  LeverConfig(UPPER,        LeverRole.inactive, None),
                  LeverConfig(UPPERLEFT,    LeverRole.l1,       None),
                  LeverConfig(LEFT,         LeverRole.l2,       None),
                  LeverConfig(LOWERLEFT,    LeverRole.l0,       None),
                  LeverConfig(LOWER,        LeverRole.l3,       None),
                  LeverConfig(LOWERRIGHT,   LeverRole.inactive, None)],
    # Trial 11. l0=LOWERRIGHT, l1=LEFT, l2=LOWERLEFT, l3=LOWER
    'trial11'  : [LeverConfig(UPPERRIGHT,   LeverRole.inactive, None),
                  LeverConfig(UPPER,        LeverRole.inactive, None),
                  LeverConfig(UPPERLEFT,    LeverRole.inactive, None),
                  LeverConfig(LEFT,         LeverRole.l1,       None),
                  LeverConfig(LOWERLEFT,    LeverRole.l2,       None),
                  LeverConfig(LOWER,        LeverRole.l3,       None),
                  LeverConfig(LOWERRIGHT,   LeverRole.l0,       None)],

    # multi-lock. l0=UPPER, l1=LOWER, l2=LEFT,
    'multi-lock': [LeverConfig(UPPERRIGHT,  LeverRole.inactive, None),
                  LeverConfig(UPPER,        LeverRole.l2,       None),
                  LeverConfig(UPPERLEFT,    LeverRole.inactive, None),
                  LeverConfig(LEFT,         LeverRole.l0,       {'lower_lim': 0.0, 'upper_lim': 2.0}),
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
    trial_list = []
    trial_base = 'trial'
    rand = np.random.randint(min_idx, max_idx+1)
    trial = trial_base + str(rand)
    trial_list.append(trial)
    while trial in completed_trials:
        rand = np.random.randint(min_idx, max_idx+1)
        trial = trial_base + str(rand)
        trial_list.append(trial)
        # all have been tried
        if len(trial_list) == max_idx - min_idx:
            return None, None

    return select_trial(trial)

