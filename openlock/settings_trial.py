import numpy as np
import re

from openlock.common import (
    TwoDConfig,
    LeverConfig,
    LeverRoleEnum,
    ObjectPositionEnum,
    LOCK_REGEX_STR,
)


NUM_LEVERS_IN_HUMAN_DATA = 7

ATTEMPT_LIMIT = 30
ACTION_LIMIT = 3

THREE_LEVER_TRIALS = ["trial1", "trial2", "trial3", "trial4", "trial5", "trial6"]
FOUR_LEVER_TRIALS = ["trial7", "trial8", "trial9", "trial10", "trial11"]
SIMPLIFIED_THREE_LEVER_TRIALS = [
    "simplified_trial1",
    "simplified_trial2",
    "simplified_trial3",
    "simplified_trial4",
]
TWO_STEP_TESTING_TRIALS = ["TwoStepTesting1"]

PARAMS = {
    "CE3-CE4": {
        "train_num_trials": 6,
        "train_scenario_name": "CE3",
        "train_attempt_limit": ATTEMPT_LIMIT,
        "train_action_limit": ACTION_LIMIT,
        "test_num_trials": 1,
        "test_scenario_name": "CE4",
        "test_attempt_limit": ATTEMPT_LIMIT,
        "test_action_limit": ACTION_LIMIT,
    },
    "CE3-CC4": {
        "train_num_trials": 6,
        "train_scenario_name": "CE3",
        "train_attempt_limit": ATTEMPT_LIMIT,
        "train_action_limit": ACTION_LIMIT,
        "test_num_trials": 1,
        "test_scenario_name": "CC4",
        "test_attempt_limit": ATTEMPT_LIMIT,
        "test_action_limit": ACTION_LIMIT,
    },
    "CC3-CE4": {
        "train_num_trials": 6,
        "train_scenario_name": "CC3",
        "train_attempt_limit": ATTEMPT_LIMIT,
        "train_action_limit": ACTION_LIMIT,
        "test_num_trials": 1,
        "test_scenario_name": "CE4",
        "test_attempt_limit": ATTEMPT_LIMIT,
        "test_action_limit": ACTION_LIMIT,
    },
    "CC3-CC4": {
        "train_num_trials": 6,
        "train_scenario_name": "CC3",
        "train_attempt_limit": ATTEMPT_LIMIT,
        "train_action_limit": ACTION_LIMIT,
        "test_num_trials": 1,
        "test_scenario_name": "CC4",
        "test_attempt_limit": ATTEMPT_LIMIT,
        "test_action_limit": ACTION_LIMIT,
    },
    "CC4": {
        "train_num_trials": 5,
        "train_scenario_name": "CC4",
        "train_attempt_limit": ATTEMPT_LIMIT,
        "train_action_limit": ACTION_LIMIT,
        "test_num_trials": 0,
        "test_scenario_name": None,
        "test_attempt_limit": ATTEMPT_LIMIT,
        "test_action_limit": ACTION_LIMIT,
    },
    "CE4": {
        "train_num_trials": 5,
        "train_scenario_name": "CE4",
        "train_attempt_limit": ATTEMPT_LIMIT,
        "train_action_limit": ACTION_LIMIT,
        "test_num_trials": 0,
        "test_scenario_name": None,
        "test_attempt_limit": ATTEMPT_LIMIT,
        "test_action_limit": ACTION_LIMIT,
    },
    "testing": {
        "train_num_trials": 1,
        "train_scenario_name": "CC3",
        "train_attempt_limit": ATTEMPT_LIMIT,
        "train_action_limit": ACTION_LIMIT,
        "test_scenario_name": None,
        "test_attempt_limit": ATTEMPT_LIMIT,
        "test_action_limit": ACTION_LIMIT,
    },
}

# maps arbitrary indices to parameter settings strings
IDX_TO_PARAMS = ["CE3-CE4", "CE3-CC4", "CC3-CE4", "CC3-CC4", "CE4", "CC4"]

# # mapping from 2dconfigs to position indices
# CONFIG_TO_IDX = {
#     ObjectPositionEnum.UPPERRIGHT.config: 0,
#     ObjectPositionEnum.UPPER.config: 1,
#     ObjectPositionEnum.UPPERLEFT.config: 2,
#     ObjectPositionEnum.LEFT.config: 3,
#     ObjectPositionEnum.LOWERLEFT.config: 4,
#     ObjectPositionEnum.LOWER.config: 5,
#     ObjectPositionEnum.LOWERRIGHT.config: 6,
#     ObjectPositionEnum.DOOR.config: 7,
#     # ObjectPositionEnum.DOOR_LOCK.config:  8,
# }
#
# # mapping from position indices to position names
IDX_TO_POSITION = {
    0: "UPPERRIGHT",
    1: "UPPER",
    2: "UPPERLEFT",
    3: "LEFT",
    4: "LOWERLEFT",
    5: "LOWER",
    6: "LOWERRIGHT",
    7: "door",
    # 8: 'door_lock'
}
#
# # mapping from position names to position indices
# POSITION_TO_IDX = {
#     "UPPERRIGHT": 0,
#     "UPPER": 1,
#     "UPPERLEFT": 2,
#     "LEFT": 3,
#     "LOWERLEFT": 4,
#     "LOWER": 5,
#     "LOWERRIGHT": 6,
#     "door": 7,
#     # 'door_lock':    8,
# }

LEVER_CONFIGS = {
    # Trial 1. l0=LeverPositionEnum.UPPERLEFT, l1=LeverPositionEnum.LOWERLEFT, l2=LeverPositionEnum.UPPERRIGHT,
    "trial1": [
        LeverConfig(ObjectPositionEnum.UPPERRIGHT, LeverRoleEnum.l0, None),
        LeverConfig(ObjectPositionEnum.UPPER, LeverRoleEnum.inactive, None),
        LeverConfig(ObjectPositionEnum.UPPERLEFT, LeverRoleEnum.l2, None),
        LeverConfig(ObjectPositionEnum.LEFT, LeverRoleEnum.inactive, None),
        LeverConfig(ObjectPositionEnum.LOWERLEFT, LeverRoleEnum.l1, None),
        LeverConfig(ObjectPositionEnum.LOWER, LeverRoleEnum.inactive, None),
        LeverConfig(ObjectPositionEnum.LOWERRIGHT, LeverRoleEnum.inactive, None),
    ],
    # Trial 2. l0=LeverPositionEnum.UPPER, l1=LeverPositionEnum.LOWER, l2=LEFT,
    "trial2": [
        LeverConfig(ObjectPositionEnum.UPPERRIGHT, LeverRoleEnum.inactive, None),
        LeverConfig(ObjectPositionEnum.UPPER, LeverRoleEnum.l2, None),
        LeverConfig(ObjectPositionEnum.UPPERLEFT, LeverRoleEnum.inactive, None),
        LeverConfig(ObjectPositionEnum.LEFT, LeverRoleEnum.l0, None),
        LeverConfig(ObjectPositionEnum.LOWERLEFT, LeverRoleEnum.inactive, None),
        LeverConfig(ObjectPositionEnum.LOWER, LeverRoleEnum.l1, None),
        LeverConfig(ObjectPositionEnum.LOWERRIGHT, LeverRoleEnum.inactive, None),
    ],
    # Trial 3. l0=LeverPositionEnum.UPPERLEFT , l1=LeverPositionEnum.LOWERLEFT, l2=LeverPositionEnum.LOWERRIGHT,
    "trial3": [
        LeverConfig(ObjectPositionEnum.UPPERRIGHT, LeverRoleEnum.inactive, None),
        LeverConfig(ObjectPositionEnum.UPPER, LeverRoleEnum.inactive, None),
        LeverConfig(ObjectPositionEnum.UPPERLEFT, LeverRoleEnum.l1, None),
        LeverConfig(ObjectPositionEnum.LEFT, LeverRoleEnum.inactive, None),
        LeverConfig(ObjectPositionEnum.LOWERLEFT, LeverRoleEnum.l2, None),
        LeverConfig(ObjectPositionEnum.LOWER, LeverRoleEnum.inactive, None),
        LeverConfig(ObjectPositionEnum.LOWERRIGHT, LeverRoleEnum.l0, None),
    ],
    # Trial 4. l0=LeverPositionEnum.UPPER, l1=LeverPositionEnum.UPPERLEFT, l2=LeverPositionEnum.UPPERRIGHT,
    "trial4": [
        LeverConfig(ObjectPositionEnum.UPPERRIGHT, LeverRoleEnum.l0, None),
        LeverConfig(ObjectPositionEnum.UPPER, LeverRoleEnum.l2, None),
        LeverConfig(ObjectPositionEnum.UPPERLEFT, LeverRoleEnum.l1, None),
        LeverConfig(ObjectPositionEnum.LEFT, LeverRoleEnum.inactive, None),
        LeverConfig(ObjectPositionEnum.LOWERLEFT, LeverRoleEnum.inactive, None),
        LeverConfig(ObjectPositionEnum.LOWER, LeverRoleEnum.inactive, None),
        LeverConfig(ObjectPositionEnum.LOWERRIGHT, LeverRoleEnum.inactive, None),
    ],
    # Trial 5. l0=LeverPositionEnum.UPPERLEFT, l1=LeverPositionEnum.LOWERLEFT, l2=LEFT,
    "trial5": [
        LeverConfig(ObjectPositionEnum.UPPERRIGHT, LeverRoleEnum.inactive, None),
        LeverConfig(ObjectPositionEnum.UPPER, LeverRoleEnum.inactive, None),
        LeverConfig(ObjectPositionEnum.UPPERLEFT, LeverRoleEnum.l2, None),
        LeverConfig(ObjectPositionEnum.LEFT, LeverRoleEnum.l0, None),
        LeverConfig(ObjectPositionEnum.LOWERLEFT, LeverRoleEnum.l1, None),
        LeverConfig(ObjectPositionEnum.LOWER, LeverRoleEnum.inactive, None),
        LeverConfig(ObjectPositionEnum.LOWERRIGHT, LeverRoleEnum.inactive, None),
    ],
    # Trial 6. l0=LeverPositionEnum.LOWERLEFT, l1=LeverPositionEnum.LOWER, l2=LeverPositionEnum.LOWERRIGHT,
    "trial6": [
        LeverConfig(ObjectPositionEnum.UPPERRIGHT, LeverRoleEnum.inactive, None),
        LeverConfig(ObjectPositionEnum.UPPER, LeverRoleEnum.inactive, None),
        LeverConfig(ObjectPositionEnum.UPPERLEFT, LeverRoleEnum.inactive, None),
        LeverConfig(ObjectPositionEnum.LEFT, LeverRoleEnum.inactive, None),
        LeverConfig(ObjectPositionEnum.LOWERLEFT, LeverRoleEnum.l2, None),
        LeverConfig(ObjectPositionEnum.LOWER, LeverRoleEnum.l1, None),
        LeverConfig(ObjectPositionEnum.LOWERRIGHT, LeverRoleEnum.l0, None),
    ],
    # Trial 7. l0=LeverPositionEnum.LOWERLEFT, l1=LeverPositionEnum.UPPERRIGHT, l2=LeverPositionEnum.LOWERRIGHT, l3=LeverPositionEnum.UPPERLEFT
    "trial7": [
        LeverConfig(ObjectPositionEnum.UPPERRIGHT, LeverRoleEnum.l1, None),
        LeverConfig(ObjectPositionEnum.UPPER, LeverRoleEnum.inactive, None),
        LeverConfig(ObjectPositionEnum.UPPERLEFT, LeverRoleEnum.l3, None),
        LeverConfig(ObjectPositionEnum.LEFT, LeverRoleEnum.inactive, None),
        LeverConfig(ObjectPositionEnum.LOWERLEFT, LeverRoleEnum.l0, None),
        LeverConfig(ObjectPositionEnum.LOWER, LeverRoleEnum.inactive, None),
        LeverConfig(ObjectPositionEnum.LOWERRIGHT, LeverRoleEnum.l2, None),
    ],
    # Trial 8. l0=LeverPositionEnum.UPPERRIGHT, l1=LeverPositionEnum.UPPER, l2=LeverPositionEnum.UPPERLEFT, l3=LEFT
    "trial8": [
        LeverConfig(ObjectPositionEnum.UPPERRIGHT, LeverRoleEnum.l0, None),
        LeverConfig(ObjectPositionEnum.UPPER, LeverRoleEnum.l1, None),
        LeverConfig(ObjectPositionEnum.UPPERLEFT, LeverRoleEnum.l2, None),
        LeverConfig(ObjectPositionEnum.LEFT, LeverRoleEnum.l3, None),
        LeverConfig(ObjectPositionEnum.LOWERLEFT, LeverRoleEnum.inactive, None),
        LeverConfig(ObjectPositionEnum.LOWER, LeverRoleEnum.inactive, None),
        LeverConfig(ObjectPositionEnum.LOWERRIGHT, LeverRoleEnum.inactive, None),
    ],
    # Trial 9. l0=LeverPositionEnum.UPPERLEFT, l1=LeverPositionEnum.UPPER, l2=LEFT, l3=LeverPositionEnum.LOWERLEFT
    "trial9": [
        LeverConfig(ObjectPositionEnum.UPPERRIGHT, LeverRoleEnum.inactive, None),
        LeverConfig(ObjectPositionEnum.UPPER, LeverRoleEnum.l1, None),
        LeverConfig(ObjectPositionEnum.UPPERLEFT, LeverRoleEnum.l0, None),
        LeverConfig(ObjectPositionEnum.LEFT, LeverRoleEnum.l2, None),
        LeverConfig(ObjectPositionEnum.LOWERLEFT, LeverRoleEnum.l3, None),
        LeverConfig(ObjectPositionEnum.LOWER, LeverRoleEnum.inactive, None),
        LeverConfig(ObjectPositionEnum.LOWERRIGHT, LeverRoleEnum.inactive, None),
    ],
    # Trial 10. l0=LeverPositionEnum.LOWERLEFT, l1=LeverPositionEnum.UPPERLEFT, l2=LEFT, l3=LeverPositionEnum.LOWER
    "trial10": [
        LeverConfig(ObjectPositionEnum.UPPERRIGHT, LeverRoleEnum.inactive, None),
        LeverConfig(ObjectPositionEnum.UPPER, LeverRoleEnum.inactive, None),
        LeverConfig(ObjectPositionEnum.UPPERLEFT, LeverRoleEnum.l1, None),
        LeverConfig(ObjectPositionEnum.LEFT, LeverRoleEnum.l2, None),
        LeverConfig(ObjectPositionEnum.LOWERLEFT, LeverRoleEnum.l0, None),
        LeverConfig(ObjectPositionEnum.LOWER, LeverRoleEnum.l3, None),
        LeverConfig(ObjectPositionEnum.LOWERRIGHT, LeverRoleEnum.inactive, None),
    ],
    # Trial 11. l0=LeverPositionEnum.LOWERRIGHT, l1=LEFT, l2=LeverPositionEnum.LOWERLEFT, l3=LeverPositionEnum.LOWER
    "trial11": [
        LeverConfig(ObjectPositionEnum.UPPERRIGHT, LeverRoleEnum.inactive, None),
        LeverConfig(ObjectPositionEnum.UPPER, LeverRoleEnum.inactive, None),
        LeverConfig(ObjectPositionEnum.UPPERLEFT, LeverRoleEnum.inactive, None),
        LeverConfig(ObjectPositionEnum.LEFT, LeverRoleEnum.l1, None),
        LeverConfig(ObjectPositionEnum.LOWERLEFT, LeverRoleEnum.l2, None),
        LeverConfig(ObjectPositionEnum.LOWER, LeverRoleEnum.l3, None),
        LeverConfig(ObjectPositionEnum.LOWERRIGHT, LeverRoleEnum.l0, None),
    ],
    # Two Step training environment
    "TwoStepTesting1": [
        LeverConfig(ObjectPositionEnum.LEFT, LeverRoleEnum.l1, None),
        LeverConfig(ObjectPositionEnum.UPPER, LeverRoleEnum.l0, None),
    ],
    # multi-lock. l0=LeverPositionEnum.UPPER, l1=LeverPositionEnum.LOWER, l2=LEFT,
    "multi-lock": [
        LeverConfig(ObjectPositionEnum.UPPERRIGHT, LeverRoleEnum.inactive, None),
        LeverConfig(ObjectPositionEnum.UPPER, LeverRoleEnum.l2, None),
        LeverConfig(ObjectPositionEnum.UPPERLEFT, LeverRoleEnum.inactive, None),
        LeverConfig(
            ObjectPositionEnum.LEFT,
            LeverRoleEnum.l0,
            {"lower_lim": 0.0, "upper_lim": 2.0},
        ),
        LeverConfig(ObjectPositionEnum.LOWERLEFT, LeverRoleEnum.inactive, None),
        LeverConfig(ObjectPositionEnum.LOWER, LeverRoleEnum.l1, None),
        LeverConfig(ObjectPositionEnum.LOWERRIGHT, LeverRoleEnum.inactive, None),
    ],
    # full.
    "full": [
        LeverConfig(ObjectPositionEnum.UPPERRIGHT, LeverRoleEnum.l0, None),
        LeverConfig(ObjectPositionEnum.UPPER, LeverRoleEnum.l1, None),
        LeverConfig(ObjectPositionEnum.UPPERLEFT, LeverRoleEnum.l2, None),
        LeverConfig(ObjectPositionEnum.LEFT, LeverRoleEnum.l3, None),
        LeverConfig(ObjectPositionEnum.LOWERLEFT, LeverRoleEnum.l4, None),
        LeverConfig(ObjectPositionEnum.LOWER, LeverRoleEnum.l5, None),
        LeverConfig(ObjectPositionEnum.LOWERRIGHT, LeverRoleEnum.l6, None),
    ],
    # add in simplified trials. These trials always use UPPERRIGHT, UPPERLEFT, LOWERLEFT, LOWERRIGHT. 4 levers total.
    "simplified_trial1": [
        LeverConfig(ObjectPositionEnum.UPPERRIGHT, LeverRoleEnum.l0, None),
        LeverConfig(ObjectPositionEnum.UPPERLEFT, LeverRoleEnum.l2, None),
        LeverConfig(ObjectPositionEnum.LOWERLEFT, LeverRoleEnum.l1, None),
        LeverConfig(ObjectPositionEnum.LOWERRIGHT, LeverRoleEnum.inactive, None),
    ],
    "simplified_trial2": [
        LeverConfig(ObjectPositionEnum.UPPERLEFT, LeverRoleEnum.l2, None),
        LeverConfig(ObjectPositionEnum.UPPERRIGHT, LeverRoleEnum.l0, None),
        LeverConfig(ObjectPositionEnum.LOWERLEFT, LeverRoleEnum.inactive, None),
        LeverConfig(ObjectPositionEnum.LOWERRIGHT, LeverRoleEnum.l1, None),
    ],
    "simplified_trial3": [
        LeverConfig(ObjectPositionEnum.LOWERLEFT, LeverRoleEnum.l0, None),
        LeverConfig(ObjectPositionEnum.UPPERRIGHT, LeverRoleEnum.inactive, None),
        LeverConfig(ObjectPositionEnum.LOWERRIGHT, LeverRoleEnum.l1, None),
        LeverConfig(ObjectPositionEnum.UPPERLEFT, LeverRoleEnum.l2, None),
    ],
    "simplified_trial4": [
        LeverConfig(ObjectPositionEnum.UPPERLEFT, LeverRoleEnum.inactive, None),
        LeverConfig(ObjectPositionEnum.LOWERLEFT, LeverRoleEnum.l0, None),
        LeverConfig(ObjectPositionEnum.UPPERRIGHT, LeverRoleEnum.l2, None),
        LeverConfig(ObjectPositionEnum.LOWERRIGHT, LeverRoleEnum.l1, None),
    ],
}


def generate_attributes_by_trial():
    attributes_by_trial = dict()
    for trial, lever_configs in LEVER_CONFIGS.items():
        attributes_by_position = dict()
        for lever_config in lever_configs:
            position, role, opt = lever_config
            lock_regex = re.compile(LOCK_REGEX_STR)
            # todo extend this to handle multiple attributes
            if re.match(lock_regex, role):
                color = "GREY"
            else:
                color = "WHITE"
            attributes_by_position[position.name] = (position.name, color)
        attributes_by_position["door"] = ("door", "GREY")
        attributes_by_position["door_lock"] = ("door_lock", "GREY")
        attributes_by_trial[trial] = attributes_by_position
    return attributes_by_trial


def select_trial(trial):
    return trial, LEVER_CONFIGS[trial]


def get_possible_trials(name):
    if name == "CE3" or name == "CC3":
        return THREE_LEVER_TRIALS
    if name == "CE3_simplified" or name == "CC3_simplified":
        return SIMPLIFIED_THREE_LEVER_TRIALS
    if name == "CE4" or name == "CC4":
        return FOUR_LEVER_TRIALS
    else:
        raise ValueError("Unknown trial type")


def get_trial(scenario_name, completed_trials=None):
    """
    Apply specific rules for selecting random trials.
    Namely, For CE4 & CC4, only selects from trials 7-11, otherwise only selects from trials 1-6.

    :param scenario_name: Name of trial
    :param completed_trials:
    :return: trial and configs
    """
    if completed_trials is None:
        completed_trials = []
    # select a random trial and add it to the scenario
    if scenario_name == "CE3" or scenario_name == "CC3":
        # trials 1-6 have 3 levers for CC3/CE3
        trial, configs = select_random_trial(completed_trials, THREE_LEVER_TRIALS)
    elif scenario_name == "CE4" or scenario_name == "CC4":
        # trials 7-11 have 4 levers for CC4/CE4
        trial, configs = select_random_trial(completed_trials, FOUR_LEVER_TRIALS)
    elif scenario_name == "CE3_simplified" or scenario_name == "CC3_simplified":
        # trials 1-6 have 3 levers for CC3/CE3
        trial, configs = select_random_trial(
            completed_trials, SIMPLIFIED_THREE_LEVER_TRIALS
        )
    elif scenario_name == "TwoStepTestingScenario":
        trial, configs = select_random_trial(completed_trials, TWO_STEP_TESTING_TRIALS)
    else:
        error_str = "Invalid scenario name: {}".format(scenario_name)
        raise ValueError(error_str)

    return trial, configs


def select_random_trial(completed_trials, possible_trials):
    """
    sets a new random trial
    :param completed_trials: list of trials already selected
    :param possible_trials: list of all trials possible
    :return:
    """
    if completed_trials is None:
        completed_trials = []
    if len(completed_trials) == len(possible_trials):
        return None, None

    incomplete_trials = np.setdiff1d(possible_trials, completed_trials)
    rand_trial_idx = np.random.randint(0, len(incomplete_trials))
    trial = incomplete_trials[rand_trial_idx]

    return select_trial(trial)
