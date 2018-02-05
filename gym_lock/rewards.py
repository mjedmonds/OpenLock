from gym_lock.common import ENTITY_STATES

REWARD_NONE = 0
REWARD_CHANGE_OBS = 0.5
REWARD_IMMOVABLE = -0.5
REWARD_REPEATED_ACTION = -0.25
REWARD_PARTIAL_SEQ = 1
REWARD_UNLOCK = 10
REWARD_OPEN = 50
SOLUTION_MULTIPLIER = 1.5


def determine_reward(env, action, reward_mode):
    # todo: this reward does not consider whether or not the action sequence has been finished before
    # todo: success also has the same limitation
    reward = 0
    success = door_open(env, action)
    if reward_mode == 'basic':
        reward = reward_basic(env, action)
    elif reward_mode == 'change_state':
        reward = reward_change_state(env, action)
    elif reward_mode == 'unique_solutions':
        reward = reward_unique_solution(env, action)
    elif reward_mode == 'change_state_unique_solutions':
        reward = reward_change_state_unique_solution(env, action)
    elif reward_mode == 'negative_immovable_unique_solutions':
        reward = reward_negative_immovable_unique_solutions(env, action)
    elif reward_mode == 'negative_immovable':
        reward = reward_negative_immovable(env, action)
    elif reward_mode == 'negative_immovable_partial_action_seq':
        reward = reward_negative_immovable_partial_seq(env, action)
    elif reward_mode == 'negative_immovable_negative_repeat':
        reward = reward_negative_immovable_negative_repeat(env, action)
    elif reward_mode == 'negative_immovable_solution_multiplier':
        reward = reward_negative_immovable_solution_multiplier(env, action)
    elif reward_mode == 'negative_immovable_partial_action_seq_solution_multiplier':
        reward = reward_negative_immovable_partial_seq_solution_multiplier(env, action)
    else:
        raise ValueError(str('Unknown reward function mode: %s'.format(reward_mode)))

    return reward, success


def door_open(env, action):
    if door_unlocked(env) and action.name is 'push' and action.obj is 'door':
        return True
    else:
        return False


def door_unlocked(env):
    door_lock_state = env.get_state()['OBJ_STATES']['door_lock']
    if door_lock_state == ENTITY_STATES['DOOR_UNLOCKED']:
        return True
    else:
        return False


def reward_basic(env, action):
    '''
    Give reward of REWARD_UNLOCK for unlocking the door
    Give reward of REWARD_OPEN for opening the door
    Give reward of REWARD_NONE for anything else
    '''
    # door unlocked and pushed on door
    if door_open(env, action):
        reward = REWARD_OPEN
    # door unlocked
    elif door_unlocked(env):
        reward = REWARD_UNLOCK
    # door locked
    else:
        reward = REWARD_NONE
    return reward


def reward_change_state(env, action):
    '''
    Give reward of REWARD_UNLOCK for unlocking the door
    Give reward of REWARD_OPEN for opening the door
    Give reward of REWARD_CHANGE_OBS for chanding the observation state
    Give reward of REWARD_NONE for anything else
    '''
    # door unlocked, push_door
    if door_open(env, action):
        reward = REWARD_OPEN
    # door unlocked
    elif door_unlocked(env):
        reward = REWARD_UNLOCK
    # state change
    elif env.determine_fluent_change():
        reward = REWARD_CHANGE_OBS
    # door locked, no state change
    else:
        reward = REWARD_NONE
    return reward


def reward_unique_solution(env, action):
    '''
    Give reward of REWARD_UNLOCK for unlocking the door with a new action sequence
    Give reward of REWARD_OPEN for opening the door with a new action sequence
    Give reward of REWARD_CHANGE_OBS for chanding the observation state
    Give reward of REWARD_NONE for anything else
    '''
    unique_seq = env.logger.cur_trial.determine_unique()
    # door unlocked, push_door
    if door_open(env, action) and unique_seq:
        reward = REWARD_OPEN
    # door unlocked, unique solution
    elif door_unlocked(env) and unique_seq:
        reward = REWARD_UNLOCK
    # door locked, no state change
    else:
        reward = REWARD_NONE
    return reward


def reward_change_state_unique_solution(env, action):
    '''
    Give reward of REWARD_UNLOCK for unlocking the door with a new action sequence
    Give reward of REWARD_OPEN for opening the door with a new action sequence
    Give reward of REWARD_CHANGE_OBS for chanding the observation state
    Give reward of REWARD_NONE for anything else
    '''
    unique_seq = env.logger.cur_trial.determine_unique()
    # door locked, state change
    if door_open(env, action) and unique_seq:
        reward = REWARD_OPEN
    # door unlocked
    elif door_unlocked(env) and unique_seq:
        reward = REWARD_UNLOCK
    # state change
    elif env.determine_fluent_change():
        reward = REWARD_CHANGE_OBS
    # door locked, no state change
    else:
        reward = REWARD_NONE
    return reward


def reward_negative_immovable_unique_solutions(env, action):
    '''
    Give reward of REWARD_UNLOCK for unlocking the door with a new action sequence
    Give reward of REWARD_OPEN for opening the door with a new action sequence
    Give reward of REWARD_IMMOVABLE for interacting with a lever that cannot move
    Give reward of REWARD_NONE for anything else
    '''
    unique_seq = env.logger.cur_trial.determine_unique()
    # door locked, state change
    if door_open(env, action) and unique_seq:
        reward = REWARD_OPEN
    # door unlocked
    elif door_unlocked(env) and unique_seq:
        reward = REWARD_UNLOCK
    # determine if movable
    elif action.obj is not 'door' and not env.determine_moveable_action(action):
        reward = REWARD_IMMOVABLE
    # door locked, no state change
    else:
        reward = REWARD_NONE
    return reward


def reward_negative_immovable(env, action):
    '''
    Give reward of REWARD_UNLOCK for unlocking the door with a new action sequence
    Give reward of REWARD_OPEN for opening the door with a new action sequence
    Give reward of REWARD_IMMOVABLE for interacting with a lever that cannot move
    Give reward of REWARD_NONE for anything else
    '''
    # door locked, state change
    if door_open(env, action):
        reward = REWARD_OPEN
    # door unlocked
    elif door_unlocked(env):
        reward = REWARD_UNLOCK
    # determine if movable
    elif action.obj is not 'door' and not env.determine_moveable_action(action):
        reward = REWARD_IMMOVABLE
    # door locked, no state change
    else:
        reward = REWARD_NONE
    return reward


def reward_negative_immovable_partial_seq(env, action):
    '''
    Give reward of REWARD_UNLOCK for unlocking the door with a new action sequence
    Give reward of REWARD_OPEN for opening the door with a new action sequence
    Give reward of REWARD_IMMOVABLE for interacting with a lever that cannot move
    Give reward of REWARD_PARTIAL_SEQ for any partial subsquence
    Give reward of REWARD_NONE for anything else
    '''
    # door locked, state change
    if door_open(env, action):
        reward = REWARD_OPEN
    # door unlocked
    elif door_unlocked(env):
        reward = REWARD_UNLOCK
    # determine if partial subsequence of a solution action seq
    elif env.determine_partial_seq():
        reward = REWARD_PARTIAL_SEQ
    # determine if movable
    elif action.obj is not 'door' and not env.determine_moveable_action(action):
        reward = REWARD_IMMOVABLE
    # door locked, no state change
    else:
        reward = REWARD_NONE
    return reward


def reward_negative_immovable_negative_repeat(env, action):
    '''
    Give reward of REWARD_UNLOCK for unlocking the door with a new action sequence
    Give reward of REWARD_OPEN for opening the door with a new action sequence
    Give reward of REWARD_IMMOVABLE for interacting with a lever that cannot move
    Give reward of REWARD_PARTIAL_SEQ for any partial subsquence
    Give reward of REWARD_NONE for anything else
    '''
    # door locked, state change
    if door_open(env, action):
        reward = REWARD_OPEN
    # door unlocked
    elif door_unlocked(env):
        reward = REWARD_UNLOCK
    # determine if partial subsequence of a solution action seq
    elif env.determine_repeated_action():
        reward = REWARD_REPEATED_ACTION
    # determine if movable
    elif action.obj is not 'door' and not env.determine_moveable_action(action):
        reward = REWARD_IMMOVABLE
    # door locked, no state change
    else:
        reward = REWARD_NONE
    return reward


def reward_negative_immovable_solution_multiplier(env, action):
    '''
    Give reward of REWARD_UNLOCK for unlocking the door with a new action sequence
    Give reward of REWARD_OPEN for opening the door with a new action sequence
    Give reward of REWARD_IMMOVABLE for interacting with a lever that cannot move
    Give reward of REWARD_NONE for anything else

    Each new, unique solution found is multiplied by 1.5x the previous multiplier.
    The multiplier starts at 1. For instance, the reward for finding the first unique
    solution would be REWARD_OPEN, but the reward for finding the section unique solution
    would be 1.5 * REWARD_OPEN, the third 1.5 * 1.5 * REWARD_OPEN. This encourages the
    agent to find unique solutions without penalizing for finding repeated solutions
    '''
    num_solutions_found = len(env.logger.cur_trial.completed_solutions)
    multiplier = max(1, 1 * SOLUTION_MULTIPLIER * num_solutions_found)
    unique_seq = env.logger.cur_trial.determine_unique()
    # door unlocked
    if door_open(env, action) and unique_seq:
        reward = REWARD_OPEN * multiplier
    # door unlocked
    elif door_open(env, action):
        reward = REWARD_OPEN
    # door unlocked
    elif door_unlocked(env) and unique_seq:
        reward = REWARD_UNLOCK * multiplier
    # door unlocked
    elif door_unlocked(env):
        reward = REWARD_UNLOCK
    # determine if movable
    elif action.obj is not 'door' and not env.determine_moveable_action(action):
        reward = REWARD_IMMOVABLE
    # door locked, no state change
    else:
        reward = REWARD_NONE
    return reward


def reward_negative_immovable_partial_seq_solution_multiplier(env, action):
    '''
    Give reward of REWARD_UNLOCK for unlocking the door with a new action sequence
    Give reward of REWARD_OPEN for opening the door with a new action sequence
    Give reward of REWARD_PARTIAL_SEQ for any partial subsquence
    Give reward of REWARD_IMMOVABLE for interacting with a lever that cannot move
    Give reward of REWARD_NONE for anything else

    Each new, unique solution found is multiplied by 1.5x the previous multiplier.
    The multiplier starts at 1. For instance, the reward for finding the first unique
    solution would be REWARD_OPEN, but the reward for finding the section unique solution
    would be 1.5 * REWARD_OPEN, the third 1.5 * 1.5 * REWARD_OPEN. This encourages the
    agent to find unique solutions without penalizing for finding repeated solutions
    '''
    num_solutions_found = len(env.logger.cur_trial.completed_solutions)
    multiplier = max(1, 1 * SOLUTION_MULTIPLIER * num_solutions_found)
    unique_seq = env.logger.cur_trial.determine_unique()
    # door unlocked
    if door_open(env, action) and unique_seq:
        reward = REWARD_OPEN * multiplier
    # door unlocked
    elif door_open(env, action):
        reward = REWARD_OPEN
    # door unlocked
    elif door_unlocked(env) and unique_seq:
        reward = REWARD_UNLOCK * multiplier
    # door unlocked
    elif door_unlocked(env):
        reward = REWARD_UNLOCK
    # determine if partial subsequence of a solution action seq
    elif env.determine_partial_seq():
        reward = REWARD_PARTIAL_SEQ
    # determine if movable
    elif action.obj is not 'door' and not env.determine_moveable_action(action):
        reward = REWARD_IMMOVABLE
    # door locked, no state change
    else:
        reward = REWARD_NONE
    return reward
