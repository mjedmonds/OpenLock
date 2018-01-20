from gym_lock.common import ENTITY_STATES

REWARD_NONE = 0
REWARD_CHANGE_OBS = 0.5
REWARD_IMMOVABLE = -0.5
REWARD_UNLOCK = 10
REWARD_OPEN = 50


def determine_reward(env, action, reward_mode):
    # todo: this reward does not consider whether or not the action sequence has been finished before
    # todo: success also has the same limitation
    if reward_mode == 'basic':
        return reward_basic(env, action)
    if reward_mode == 'change_state':
        return reward_change_state(env, action)
    if reward_mode == 'unique_solutions':
        return reward_unique_solution(env, action)
    if reward_mode == 'change_state_unique_solutions':
        return reward_change_state_unique_solution(env, action)
    if reward_mode == 'negative_immovable_unique_solutions':
        return reward_negative_immovable_unique_solutions(env, action)


def reward_basic(env, action):
    '''
    Give reward of REWARD_UNLOCK for unlocking the door
    Give reward of REWARD_OPEN for opening the door
    Give reward of REWARD_NONE for anything else
    '''
    success = False
    door_lock_state = env.get_state()['OBJ_STATES']['door_lock']
    door_unlocked = door_lock_state == ENTITY_STATES['DOOR_UNLOCKED']
    # door unlocked and pushed on door
    if door_unlocked and action.name is 'push' and action.obj is 'door':
        reward = REWARD_OPEN
        success = True
    # door unlocked
    elif door_unlocked:
        reward = REWARD_UNLOCK
    # door locked
    else:
        reward = REWARD_NONE

    return reward, success


def reward_change_state(env, action):
    '''
    Give reward of REWARD_UNLOCK for unlocking the door
    Give reward of REWARD_OPEN for opening the door
    Give reward of REWARD_CHANGE_OBS for chanding the observation state
    Give reward of REWARD_NONE for anything else
    '''
    success = False
    door_lock_state = env.get_state()['OBJ_STATES']['door_lock']
    door_unlocked = door_lock_state == ENTITY_STATES['DOOR_UNLOCKED']
    # door unlocked, push_door
    if door_unlocked and action.name is 'push' and action.obj is 'door':
        reward = REWARD_OPEN
        success = True
    # door unlocked
    elif door_unlocked:
        reward = REWARD_UNLOCK
    # state change
    elif env.determine_fluent_change():
        reward = REWARD_CHANGE_OBS
    # door locked, no state change
    else:
        reward = REWARD_NONE

    return reward, success


def reward_unique_solution(env, action):
    '''
    Give reward of REWARD_UNLOCK for unlocking the door with a new action sequence
    Give reward of REWARD_OPEN for opening the door with a new action sequence
    Give reward of REWARD_CHANGE_OBS for chanding the observation state
    Give reward of REWARD_NONE for anything else
    '''
    success = False
    unique_seq = env.logger.cur_trial.determine_unique()
    door_lock_state = env.get_state()['OBJ_STATES']['door_lock']
    door_unlocked = door_lock_state == ENTITY_STATES['DOOR_UNLOCKED']
    # door unlocked, push_door
    if door_unlocked and action.name is 'push' and action.obj is 'door' and unique_seq:
        reward = REWARD_OPEN
        success = True
    # door unlocked, unique solution
    elif door_unlocked and unique_seq:
        reward = REWARD_UNLOCK
    # door locked, no state change
    else:
        reward = REWARD_NONE

    return reward, success


def reward_change_state_unique_solution(env, action):
    '''
    Give reward of REWARD_UNLOCK for unlocking the door with a new action sequence
    Give reward of REWARD_OPEN for opening the door with a new action sequence
    Give reward of REWARD_CHANGE_OBS for chanding the observation state
    Give reward of REWARD_NONE for anything else
    '''
    success = False
    unique_seq = env.logger.cur_trial.determine_unique()
    door_lock_state = env.get_state()['OBJ_STATES']['door_lock']
    door_unlocked = door_lock_state == ENTITY_STATES['DOOR_UNLOCKED']
    # door locked, state change
    if door_unlocked and action.name is 'push' and action.obj is 'door' and unique_seq:
        reward = REWARD_OPEN
        success = True
    # door unlocked
    elif door_unlocked and unique_seq:
        reward = REWARD_UNLOCK
    # state change
    elif env.determine_fluent_change():
        reward = REWARD_CHANGE_OBS
    # door locked, no state change
    else:
        reward = REWARD_NONE

    return reward, success

def reward_negative_immovable_unique_solutions(env, action):
    '''
    Give reward of REWARD_UNLOCK for unlocking the door with a new action sequence
    Give reward of REWARD_OPEN for opening the door with a new action sequence
    Give reward of REWARD_CHANGE_OBS for chanding the observation state
    Give reward of REWARD_NONE for anything else
    '''
    success = False
    unique_seq = env.logger.cur_trial.determine_unique()
    door_lock_state = env.get_state()['OBJ_STATES']['door_lock']
    door_unlocked = door_lock_state == ENTITY_STATES['DOOR_UNLOCKED']
    # door locked, state change
    if door_unlocked and action.name is 'push' and action.obj is 'door' and unique_seq:
        reward = REWARD_OPEN
        success = True
    # door unlocked
    elif door_unlocked and unique_seq:
        reward = REWARD_UNLOCK
    # determine if movable
    elif action.obj is not 'door' and not env.determine_moveable_action(action):
        reward = REWARD_IMMOVABLE
    # door locked, no state change
    else:
        reward = REWARD_NONE

    return reward, success



