REWARD_NONE = 0
REWARD_CHANGE_OBS = 0.5
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


def reward_basic(env, action):
    '''
    Give reward of REWARD_UNLOCK for unlocking the door
    Give reward of REWARD_OPEN for opening the door
    Give reward of REWARD_NONE for anything else
    '''
    success = False
    # door unlocked and pushed on door
    if env.world_def.door.lock is None and action.name is 'push' and action.params[0] is 'door':
        reward = REWARD_OPEN
        success = True
    # door unlocked
    elif env.world_def.door.lock is None:
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
    # door unlocked, push_door
    if env.world_def.door.lock is None and action.name is 'push' and action.params[0] is 'door':
        reward = REWARD_OPEN
        success = True
    # door unlocked
    elif env.world_def.door.lock is None:
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
    # door unlocked, push_door
    if env.world_def.door.lock is None and action.name is 'push' and action.params[0] is 'door' and unique_seq:
        reward = REWARD_OPEN
        success = True
    # door unlocked, unique solution
    elif env.world_def.door.lock is None and unique_seq:
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
    # door locked, state change
    if env.world_def.door.lock is None and action.name is 'push' and action.params[0] is 'door' and unique_seq:
        reward = REWARD_OPEN
        success = True
    # door unlocked
    elif env.world_def.door.lock is None and unique_seq:
        reward = REWARD_UNLOCK
    # state change
    elif env.determine_fluent_change():
        reward = REWARD_CHANGE_OBS
    # door locked, no state change
    else:
        reward = REWARD_NONE

    return reward, success





