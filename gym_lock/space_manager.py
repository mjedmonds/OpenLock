import numpy as np

from gym_lock.settings_trial import UPPER, LEFT, LOWER, UPPERLEFT, UPPERRIGHT, LOWERLEFT, LOWERRIGHT, CONFIG_TO_IDX, NUM_LEVERS
import gym_lock.common as common
from gym.spaces import MultiDiscrete


class ActionSpace:

    def __init__(self):
        pass

    @staticmethod
    def create_action_space(obj_map):
        # this must be preallocated; they are filled by position, not by symbol
        push_action_space = [None] * NUM_LEVERS
        pull_action_space = [None] * NUM_LEVERS
        door_action_space = []
        action_map = dict()
        for obj, val in obj_map.items():
            if 'button' not in obj and 'door' not in obj:
                # use position to map to an integer index
                twod_config = val.config
                lever_idx = CONFIG_TO_IDX[twod_config]

                push = 'push_{}'.format(obj)
                pull = 'pull_{}'.format(obj)

                push_action_space[lever_idx] = push
                pull_action_space[lever_idx] = pull

                action_map[push] = common.Action('push', (obj, 4))
                action_map[pull] = common.Action('pull', (obj, 4))
            if 'button' not in obj and 'door' in obj:
                push = 'push_{}'.format(obj)

                door_action_space.append(push)

                action_map[push] = common.Action('push', (obj, 4))

        action_space = push_action_space + pull_action_space + door_action_space

        return action_space, action_map


class ObservationSpace:

    def __init__(self, num_levers):
        self.multi_discrete = self.create_observation_space(num_levers)
        self.num_levers = num_levers

    @staticmethod
    def create_observation_space(num_levers):
        discrete_space = []
        # first num_levers represent the state of the levers
        for i in range(num_levers):
            discrete_space.append([0, 1])
        # second num_levers represent the colors of the levers
        for i in range(num_levers):
            discrete_space.append([0, 1])
        discrete_space.append([0, 1])       # door lock
        discrete_space.append([0, 1])       # door open
        multi_discrete = MultiDiscrete(discrete_space)
        return multi_discrete

    def create_discrete_observation_from_state(self, world_def):
        locks = world_def.get_locks()
        world_state = world_def.get_state()

        # need one element for state and color of each lock, need two addition for door lock status and door status
        observation = [None] * (len(locks) * 2 + 2)
        observation_labels = [None] * (len(locks) * 2 + 2)

        for lock in locks:
            # convert to index based on lever position
            lever_idx = CONFIG_TO_IDX[lock.config]

            lever_state = np.int8(world_state['OBJ_STATES'][lock.name])
            lever_active = np.int8(lock.determine_active())

            observation_labels[lever_idx] = lock.name
            observation[lever_idx] = lever_state
            observation_labels[lever_idx + self.num_levers] = lock.name + '_active'
            observation[lever_idx + self.num_levers] = lever_active

        observation_labels[-1] = 'door'
        observation[-1] = np.int8(world_state['OBJ_STATES']['door'])
        observation_labels[-2] = 'door_lock'
        observation[-2] = np.int8(world_state['OBJ_STATES']['door_lock'])

        return observation, observation_labels


