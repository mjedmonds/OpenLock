import numpy as np
import re

from gym_lock.settings_trial import UPPER, LEFT, LOWER, UPPERLEFT, UPPERRIGHT, LOWERLEFT, LOWERRIGHT, CONFIG_TO_IDX, NUM_LEVERS, LEVER_CONFIGS
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
        self.state = None
        self.state_labels = None

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

    def create_discrete_observation_from_simulator(self, world_def):
        '''
        Constructs a discrete observation from the physics simulator
        :param world_def:
        :return:
        '''
        levers = world_def.get_levers()
        self.num_levers = len(levers)
        world_state = world_def.get_state()

        # need one element for state and color of each lock, need two addition for door lock status and door status
        self.state = [None] * (self.num_levers * 2 + 2)
        self.state_labels = [None] * (self.num_levers * 2 + 2)

        for lever in levers:
            # convert to index based on lever position
            lever_idx = CONFIG_TO_IDX[lever.config]

            lever_state = np.int8(world_state['OBJ_STATES'][lever.name])
            lever_active = np.int8(lever.determine_active())

            self.state_labels[lever_idx] = lever.name
            self.state[lever_idx] = lever_state
            self.state_labels[lever_idx + self.num_levers] = lever.name + '_active'
            self.state[lever_idx + self.num_levers] = lever_active

        self.state_labels[-1] = 'door'
        self.state[-1] = np.int8(world_state['OBJ_STATES']['door'])
        self.state_labels[-2] = 'door_lock'
        self.state[-2] = np.int8(world_state['OBJ_STATES']['door_lock'])

        return self.state, self.state_labels

    def create_discrete_observation_from_fsm(self, scenario):
        '''
        constructs a discrete observation from the underlying FSM
        Used when the physics simulator is being bypassed
        :param fsmm:
        :return:
        '''
        levers = scenario.levers
        self.num_levers = len(levers)
        scenario_state = scenario.get_state()

        # need one element for state and color of each lock, need two addition for door lock status and door status
        self.state = [None] * (self.num_levers * 2 + 2)
        self.state_labels = [None] * (self.num_levers * 2 + 2)

        inactive_lock_regex = '^inactive[0-9]+$'

        # lever states
        for lever in levers:
            lever_idx = CONFIG_TO_IDX[lever.config]

            # inactive lever, state is constant
            if re.search(inactive_lock_regex, lever.name):
                lever_active = np.int8(common.ENTITY_STATES['LEVER_INACTIVE'])
            else:
                lever_active = np.int8(common.ENTITY_STATES['LEVER_ACTIVE'])

            lever_state = np.int8(scenario_state['OBJ_STATES'][lever.name])

            self.state_labels[lever_idx] = lever.name
            self.state[lever_idx] = lever_state

            self.state_labels[lever_idx + self.num_levers] = lever.name + '_active'
            self.state[lever_idx + self.num_levers] = lever_active

        # update door state
        door_lock_name = 'door_lock'
        door_lock_state = np.int8(scenario_state['OBJ_STATES'][door_lock_name])

        # todo: this is a hack to get whether or not the door is actually open; it should be part of the FSM
        door_name = 'door'
        door_state = np.int8(scenario_state['OBJ_STATES'][door_name])

        self.state_labels[-1] = door_name
        self.state[-1] = door_state
        self.state_labels[-2] = door_lock_name
        self.state[-2] = door_lock_state

        return self.state, self.state_labels
