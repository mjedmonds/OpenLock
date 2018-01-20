'''
Outlines the structure and common functionality across scenarios
'''
import re
import numpy as np

import gym_lock.common as common
from gym_lock.settings_trial import UPPER, LEFT, LOWER, UPPERLEFT, UPPERRIGHT, LOWERLEFT, LOWERRIGHT, CONFIG_TO_IDX, NUM_LEVERS, LEVER_CONFIGS


class Scenario(object):

    def __init__(self, use_physics=True):
        self.use_physics = use_physics
        self.levers = []
        self.lever_configs = None
        self.world_def = None
        self.door_state = common.ENTITY_STATES['DOOR_CLOSED']
        self.obj_map = dict()

    def set_lever_configs(self, lever_configs):
        self.lever_configs = lever_configs
        self.levers = []

        num_inactive = 0        # give inactive levers a unique name
        for lever_config in self.lever_configs:
            two_d_config, role, opt_params = lever_config
            # give unique names to every inactive
            if role == 'inactive':
                role = 'inactive{}'.format(num_inactive)
                num_inactive += 1
                color = common.COLORS['inactive']
            else:
                color = common.COLORS['active']

            # world_def will be initialized with init_scenario_env
            lever = common.Lever(role, two_d_config, color, opt_params)
            self.levers.append(lever)

    def add_no_ops(self, lock, pushed, pulled):
        # add transitions from state back to same state when performing an action that already matches the state
        for state in pulled:
            self.fsmm.observable_fsm.machine.add_transition('pull_{}'.format(lock), state, state)
        for state in pushed:
            self.fsmm.observable_fsm.machine.add_transition('push_{}'.format(lock), state, state)
        # generate the complement states that don't have transitions
        comp_pulled, comp_pushed = self.generate_complement_states(lock, pushed, pulled)
        # add transitions from state back to same state when
        for state in comp_pushed:
            self.fsmm.observable_fsm.machine.add_transition('pull_{}'.format(lock), state, state)
        for state in comp_pulled:
            self.fsmm.observable_fsm.machine.add_transition('push_{}'.format(lock), state, state)

    def generate_complement_states(self, lock, pushed, pulled):
        comp_pulled = [s for s in self.fsmm.observable_fsm.state_permutations if s not in pulled and lock + 'pulled,' in s]
        comp_pushed = [s for s in self.fsmm.observable_fsm.state_permutations if s not in pushed and lock + 'pushed,' in s]
        return comp_pulled, comp_pushed

    def add_nothing_transition(self):
        # add nothing transition
        for state in self.fsmm.observable_fsm.state_permutations:
            self.fsmm.observable_fsm.machine.add_transition('nothing', state, state)
        for state in self.fsmm.latent_fsm.state_permutations:
            self.fsmm.latent_fsm.machine.add_transition('nothing', state, state)

    def add_door_transitions(self):
        for door in self.latent_vars:
            #todo: only supports one door
            self.fsmm.latent_fsm.machine.add_transition('lock_{}'.format(door), 'door:locked,', 'door:locked,')
            self.fsmm.latent_fsm.machine.add_transition('lock_{}'.format(door), 'door:unlocked,', 'door:locked,')
            self.fsmm.latent_fsm.machine.add_transition('unlock_{}'.format(door), 'door:locked,', 'door:unlocked,')
            self.fsmm.latent_fsm.machine.add_transition('unlock_{}'.format(door), 'door:unlocked,', 'door:unlocked,')

    def update_latent(self):
        '''
        logic to transition in the latent state space based on the observable state space, if needed
        '''
        observable_state = self.fsmm.observable_fsm.state
        if observable_state in self.door_unlock_criteria:
            # todo: currently this will unlock all doors, need to make it so each door has it's own connection to observable state
            for door in self.latent_vars:
                self.fsmm.latent_fsm.trigger('unlock_{}'.format(door))
        else:
            # todo: currently this will lock all doors, need to make it so each door has it's own connection to observable state
            for door in self.latent_vars:
                if self.fsmm.extract_entity_state(self.fsmm.latent_fsm.state, door) != 'locked,':
                    self.fsmm.latent_fsm.trigger('lock_{}'.format(door))

    def reset(self):
        '''
        resets the FSM to the initial state for both FSMs
        :return:
        '''
        self.fsmm.reset()
        self.door_state = common.ENTITY_STATES['DOOR_CLOSED']

    def get_obj_state(self):
        state = dict()

        inactive_lock_regex = '^inactive[0-9]+$'

        fsm_observable_states = self.fsmm.get_observable_states()
        fsm_latent_states = self.fsmm.get_latent_states()

        # lever states
        for lever in self.levers:
            # inactive lever, state is constant
            if re.search(inactive_lock_regex, lever.name):
                lever_state = np.int8(common.ENTITY_STATES['LEVER_PULLED'])
            else:
                fsm_name = lever.name + ':'
                lever_state = fsm_observable_states[fsm_name]
                lever_state = lever_state[:len(lever_state)-1].upper()
                lever_state = np.int8(common.ENTITY_STATES['LEVER_' + lever_state])

            state[lever.name] = lever_state

        # update door state
        door_lock_state = fsm_latent_states['door:']
        door_lock_state = door_lock_state[:len(door_lock_state)-1].upper()
        door_lock_state = np.int8(common.ENTITY_STATES['DOOR_' + door_lock_state])

        # todo: this is a hack to get whether or not the door is actually open; it should be part of the FSM
        door_state = np.int8(self.door_state)

        state['door'] = door_state
        state['door_lock'] = door_lock_state

        return state

    def get_state(self):
        obj_states = self.get_obj_state()
        fsm_state = self.fsmm.get_internal_state()
        return {
            'OBJ_STATES': obj_states,
            '_FSM_STATE': fsm_state,
        }

    def update_observable(self):
        '''
        updates observable fsm based on some change in the observable fsm, if needed
        '''
        pass

    def update_state_machine(self, action=None):
        '''
        Updates the finite state machines according to object status in the Box2D environment
        '''
        prev_state = self.fsmm.observable_fsm.state

        # do not bypass the simulator
        if self.use_physics:
            # execute state transitions
            # check locks
            for name, obj in self.world_def.obj_map.items():
                fsm_name = name + ':'
                if 'button' not in name and 'door' not in name and 'inactive' not in name:
                    if obj.int_test(obj.joint):
                        self.execute_push(fsm_name)
                    else:
                        self.execute_pull(fsm_name)
            # todo, this is a dirty hack to see if the door is actually opened
            if action is not None and action.name is 'push' and action.obj == 'door':
                self.push_door()
        # bypass physics, action must be specified
        else:
            if action is not None:
                self.execute_action(action)

    def execute_action(self, action):
        if self.use_physics:
            raise RuntimeError('Attempting to directly run FSM action without bypassing physics simulator')
        obj_name = action.obj
        fsm_name = obj_name + ':'
        inactive_lock_regex = '^inactive[0-9]+$'
        # inactive levers are always no-ops in FSM
        if not re.search(inactive_lock_regex, obj_name):
            if action.name == 'push':
                self.execute_push(fsm_name)
            elif action.name == 'pull':
                self.execute_pull(fsm_name)

            self.update_latent()

    def execute_push(self, obj_name):
        if self.fsmm.extract_entity_state(self.fsmm.observable_fsm.state, obj_name) != 'pushed,':
            # push lever
            action = 'push_{}'.format(obj_name)
            self.fsmm.execute_action(action)
            self._update_env()
            self.update_latent()

    def execute_pull(self, obj_name):
        if self.fsmm.extract_entity_state(self.fsmm.observable_fsm.state, obj_name) != 'pulled,':
            # push lever
            action = 'pull_{}'.format(obj_name)
            self.fsmm.execute_action(action)
            self._update_env()
            self.update_latent()

    # todo: this is a quick hack to represent actually opening the door, which is not included in any transition
    def push_door(self):
        if self.fsmm.extract_entity_state(self.fsmm.latent_fsm.state, 'door:') == 'unlocked,':
            self.door_state = common.ENTITY_STATES['DOOR_OPENED']
            self.update_latent()

    def init_scenario_env(self, world_def=None):
        if self.use_physics and world_def is None:
            raise ValueError('No world_def passed to init_scenario_env while using physics')

        if self.use_physics:
            # todo: come up with a better way to set self.world_def without passing as an argument here
            self.world_def = world_def

            for lever in self.levers:
                if lever.opt_params:
                    lever.create_lever(self.world_def, lever.config, **lever.opt_params)
                else:
                    lever.create_lever(self.world_def, lever.config)
                self.world_def.obj_map[lever.name] = lever
            self.obj_map = self.world_def.obj_map
        # bypassing physics, obj_map consists of door and levers
        else:
            for lever in self.levers:
                self.obj_map[lever.name] = lever

    def _update_env(self):
        '''
        updates the Box2D environment based on the state of the finite state machine
        '''
        # update physics simulator environment based on FSM changes
        if self.use_physics:
            self._update_latent_objs()
            self._update_observable_objs()

    def _update_latent_objs(self):
        latent_states = self.fsmm.get_latent_states()
        for latent_var in latent_states.keys():
            # ---------------------------------------------------------------
            # Add code to change part of the environment corresponding to a latent variable here
            # ---------------------------------------------------------------
            if latent_var == 'door:':
                if latent_states[latent_var] == 'locked,' and self.world_def.door.lock is None:
                    self.world_def.lock_door()
                elif latent_states[latent_var] == 'unlocked,' and self.world_def.door.lock is not None:
                    self.world_def.unlock_door()

    def _update_observable_objs(self):
        '''
        updates observable objects in the Box2D environment based on the observable state of the FSM.
        Almost always is Scenario-specific, so we pass here
        :return:
        '''
        pass
