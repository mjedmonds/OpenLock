'''
Outlines the structure and common functionality across scenarios
'''
import gym_lock.common as common


class Scenario(object):

    def set_lever_configs(self, lever_configs):
        self.lever_configs = lever_configs

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
                if self.fsmm._extract_entity_state(self.fsmm.latent_fsm.state, door) != 'locked,':
                    self.fsmm.latent_fsm.trigger('lock_{}'.format(door))

    def reset(self):
        '''
        resets the FSM to the initial state for both FSMs
        :return:
        '''
        self.fsmm.reset()

    def update_observable(self):
        '''
        updates observable fsm based on some change in the observable fsm, if needed
        '''
        pass

    def update_state_machine(self):
        '''
        Updates the finite state machines according to object status in the Box2D environment
        '''
        prev_state = self.fsmm.observable_fsm.state

        # execute state transitions
        # check locks
        for name, obj in self.world_def.obj_map.items():
            fsm_name = name + ':'
            if 'button' not in name and 'door' not in name and 'inactive' not in name:
                if obj.int_test(obj.joint):
                    if self.fsmm._extract_entity_state(self.fsmm.observable_fsm.state, fsm_name) != 'pushed,':
                        # push lever
                        action = 'push_{}'.format(fsm_name)
                        self.fsmm.execute_action(action)
                        self._update_env()
                else:
                    if self.fsmm._extract_entity_state(self.fsmm.observable_fsm.state, fsm_name) != 'pulled,':
                        # push lever
                        action = 'pull_{}'.format(fsm_name)
                        self.fsmm.execute_action(action)
                        self._update_env()

    def init_scenario_env(self, world_def):
        # todo: come up with a better way to set self.world_def without passing as an argument here
        self.world_def = world_def

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

            lock = common.Lock(self.world_def, role, two_d_config, color, opt_params)
            self.world_def.obj_map[role] = lock

    def _update_latent_objs(self):
        latent_states = self.fsmm.get_latent_states()
        for latent_var in latent_states.keys():
            # ---------------------------------------------------------------
            # Add code to change part of the environment corresponding to a latent variable here
            # ---------------------------------------------------------------
            if latent_var == 'door:':
                if latent_states[latent_var] == 'locked,' and self.world_def.door_lock is None:
                    self.world_def.lock_door()
                elif latent_states[latent_var] == 'unlocked,' and self.world_def.door_lock is not None:
                    self.world_def.unlock_door()

    def _update_observable_objs(self):
        '''
        updates observable objects in the Box2D environment based on the observable state of the FSM.
        Almost always is Scenario-specific, so we pass here
        :return:
        '''
        pass