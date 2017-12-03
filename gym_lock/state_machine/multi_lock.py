from transitions.extensions import GraphMachine as Machine
from gym_lock.state_machine.mdp import StateMachineMDP

import mdptoolbox
import itertools
import copy


def cartesian_product(*lists):
    result = [[]]
    for list in lists:
        result = [x + [y] for x in result for y in list]
    return [''.join(elem) for elem in result]


class FiniteStateMachine():
    def __init__(self, fsm_manager, name, vars, states, initial_state):
        self.fsm_manager = fsm_manager
        self.name = name
        self.vars = vars
        self.state_permutations = self._permutate_states(states)
        self.initial_state = initial_state

        self.machine = Machine(model=self,
                               states=self.state_permutations,
                               initial=self.initial_state,
                               ignore_invalid_triggers=True,
                               auto_transitions=False)

    def _permutate_states(self, states):
        assert len(self.vars) > 0

        v_list = cartesian_product([self.vars[0]], states)
        for i in range(1, len(self.vars)):
            v_list = cartesian_product(v_list, cartesian_product([self.vars[i]], states))

        # return cartesian_product(observable_v_list, door_list)
        return v_list

    def update_manager(self):
        '''
        tells FSM manager to update the other FSM (latent/observable) based on the changes this FSM (obserable/latent) made
        :return:
        '''
        if self.name == 'observable':
            self.fsm_manager.update_latent()
        else:
            self.fsm_manager.update_observable()

class MultiDoorLockFSM(object):

    observable_states = ['pulled,', 'pushed,']     # '+' -> locked/pulled, '-' -> unlocked/pushed
    # todo: make names between obj_map in env consistent with names in FSM (extra ':' in FSM)
    observable_vars = ['l0:', 'l1:', 'l2:']
    observable_initial_state = 'l0:pulled,l1:pulled,l2:pushed,'

    latent_states = ['unlocked,', 'locked,']     # '+' -> open, '-' -> closed
    latent_vars = ['door:']
    latent_initial_state = 'door:locked,'

    # define observable states that trigger changes in the latent space;
    # this is the clue between the two machines.
    # Here we assume if the observable case is in any criteria than those listed, the door is locked
    door_unlock_criteria = ['l0:pushed,l1:pushed,l2:pulled,']



    def __init__(self, env):
        self.env = env # handle to the Box2D environment

        self.actions = ['nothing'] \
                       + ['pull_{}'.format(lock) for lock in self.observable_vars] \
                       + ['push_{}'.format(lock) for lock in self.observable_vars]

        self.observable_fsm = FiniteStateMachine(fsm_manager=self,
                                                 name='observable',
                                                 vars=self.observable_vars,
                                                 states=self.observable_states,
                                                 initial_state=self.observable_initial_state)

        self.latent_fsm = FiniteStateMachine(fsm_manager=self,
                                             name='latent',
                                             vars=self.latent_vars,
                                             states=self.latent_states,
                                             initial_state=self.latent_initial_state)

        # add unlock/lock transition for every lock
        for lock in self.observable_fsm.vars:
            if lock == 'l2:':
                pulled = [s for s in self.observable_fsm.state_permutations if lock + 'pulled,' in s and 'l0:pushed,' in s and 'l1:pushed,' in s]
                pushed = [s for s in self.observable_fsm.state_permutations if lock + 'pushed,' in s and 'l0:pushed,' in s and 'l1:pushed,' in s]
            else:
                pulled = [s for s in self.observable_fsm.state_permutations if lock + 'pulled,' in s]
                pushed = [s for s in self.observable_fsm.state_permutations if lock + 'pushed,' in s]
            for pulled_state, pushed_state in zip(pulled, pushed):
                # these transitions need to change the latent FSM, so we update the manager after executing them
                self.observable_fsm.machine.add_transition('pull_{}'.format(lock), pushed_state, pulled_state, after='update_manager')
                self.observable_fsm.machine.add_transition('push_{}'.format(lock), pulled_state, pushed_state, after='update_manager')

        # add nothing transition
        for state in self.observable_fsm.state_permutations:
            self.observable_fsm.machine.add_transition('nothing', state, state)
        for state in self.latent_fsm.state_permutations:
            self.latent_fsm.machine.add_transition('nothing', state, state)

        for door in self.latent_vars:
            self.latent_fsm.machine.add_transition('lock_{}'.format(door), 'door:unlocked,', 'door:locked,')
            self.latent_fsm.machine.add_transition('unlock_{}'.format(door), 'door:locked,', 'door:unlocked,')


        # add door open/close transition
        # self.machine.add_transition('open', 'l0-l1-l2-o-', 'l0-l1-l2-o+')
        # self.machine.add_transition('close', 'l0-l1-l2-o+', 'l0-l1-l2-o-')
        # Door has one condition to unlock it, (from 'l0:pushed,l1:pushed,l2:pulled,')
        # but many potential transitions to re-lock it (if any of the other locks change)
        # self.machine.add_transition('unlock', 'l0:pushed,l1:pushed,l2:pulled,door:locked,', self.door_unlocked_state)
        # Any of the levers may cause the door to lock; need transition for each
        # for lock in MultiDoorLockFSM.locks:
        #     next_lock_state = self._negate_obj_state(self.door_unlocked_state, lock)
        #     lock_state_change = self.change_obj_state(self.door_unlocked_state, lock, next_lock_state)
        #     final_state = self.change_obj_state(lock_state_change, 'door:', 'locked,')
        #     self.machine.add_transition('{}_door_lock'.format(lock), self.door_unlocked_state, final_state)

    # logic to transition in the latent state space based on the observable state space
    def update_latent(self):
        observable_state = self.observable_fsm.state
        if observable_state in self.door_unlock_criteria:
            # todo: currently this will unlock all doors, need to make it so each door has it's own connection to observable state
            for door in self.latent_vars:
                self.latent_fsm.trigger('unlock_{}'.format(door))
        else:
            # todo: currently this will lock all doors, need to make it so each door has it's own connection to observable state
            for door in self.latent_vars:
                if self._extract_entity_state(self.latent_fsm.state, door) != 'locked,':
                    self.latent_fsm.trigger('lock_{}'.format(door))

    # updates observable fsm based on some change in the latent fsm, if needed
    def update_observable(self):
        pass

    def get_latent_states(self):
        '''
        extracts latent variables and their state into a dictonary. key: variable. value: variable state
        :return: dictionary of variables to their corresponding variable state
        '''
        latent_states = dict()
        for latent_var in self.latent_vars:
            latent_states[latent_var] = self._extract_entity_state(self.latent_fsm.state, latent_var)
        return latent_states

        # parses out the state of a specified object from a full state string

    def get_observable_states(self):
        '''
        extracts observable variables and their state into a dictonary. key: variable. value: variable state
        :return: dictionary of variables to their corresponding variable state
        '''
        observable_states = dict()
        for observable_var in self.observable_vars:
            observable_states[observable_var] = self._extract_entity_state(self.observable_fsm.state, observable_var)
        return observable_states

    def update_state_machine(self):
        '''
        Updates the finite state machines according to object status in the Box2D environment
        '''
        prev_state = self.observable_fsm.state

        # execute state transitions
        # check locks
        for name, obj in self.env.obj_map.items():
            fsm_name = name + ':'
            if 'button' not in name and 'door' not in name:
                if obj.int_test(obj.joint):
                    if self._extract_entity_state(self.observable_fsm.state, fsm_name) != 'pushed,':
                        # push lever
                        action = 'push_{}'.format(fsm_name)
                        self._execute_action(action)
                        self._update_env()
                else:
                    if self._extract_entity_state(self.observable_fsm.state, fsm_name) != 'pulled,':
                        # push lever
                        action = 'pull_{}'.format(fsm_name)
                        self._execute_action(action)
                        self._update_env()

    def _execute_action(self, action):
        if action in self.actions:
            # changes in observable FSM will trigger a callback to update the latent FSM if needed
            self.observable_fsm.trigger(action)
        else:
            raise ValueError('unknown action \'{}'.format(action) + '\'')

    def _update_env(self):
        '''
        updates the Box2D environment based on the state of the finite state machine
        '''
        self._update_latent_objs()
        self._update_observable_objs()

    def _update_latent_objs(self):
        '''
        updates latent objects in the Box2D environment based on state of the latent finite state machine
        '''
        latent_states = self.get_latent_states()
        for latent_var in latent_states.keys():
            # ---------------------------------------------------------------
            # Add code to change part of the environment corresponding to a latent variable here
            # ---------------------------------------------------------------
            if latent_var == 'door:':
                if latent_states[latent_var] == 'locked,' and self.env.door_lock is None:
                    self.env._lock_door()
                elif latent_states[latent_var] == 'unlocked,' and self.env.door_lock is not None:
                    self.env._unlock_door()

    def _update_observable_objs(self):
        '''
        updates observable objects in the Box2D environment based on the observable state of the finite state machine
        '''
        observable_states = self.get_observable_states()
        for observable_var in observable_states.keys():
            # ---------------------------------------------------------------
            # add code to change part of the environment based on the state of an observable variable here
            # ---------------------------------------------------------------
            if observable_var == 'l2:':
                # unlock l2 based on status of l0, l1, part of multi-lock FSM
                if 'l0:pushed,' in self.observable_fsm.state and 'l1:pushed,' in self.observable_fsm.state:
                    self.env.unlock_lever('l2')
                else:
                    self.env.lock_lever('l2')

    # @property
    # def actions(self):
    #     return self.observable_fsm.machine.get_triggers(self.observable_fsm.state)

    @classmethod
    def _init_states(self):
        assert len(self.observable_vars) > 0

        observable_v_list = cartesian_product([self.observable_vars[0]], self.observable_states)
        for i in range(1, len(self.observable_vars)):
            observable_v_list = cartesian_product(observable_v_list, cartesian_product([self.observable_vars[i]], self.observable_states))

        latent_v_list = cartesian_product([self.latent_vars[0]], self.latent_states)
        for i in range(1, len(self.latent_vars)):
            door_list = cartesian_product(latent_v_list, cartesian_product([self.latent_vars[i]], self.latent_states))

        # return cartesian_product(observable_v_list, door_list)
        return observable_v_list, latent_v_list

    # todo: make objects a class (e.g. in common.py) so state is stored internally and negation operator is specific to the object
    def _negate_entity_state(self, state, obj):
        current_state = self._extract_entity_state(state, obj)
        # assumed boolean, two variable state
        if obj == 'door:':
            door_states = copy.deepcopy(self.latent_states)
            door_states.remove(current_state)
            next_state = door_states[0]
        else:
            lock_states = copy.deepcopy(self.observable_states)
            lock_states.remove(current_state)
            next_state = lock_states[0]

        return next_state

    @staticmethod
    def _extract_entity_state(state, obj):
        obj_start_idx = state.find(obj)
        # extract object name + state
        obj_str = state[obj_start_idx:state.find(',', obj_start_idx) + 1]
        # extract state up to next ',', inlcuding the ','
        obj_state = obj_str[obj_str.find(':') + 1:obj_str.find(',') + 1]
        return obj_state

    # changes obj's state in state (full state) to next_obj_state
    @staticmethod
    def _change_entity_state(state, entity, next_obj_state):
        tokens = state.split(',')
        tokens.pop(len(tokens)-1) # remove empty string at end of array
        for i in range(len(tokens)):
            token = tokens[i]
            token_lock = token[:token.find(':')+1]
            # update this token's state
            if token_lock == entity:
                tokens[i] = entity + next_obj_state
            else:
                tokens[i] += ',' # next_obj_state should contain ',', but split removes ',' from all others
        new_state = ''.join(tokens)
        return new_state

    # update fixtures, joints, in Box2D environment
    # def update_env(self, env):

# class MultiDoorLockMDP(StateMachineMDP):
#
#     lock_prob = 0.2
#     door_prob = 0.8
#
#     def __init__(self):
#         self.fsm = MultiDoorLockFSM()
#         self.fsm.get_graph().draw('my_state_diagram.png', prog='dot')
#         exit()
#
#         super(MultiDoorLockMDP, self).__init__()
#
#         # add unlock/lock transition for every lock
#         for lock in MultiDoorLockFSM.locks:
#             locked = [s for s in self.fsm.states if lock + '+' in s]
#             unlocked = [s for s in self.fsm.states if lock + '-' in s]
#             for locked_state, unlocked_state in zip(locked, unlocked):
#                 self.add_transition('lock_{}'.format(lock), unlocked_state, locked_state, MultiDoorLockMDP.lock_prob)
#                 self.add_transition('lock_{}'.format(lock), unlocked_state, unlocked_state, 1 - MultiDoorLockMDP.lock_prob)
#
#                 self.add_transition('unlock_{}'.format(lock), locked_state, unlocked_state, MultiDoorLockMDP.lock_prob)
#                 self.add_transition('unlock_{}'.format(lock), locked_state, locked_state, 1 - MultiDoorLockMDP.lock_prob)
#
#         # add nothing transition
#         for state in self.fsm.states:
#             self.add_transition('nothing', state, state, 1)
#
#         # add door open/close transition
#             self.add_transition('open', 'l0-l1-l2-o-', 'l0-l1-l2-o+', MultiDoorLockMDP.door_prob)
#             self.add_transition('open', 'l0-l1-l2-o-', 'l0-l1-l2-o-', 1 - MultiDoorLockMDP.door_prob)
#
#             self.add_transition('close', 'l0-l1-l2-o+', 'l0-l1-l2-o-', MultiDoorLockMDP.door_prob)
#             self.add_transition('close', 'l0-l1-l2-o+', 'l0-l1-l2-o+', 1 - MultiDoorLockMDP.door_prob)
#
#         self.add_reward('l0-l1-l2-o+', 10)
#
#         self.alg = mdptoolbox.mdp.PolicyIteration(self.transitions,
#                                                   self.rewards,
#                                                   0.9,
#                                                   skip_check=True)

if __name__ == '__main__':
    simple = MultiDoorLockFSM()
    simple.observable_machine.get_graph().draw('observable_diagram.png', prog='dot')
    simple.latent_machine.get_graph().draw('latent_diagram.png', prog='dot')
