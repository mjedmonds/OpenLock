
from gym_lock.common import *
from gym_lock.finite_state_machine import *


class CommonEffect3Scenario(object):

    name = 'CE3'

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

    lever_configs = [TwoDConfig(0, 15, 0), TwoDConfig(-15, 0, np.pi / 2), TwoDConfig(0, -15, -np.pi)]

    actions = ['nothing'] \
                   + ['pull_{}'.format(lock) for lock in observable_vars] \
                   + ['push_{}'.format(lock) for lock in observable_vars]

    def __init__(self):
        self.world_def = None # handle to the Box2D world

        self.fsmm = FiniteStateMachineManager(scenario=self,
                                              o_states=self.observable_states,
                                              o_vars=self.observable_vars,
                                              o_initial=self.observable_initial_state,
                                              l_states=self.latent_states,
                                              l_vars=self.latent_vars,
                                              l_initial=self.latent_initial_state,
                                              actions=self.actions)

        # add unlock/lock transition for every lock
        for lock in self.fsmm.observable_fsm.vars:
            if lock == 'l2:':
                pulled = [s for s in self.fsmm.observable_fsm.state_permutations if lock + 'pulled,' in s and 'l0:pushed,' in s and 'l1:pushed,' in s]
                pushed = [s for s in self.fsmm.observable_fsm.state_permutations if lock + 'pushed,' in s and 'l0:pushed,' in s and 'l1:pushed,' in s]
            else:
                pulled = [s for s in self.fsmm.observable_fsm.state_permutations if lock + 'pulled,' in s]
                pushed = [s for s in self.fsmm.observable_fsm.state_permutations if lock + 'pushed,' in s]
            for pulled_state, pushed_state in zip(pulled, pushed):
                # these transitions need to change the latent FSM, so we update the manager after executing them
                self.fsmm.observable_fsm.machine.add_transition('pull_{}'.format(lock), pushed_state, pulled_state, after='update_manager')
                self.fsmm.observable_fsm.machine.add_transition('push_{}'.format(lock), pulled_state, pushed_state, after='update_manager')

        # add nothing transition
        for state in self.fsmm.observable_fsm.state_permutations:
            self.fsmm.observable_fsm.machine.add_transition('nothing', state, state)
        for state in self.fsmm.latent_fsm.state_permutations:
            self.fsmm.latent_fsm.machine.add_transition('nothing', state, state)

        for door in self.latent_vars:
            self.fsmm.latent_fsm.machine.add_transition('lock_{}'.format(door), 'door:unlocked,', 'door:locked,')
            self.fsmm.latent_fsm.machine.add_transition('unlock_{}'.format(door), 'door:locked,', 'door:unlocked,')

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
            if 'button' not in name and 'door' not in name:
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
        '''
        initializes the scenario-specific components of the box2d world (e.g. levers)
        :return:
        '''

        # todo: come up with a better way to set self.world_def without passing as an argument here
        self.world_def = world_def

        opt_params = [None, None, {'lower_lim': 0.0, 'upper_lim': 2.0}]

        for i in range(0, len(self.lever_configs)):
            name = 'l{}'.format(i)
            lock = Lock(self.world_def, name, self.lever_configs[i], opt_params[i])
            self.world_def.obj_map[name] = lock

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
        updates observable objects in the Box2D environment based on the observable state of the finite state machine
        '''
        observable_states = self.fsmm.get_observable_states()
        for observable_var in observable_states.keys():
            # ---------------------------------------------------------------
            # add code to change part of the environment based on the state of an observable variable here
            # ---------------------------------------------------------------
            if observable_var == 'l2:':
                # unlock l2 based on status of l0, l1, part of multi-lock FSM
                if 'l0:pushed,' in self.fsmm.observable_fsm.state and 'l1:pushed,' in self.fsmm.observable_fsm.state:
                    self.world_def.unlock_lever('l2')
                else:
                    self.world_def.lock_lever('l2')

    # @property
    # def actions(self):
    #     return self.observable_fsm.machine.get_triggers(self.observable_fsm.state)

    # @classmethod
    # def _init_states(self):
    #     assert len(self.observable_vars) > 0
    #
    #     observable_v_list = cartesian_product([self.observable_vars[0]], self.observable_states)
    #     for i in range(1, len(self.observable_vars)):
    #         observable_v_list = cartesian_product(observable_v_list, cartesian_product([self.observable_vars[i]], self.observable_states))
    #
    #     latent_v_list = cartesian_product([self.latent_vars[0]], self.latent_states)
    #     for i in range(1, len(self.latent_vars)):
    #         door_list = cartesian_product(latent_v_list, cartesian_product([self.latent_vars[i]], self.latent_states))
    #
    #     # return cartesian_product(observable_v_list, door_list)
    #     return observable_v_list, latent_v_list

action_script = [Action('push_', ('l2', 4)),  # try to unlock l2, but it doesn't budge!
                 Action('pull_', ('l2', 4)),  # try pulling l2 instead, still won't budge
                 Action('push_', ('l0', 4)),  # unlock l0
                 Action('push_', ('l1', 4)),  # unlock l1
                 Action('push_', ('l2', 4)),  # try to unlock l2, but it still doesn't budge!
                 Action('pull_', ('l2', 4)),  # try pulling l2 instead, it works
                 Action('push_', ('door', 4)),  # open the door
                 Action('pull_', ('l1', 4)),  # lock l1 (door locks too!)
                 Action('push_', ('l2', 4)),  # try to move l2 again
                 Action('pull_', ('l2', 4)),  # and it now doesn't work because l1 is locked!
                 Action('push_', ('l1', 4)),  # so let's re-unlock l1 (re-unlocks door!)
                 Action('pull_', ('door', 1)),  # close the door
                 Action('pull_', ('door', 1)),
                 Action('pull_', ('door', 1)),
                 Action('push_', ('l2', 4)),  # then re-lock l2
                 Action('pull_', ('l1', 4)),  # re-lock l1
                 Action('pull_', ('l0', 4))]  # re-lock l0

