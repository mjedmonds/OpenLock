from openlock.finite_state_machine import FiniteStateMachineManager
from openlock.scenarios.scenario import Scenario
from openlock.logger_env import ActionLog


class CommonCause4Scenario(Scenario):

    name = "CC4"

    observable_states = [
        "pulled,",
        "pushed,",
    ]  # '+' -> locked/pulled, '-' -> unlocked/pushed
    # todo: make names between obj_map in env consistent with names in FSM (extra ':' in FSM)
    observable_vars = ["l0:", "l1:", "l2:", "l3:"]
    observable_initial_state = "l0:pulled,l1:pulled,l2:pulled,l3:pulled,"

    latent_states = ["unlocked,", "locked,"]  # '+' -> open, '-' -> closed
    latent_vars = ["door:"]
    latent_initial_state = "door:locked,"

    actions = (
        ["nothing"]
        + ["pull_{}".format(lock) for lock in observable_vars]
        + ["push_{}".format(lock) for lock in observable_vars]
    )

    # lists of actions that represent solution sequences
    solutions = [
        [
            ActionLog("push_l0", None),
            ActionLog("push_l1", None),
            ActionLog("push_door", None),
        ],
        [
            ActionLog("push_l0", None),
            ActionLog("push_l2", None),
            ActionLog("push_door", None),
        ],
        [
            ActionLog("push_l0", None),
            ActionLog("push_l3", None),
            ActionLog("push_door", None),
        ],
    ]

    def __init__(self, use_physics=True):
        super(CommonCause4Scenario, self).__init__(use_physics=use_physics)

        self.fsmm = FiniteStateMachineManager(
            scenario=self,
            o_states=self.observable_states,
            o_vars=self.observable_vars,
            o_initial=self.observable_initial_state,
            l_states=self.latent_states,
            l_vars=self.latent_vars,
            l_initial=self.latent_initial_state,
            actions=self.actions,
        )

        self.lever_configs = None
        self.lever_opt_params = None

        # define observable states that trigger changes in the latent space;
        # this is the clue between the two machines.
        # Here we assume if the observable case is in any criteria than those listed, the door is locked
        self.door_unlock_criteria = [
            s
            for s in self.fsmm.observable_fsm.state_permutations
            if "l1:pushed," in s or "l2:pushed," in s or "l3:pushed," in s
        ]

        # add unlock/lock transition for every lock
        for lock in self.fsmm.observable_fsm.vars:
            if lock == "l1:":
                pulled = [
                    s
                    for s in self.fsmm.observable_fsm.state_permutations
                    if lock + "pulled," in s and "l0:pushed," in s
                ]
                pushed = [
                    s
                    for s in self.fsmm.observable_fsm.state_permutations
                    if lock + "pushed," in s and "l0:pushed," in s
                ]
                super(CommonCause4Scenario, self).add_no_ops(lock, pushed, pulled)
            elif lock == "l2:":
                pulled = [
                    s
                    for s in self.fsmm.observable_fsm.state_permutations
                    if lock + "pulled," in s and "l0:pushed," in s
                ]
                pushed = [
                    s
                    for s in self.fsmm.observable_fsm.state_permutations
                    if lock + "pushed," in s and "l0:pushed," in s
                ]
                super(CommonCause4Scenario, self).add_no_ops(lock, pushed, pulled)
            elif lock == "l3:":
                pulled = [
                    s
                    for s in self.fsmm.observable_fsm.state_permutations
                    if lock + "pulled," in s and "l0:pushed," in s
                ]
                pushed = [
                    s
                    for s in self.fsmm.observable_fsm.state_permutations
                    if lock + "pushed," in s and "l0:pushed," in s
                ]
                super(CommonCause4Scenario, self).add_no_ops(lock, pushed, pulled)
            else:
                pulled = [
                    s
                    for s in self.fsmm.observable_fsm.state_permutations
                    if lock + "pulled," in s
                ]
                pushed = [
                    s
                    for s in self.fsmm.observable_fsm.state_permutations
                    if lock + "pushed," in s
                ]
            for pulled_state, pushed_state in zip(pulled, pushed):
                # these transitions need to change the latent FSM, so we update the manager after executing them
                self.fsmm.observable_fsm.machine.add_transition(
                    "pull_{}".format(lock),
                    pushed_state,
                    pulled_state,
                    after="update_manager",
                )
                self.fsmm.observable_fsm.machine.add_transition(
                    "push_{}".format(lock),
                    pulled_state,
                    pushed_state,
                    after="update_manager",
                )

        super(CommonCause4Scenario, self).add_nothing_transition()

        super(CommonCause4Scenario, self).add_door_transitions()

    def update_latent(self):
        """
        logic to transition in the latent state space based on the observable state space, if needed
        """
        super(CommonCause4Scenario, self).update_latent()

    def update_observable(self):
        """
        updates observable fsm based on some change in the observable fsm, if needed
        """
        super(CommonCause4Scenario, self).update_observable()

    def update_state_machine(self, action=None):
        """
        Updates the finite state machines according to object status in the Box2D environment
        """
        super(CommonCause4Scenario, self).update_state_machine(action)

    def init_scenario_env(self, world_def=None, effect_probabilities=None):
        """
        initializes the scenario-specific components of the box2d world (e.g. levers)
        :return:
        """

        super(CommonCause4Scenario, self).init_scenario_env(world_def, effect_probabilities=effect_probabilities)

        if self.use_physics:
            self.obj_map["l1"].lock()  # initially lock l1
            self.obj_map["l2"].lock()  # initially lock l2
            self.obj_map["l3"].lock()  # initially lock l3

    def _update_env(self):
        """
        updates the Box2D environment based on the state of the finite state machine
        """
        super(CommonCause4Scenario, self)._update_env()

    def _update_latent_objs(self):
        """
        updates latent objects in the Box2D environment based on state of the latent finite state machine
        """
        super(CommonCause4Scenario, self)._update_latent_objs()

    def _update_observable_objs(self):
        """
        updates observable objects in the Box2D environment based on the observable state of the finite state machine
        """
        observable_states = self.fsmm.get_observable_states()
        for observable_var in list(observable_states.keys()):
            # ---------------------------------------------------------------
            # add code to change part of the environment based on the state of an observable variable here
            # ---------------------------------------------------------------
            if observable_var == "l1:":
                # l1 unlocks if l0 is pushed
                if "l0:pushed," in self.fsmm.observable_fsm.state:
                    self.obj_map["l1"].unlock()
                else:
                    self.obj_map["l1"].lock()
            if observable_var == "l2:":
                # l2 unlocks if l0 is pushed
                if "l0:pushed," in self.fsmm.observable_fsm.state:
                    self.obj_map["l2"].unlock()
                else:
                    self.obj_map["l2"].lock()
            if observable_var == "l3:":
                # l3 unlocks if l0 is pushed
                if "l0:pushed," in self.fsmm.observable_fsm.state:
                    self.obj_map["l3"].unlock()
                else:
                    self.obj_map["l3"].lock()

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
