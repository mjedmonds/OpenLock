from openlock.finite_state_machine import FiniteStateMachineManager
from openlock.scenarios.scenario import Scenario
from openlock.logger_env import ActionLog


class CommonCause3Scenario(Scenario):

    name = "CC3"

    observable_states = [
        "pulled,",
        "pushed,",
    ]  # '+' -> locked/pulled, '-' -> unlocked/pushed
    # todo: make names between obj_map in env consistent with names in FSM (extra ':' in FSM)
    observable_vars = ["l0:", "l1:", "l2:"]
    observable_initial_state = "l0:pulled,l1:pulled,l2:pulled,"

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
    ]

    def __init__(self, use_physics=True):
        super(CommonCause3Scenario, self).__init__(use_physics=use_physics)

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
            if "l1:pushed," in s or "l2:pushed," in s
        ]

        # add unlock/lock transition for every lock
        for lock in self.fsmm.observable_fsm.vars:
            if lock == "l2:":
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
                super(CommonCause3Scenario, self).add_no_ops(lock, pushed, pulled)
            elif lock == "l1:":
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
                super(CommonCause3Scenario, self).add_no_ops(lock, pushed, pulled)
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

        super(CommonCause3Scenario, self).add_nothing_transition()

        super(CommonCause3Scenario, self).add_door_transitions()

    def update_latent(self):
        """
        logic to transition in the latent state space based on the observable state space, if needed
        """
        super(CommonCause3Scenario, self).update_latent()

    def update_observable(self):
        """
        updates observable fsm based on some change in the observable fsm, if needed
        """
        super(CommonCause3Scenario, self).update_observable()

    def update_state_machine(self, action=None):
        """
        Updates the finite state machines according to object status in the Box2D environment
        """
        super(CommonCause3Scenario, self).update_state_machine(action)

    def init_scenario_env(self, world_def=None, effect_probabilities=None):
        """
        initializes the scenario-specific components of the box2d world (e.g. levers)
        :return:
        """

        super(CommonCause3Scenario, self).init_scenario_env(world_def, effect_probabilities=effect_probabilities)

        if self.use_physics:
            self.obj_map["l2"].lock()  # initially lock l2
            self.obj_map["l1"].lock()  # initially lock l1

    def _update_env(self):
        """
        updates the Box2D environment based on the state of the finite state machine
        """
        super(CommonCause3Scenario, self)._update_env()

    def _update_latent_objs(self):
        """
        updates latent objects in the Box2D environment based on state of the latent finite state machine
        """
        super(CommonCause3Scenario, self)._update_latent_objs()

    def _update_observable_objs(self):
        """
        updates observable objects in the Box2D environment based on the observable state of the finite state machine
        """
        observable_states = self.fsmm.get_observable_states()
        for observable_var in list(observable_states.keys()):
            # ---------------------------------------------------------------
            # add code to change part of the environment based on the state of an observable variable here
            # ---------------------------------------------------------------
            if observable_var == "l2:":
                # l2 unlocks if l0 is pushed
                if "l0:pushed," in self.fsmm.observable_fsm.state:
                    self.obj_map["l2"].unlock()
                else:
                    self.obj_map["l2"].lock()
            if observable_var == "l1:":
                # l1 unlocks if l0 is pushed
                if "l0:pushed," in self.fsmm.observable_fsm.state:
                    self.obj_map["l1"].unlock()
                else:
                    self.obj_map["l1"].lock()
