from openlock.finite_state_machine import FiniteStateMachineManager
from openlock.scenarios.scenario import Scenario
from openlock.logger_env import ActionLog


class TwoStepTestingScenario(Scenario):

    name = "TwoStepTestingScenario"

    observable_states = [
        "pulled,",
        "pushed,",
    ]  # '+' -> locked/pulled, '-' -> unlocked/pushed
    # todo: make names between obj_map in env consistent with names in FSM (extra ':' in FSM)
    observable_vars = ["l0:", "l1:"]
    observable_initial_state = "l0:pulled,l1:pulled,"

    latent_states = ["unlocked,", "locked,"]  # '+' -> open, '-' -> closed
    latent_vars = ["door:"]
    latent_initial_state = "door:locked,"

    actions = (
        ["nothing"]
        + ["pull_{}".format(lock) for lock in observable_vars]
        + ["push_{}".format(lock) for lock in observable_vars]
    )

    # lists of actions that represent solution sequences
    solutions = [[ActionLog("push_l0", None), ActionLog("push_door", None)]]

    def __init__(self, use_physics=True):
        super(TwoStepTestingScenario, self).__init__(use_physics=use_physics)

        self.world_def = None  # handle to the Box2D world

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

        # define observable states that trigger changes in the latent space;
        # this is the clue between the two machines.
        # Here we assume if the observable case is in any criteria than those listed, the door is locked
        self.door_unlock_criteria = [
            s for s in self.fsmm.observable_fsm.state_permutations if "l0:pushed," in s
        ]

        # add unlock/lock transition for every lock
        for lock in self.fsmm.observable_fsm.vars:
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

        super(TwoStepTestingScenario, self).add_nothing_transition()

        super(TwoStepTestingScenario, self).add_door_transitions()

    def update_latent(self):
        """
        logic to transition in the latent state space based on the observable state space, if needed
        """
        super(TwoStepTestingScenario, self).update_latent()

    def update_observable(self):
        """
        updates observable fsm based on some change in the observable fsm, if needed
        """
        super(TwoStepTestingScenario, self).update_observable()

    def update_state_machine(self, action=None):
        """
        Updates the finite state machines according to object status in the Box2D environment
        """
        super(TwoStepTestingScenario, self).update_state_machine(action)

    def init_scenario_env(self, world_def=None):
        """
        initializes the scenario-specific components of the box2d world (e.g. levers)
        :return:
        """

        super(TwoStepTestingScenario, self).init_scenario_env(world_def)

    def _update_env(self):
        """
        updates the Box2D environment based on the state of the finite state machine
        """
        super(TwoStepTestingScenario, self)._update_env()

    def _update_latent_objs(self):
        """
        updates latent objects in the Box2D environment based on state of the latent finite state machine
        """
        super(TwoStepTestingScenario, self)._update_latent_objs()

    def _update_observable_objs(self):
        """
        updates observable objects in the Box2D environment based on the observable state of the finite state machine
        """
        pass

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
