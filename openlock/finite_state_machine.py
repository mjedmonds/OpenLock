import re
import numpy as np

# from transitions.extensions import GraphMachine as Machine
from transitions import Machine


def cartesian_product(*lists):
    result = [[]]
    for list in lists:
        result = [x + [y] for x in result for y in list]
    return ["".join(elem) for elem in result]


class FiniteStateMachine:
    def __init__(self, fsm_manager, name, vars, states, initial_state):
        self.fsm_manager = fsm_manager
        self.name = name
        self.vars = vars
        self.state_permutations = self._permutate_states(states)
        self.initial_state = initial_state

        self.machine = Machine(
            model=self,
            states=self.state_permutations,
            initial=self.initial_state,
            ignore_invalid_triggers=True,
            auto_transitions=False,
        )

    def _permutate_states(self, states):
        assert len(self.vars) > 0

        v_list = cartesian_product([self.vars[0]], states)
        for i in range(1, len(self.vars)):
            v_list = cartesian_product(
                v_list, cartesian_product([self.vars[i]], states)
            )

        # return cartesian_product(observable_v_list, door_list)
        return v_list

    def reset(self):
        self.machine.set_state(self.initial_state)

    def update_manager(self):
        """
        tells FSM manager to update the other FSM (latent/observable) based on the changes this FSM (obserable/latent) made
        :return:
        """
        if self.name == "observable":
            self.fsm_manager.update_latent()
        else:
            self.fsm_manager.update_observable()


class FiniteStateMachineManager:
    """
    Manages the observable and latent finite state machines
    """

    def __init__(
        self,
        scenario,
        o_states,
        o_vars,
        o_initial,
        l_states,
        l_vars,
        l_initial,
        actions,
    ):
        self.scenario = scenario
        self.observable_states = o_states
        self.observable_vars = o_vars
        self.observable_initial_state = o_initial

        self.latent_states = l_states
        self.latent_vars = l_vars
        self.latent_initial_state = l_initial

        self.actions = actions

        self.observable_fsm = FiniteStateMachine(
            fsm_manager=self,
            name="observable",
            vars=self.observable_vars,
            states=self.observable_states,
            initial_state=self.observable_initial_state,
        )

        self.latent_fsm = FiniteStateMachine(
            fsm_manager=self,
            name="latent",
            vars=self.latent_vars,
            states=self.latent_states,
            initial_state=self.latent_initial_state,
        )

    def reset(self):
        """
        resets both the observable fsm and latent fsm
        :return:
        """
        self.observable_fsm.reset()
        self.latent_fsm.reset()

    def get_latent_states(self):
        """
        extracts latent variables and their state into a dictonary. key: variable. value: variable state
        :return: dictionary of variables to their corresponding variable state
        """
        latent_states = dict()
        for latent_var in self.latent_vars:
            latent_states[latent_var] = self.extract_entity_state(
                self.latent_fsm.state, latent_var
            )
        return latent_states

        # parses out the state of a specified object from a full state string

    def get_observable_states(self):
        """
        extracts observable variables and their state into a dictonary. key: variable. value: variable state
        :return: dictionary of variables to their corresponding variable state
        """
        observable_states = dict()
        for observable_var in self.observable_vars:
            observable_states[observable_var] = self.extract_entity_state(
                self.observable_fsm.state, observable_var
            )
        return observable_states

    def get_internal_state(self):
        return self.observable_fsm.state + self.latent_fsm.state

    def update_latent(self):
        """
        updates the latent state space according to the scenario
        :return:
        """
        self.scenario.update_latent()

    def update_observable(self):
        """
        updates the observable state space according to the scenario
        :return:
        """
        self.scenario.update_observable()

    def execute_action(self, action):
        if action in self.actions:
            # changes in observable FSM will trigger a callback to update the latent FSM if needed
            self.observable_fsm.trigger(action)
        else:
            # todo: dirty hack to get door pushing action
            if action == "push_door:":
                self.scenario.push_door()
            else:
                raise ValueError("unknown action '{}".format(action) + "'")

    @staticmethod
    def extract_entity_state(state, obj):
        obj_start_idx = state.find(obj)
        # extract object name + state
        obj_str = state[obj_start_idx : state.find(",", obj_start_idx) + 1]
        # extract state up to next ',', inlcuding the ','
        obj_state = obj_str[obj_str.find(":") + 1 : obj_str.find(",") + 1]
        return obj_state

    # changes obj's state in state (full state) to next_obj_state
    @staticmethod
    def change_entity_state(state, entity, next_obj_state):
        tokens = state.split(",")
        tokens.pop(len(tokens) - 1)  # remove empty string at end of array
        for i in range(len(tokens)):
            token = tokens[i]
            token_lock = token[: token.find(":") + 1]
            # update this token's state
            if token_lock == entity:
                tokens[i] = entity + next_obj_state
            else:
                tokens[
                    i
                ] += (
                    ","
                )  # next_obj_state should contain ',', but split removes ',' from all others
        new_state = "".join(tokens)
        return new_state
