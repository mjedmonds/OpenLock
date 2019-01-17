import numpy as np


class StateMachineMDP(object):
    def __init__(self):
        self.transitions = np.zeros(
            (len(self.fsm.actions), len(self.fsm.states), len(self.fsm.states))
        )  # A x S x S
        self.rewards = np.zeros(len(self.fsm.states))

        self.state_map = dict()
        for i in range(0, len(self.fsm.states)):
            self.state_map[self.fsm.states[i]] = i

        self.action_map = dict()
        for i in range(0, len(self.fsm.actions)):
            self.action_map[self.fsm.actions[i]] = i

    def add_transition(self, action, source, dest, prob):
        self.transitions[self.action_map[action]][self.state_map[source]][
            self.state_map[dest]
        ] = prob

    def add_reward(self, state, reward):
        self.rewards[self.state_map[state]] = reward  # want door open and unlocked

    def run(self):
        self.alg.run()
        print(len(self.alg.policy))
        for i in range(0, len(self.alg.policy)):
            print(
                "in state {} take action {}".format(
                    self.fsm.states[i], self.fsm.actions[self.alg.policy[i]]
                )
            )
