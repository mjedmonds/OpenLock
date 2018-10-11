import mdptoolbox
from transitions import Machine

from openlock.scenarios.mdp import StateMachineMDP


class SimpleDoorLockFSM(object):

    states = ["l+o+", "l-o+", "l+o-", "l-o-"]
    actions = ["nothing", "lock", "unlock", "open", "close"]

    def __init__(self):
        self.machine = Machine(
            model=self, states=SimpleDoorLockFSM.states, initial="l+o-"
        )

        self.machine.add_transition("unlock", source="l+o-", dest="l-o-")
        self.machine.add_transition("unlock", source="l+o+", dest="l-o+")

        self.machine.add_transition("lock", source="l-o-", dest="l+o-")
        self.machine.add_transition("lock", source="l-o+", dest="l+o+")

        self.machine.add_transition("open", source="l-o-", dest="l-o+")

        self.machine.add_transition("close", source="l-o+", dest="l-o-")


class SimpleDoorLockMDP(StateMachineMDP):
    def __init__(self):
        self.fsm = SimpleDoorLockFSM()
        super(SimpleDoorLockMDP, self).__init__()

        self.add_transition("unlock", "l+o-", "l-o-", 1)
        self.add_transition("unlock", "l+o+", "l-o+", 1)
        self.add_transition("lock", "l-o-", "l+o-", 1)
        self.add_transition("lock", "l-o+", "l+o+", 1)
        self.add_transition("open", "l-o-", "l-o+", 1)
        self.add_transition("close", "l-o+", "l-o-", 1)

        for state in self.fsm.states:
            self.add_transition("nothing", state, state, 1)

        self.add_reward("l-o+", 10)

        print(self.transitions)
        print(self.rewards)
        self.alg = mdptoolbox.mdp.PolicyIteration(
            self.transitions, self.rewards, 0.9, skip_check=True
        )


if __name__ == "__main__":
    simple = SimpleDoorLockMDP()
    simple.run()
