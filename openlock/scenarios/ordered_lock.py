import mdptoolbox
from transitions.extensions import GraphMachine as Machine

from openlock.scenarios.mdp import StateMachineMDP


def cartesian_product(*lists):
    result = [[]]
    for list in lists:
        result = [x + [y] for x in result for y in list]
    return ["".join(elem) for elem in result]


# class OrderedDoorLockFSM(object):
#
#     def __init__(self):
#
#         order_fsm = OrderedFSM()
#
#         callbacks = {'go' : self.order_fsm.go(),
#                      'back' : self.order_fsm.back(),
#                      'trash' : self.order_fsm.trash(),
#                      'reset' : self.order_fsm.reset(),}
#
#         multi_lock_fsm = MultiDoorLockFSM(callbacks)
#
#
# class OrderedFSM(object):
#
#     def __init__(self):
#         # add state machine to track order
#         self.order_state_machine = Machine(model=self,
#                                            states=['none'] + MultiDoorLockFSM.locks + ['opened!', 'trash'],
#                                            initial='none')
#
#         for i in range(0, self.order_state_machine.states.index('opened!')):
#             self.order_state_machine.add_transition('go', MultiDoorLockFSM.locks[i], MultiDoorLockFSM.locks[i + 1])
#             self.order_state_machine.add_transition('back', MultiDoorLockFSM.locks[i + 1], MultiDoorLockFSM.locks[i])
#
#         self.order_state_machine.add_transition('go', MultiDoorLockFSM.locks[-1], 'opened!')
#         self.order_state_machine.add_transition('trash', '*', 'trash')
#         self.order_state_machine.add_transition('reset', '*', 'none')


class MultiDoorLockFSM(object):

    lock_states = ["+", "-"]
    locks = ["l1", "l2", "l3"]
    door_states = ["+", "-"]
    doors = ["o"]
    states = []
    actions = (
        ["nothing", "open", "close"]
        + ["lock_{}".format(lock) for lock in locks]
        + ["unlock_{}".format(lock) for lock in locks]
    )

    ordered_path = [
        "l1-l2+l3+o-_ord",
        "l1-l2-l3+o-_ord",
        "l1-l2-l3-o-_ord",
        "l1-l2-l3-o+_ord",
    ]

    def __init__(self):
        self.states = self._init_states()
        self.initial = "l1+l2+l3+o-"
        # captures lock/door status
        self.machine = Machine(model=self, states=self.states, initial=self.initial)

        # add unlock/lock transition for every lock
        for lock in MultiDoorLockFSM.locks:
            locked = [s for s in self.states if lock + "+" in s and not "_ord" in s]
            unlocked = [s for s in self.states if lock + "-" in s and not "_ord" in s]
            for locked_state, unlocked_state in zip(locked, unlocked):

                # get onto ordered path
                if locked_state == self.initial and lock == "l1":
                    # all locked, door closed, opening l1 gets us onto ordered path
                    unlocked_state = MultiDoorLockFSM.ordered_path[0]

                self.machine.add_transition(
                    "lock_{}".format(lock), unlocked_state, locked_state
                )
                self.machine.add_transition(
                    "unlock_{}".format(lock), locked_state, unlocked_state
                )

        # add nothing transition
        for state in self.states:
            self.machine.add_transition("nothing", state, state)

        # add door open/close transition
        self.machine.add_transition(
            "open", MultiDoorLockFSM.ordered_path[-2], MultiDoorLockFSM.ordered_path[-1]
        )
        self.machine.add_transition(
            "close",
            MultiDoorLockFSM.ordered_path[-1],
            MultiDoorLockFSM.ordered_path[-2],
        )

        # add ordered path transitions
        for i in range(0, len(MultiDoorLockFSM.ordered_path) - 2):
            self.machine.add_transition(
                "unlock_{}".format(MultiDoorLockFSM.locks[i + 1]),
                MultiDoorLockFSM.ordered_path[i],
                MultiDoorLockFSM.ordered_path[i + 1],
            )
            self.machine.add_transition(
                "lock_{}".format(MultiDoorLockFSM.locks[i + 1]),
                MultiDoorLockFSM.ordered_path[i + 1],
                MultiDoorLockFSM.ordered_path[i],
            )

        # add transitions off of ordered path
        self.machine.add_transition(
            "unlock_l3", MultiDoorLockFSM.ordered_path[0], "l1-l2+l3-o-"
        )

        # bug fix...what a mess!
        self.machine.add_transition("lock_l1", "l1-l2+l3+o-", self.initial)

    @classmethod
    def _init_states(cls):
        assert len(cls.locks) > 0

        lock_list = cartesian_product([cls.locks[0]], cls.lock_states)
        for i in range(1, len(cls.locks)):
            lock_list = cartesian_product(
                lock_list, cartesian_product([cls.locks[i]], cls.lock_states)
            )

        door_list = cartesian_product([cls.doors[0]], cls.door_states)
        for i in range(1, len(cls.doors)):
            door_list = cartesian_product(
                door_list, cartesian_product([cls.doors[i]], cls.door_states)
            )

        lock_states = cartesian_product(lock_list, door_list)

        return cls.ordered_path + lock_states


class MultiDoorLockMDP(StateMachineMDP):

    lock_prob = 0.2
    door_prob = 0.8

    def __init__(self):
        self.fsm = MultiDoorLockFSM()
        self.fsm.get_graph().draw("my_state_diagram.png", prog="dot")

        super(MultiDoorLockMDP, self).__init__()

        # add unlock/lock transition for every lock
        for lock in MultiDoorLockFSM.locks:
            locked = [s for s in self.fsm.states if lock + "+" in s and not "_ord" in s]
            unlocked = [
                s for s in self.fsm.states if lock + "-" in s and not "_ord" in s
            ]
            for locked_state, unlocked_state in zip(locked, unlocked):

                # get onto ordered path
                if locked_state == self.fsm.initial and lock == "l1":
                    # all locked, door closed, opening l1 gets us onto ordered path
                    unlocked_state = MultiDoorLockFSM.ordered_path[0]

                self.add_transition(
                    "lock_{}".format(lock), unlocked_state, locked_state, 1
                )
                self.add_transition(
                    "unlock_{}".format(lock), locked_state, unlocked_state, 1
                )

        # add nothing transition
        for state in self.fsm.states:
            self.add_transition("nothing", state, state, 1)

        # add door open/close transition
        self.add_transition(
            "open",
            MultiDoorLockFSM.ordered_path[-2],
            MultiDoorLockFSM.ordered_path[-1],
            1,
        )
        self.add_transition(
            "close",
            MultiDoorLockFSM.ordered_path[-1],
            MultiDoorLockFSM.ordered_path[-2],
            1,
        )

        # add ordered path transitions
        for i in range(0, len(MultiDoorLockFSM.ordered_path) - 2):
            self.add_transition(
                "unlock_{}".format(MultiDoorLockFSM.locks[i + 1]),
                MultiDoorLockFSM.ordered_path[i],
                MultiDoorLockFSM.ordered_path[i + 1],
                1,
            )
            self.add_transition(
                "lock_{}".format(MultiDoorLockFSM.locks[i + 1]),
                MultiDoorLockFSM.ordered_path[i + 1],
                MultiDoorLockFSM.ordered_path[i],
                1,
            )

        # add transitions off of ordered path
        self.add_transition(
            "unlock_l3", MultiDoorLockFSM.ordered_path[0], "l1-l2+l3-o-", 1
        )

        # bug fix...what a mess!
        self.add_transition("lock_l1", "l1-l2+l3+o-", self.fsm.initial, 1)

        self.add_reward("l1-l2-l3-o+_ord", 10)

        self.alg = mdptoolbox.mdp.PolicyIteration(
            self.transitions, self.rewards, 0.9, skip_check=True
        )


if __name__ == "__main__":
    simple = MultiDoorLockMDP()
    simple.run()
