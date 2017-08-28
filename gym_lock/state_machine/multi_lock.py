from transitions.extensions import GraphMachine as Machine
from gym_lock.state_machine.mdp import StateMachineMDP

import mdptoolbox
import itertools


def cartesian_product(*lists):
    result = [[]]
    for list in lists:
        result = [x + [y] for x in result for y in list]
    return [''.join(elem) for elem in result]

class MultiDoorLockFSM(object):

    lock_states = ['+', '-']
    locks = ['l0', 'l1', 'l2']
    door_states = ['+', '-']
    doors = ['o']
    states = []
    actions = ['nothing', 'open', 'close'] \
              + ['lock_{}'.format(lock) for lock in locks] \
              + ['unlock_{}'.format(lock) for lock in locks]

    def __init__(self):
        self.states = self._init_states()

        self.machine = Machine(model=self,
                               states=self.states,
                               initial='l0+l1+l2+o-',
                               ignore_invalid_triggers=True,
                               auto_transitions=False)

        # add unlock/lock transition for every lock
        for lock in MultiDoorLockFSM.locks:
            if lock == 'l2':
                locked = [s for s in self.states if lock + '+' in s and 'l0-' in s and 'l1-' in s]
                unlocked = [s for s in self.states if lock + '-' in s and 'l0-' in s and 'l1-' in s]
            else:
                locked = [s for s in self.states if lock + '+' in s]
                unlocked = [s for s in self.states if lock + '-' in s]
            for locked_state, unlocked_state in zip(locked, unlocked):
                    self.machine.add_transition('lock_{}'.format(lock), unlocked_state, locked_state)
                    self.machine.add_transition('unlock_{}'.format(lock), locked_state, unlocked_state)

        # add nothing transition
        for state in self.states:
            self.machine.add_transition('nothing', state, state)

        # add door open/close transition
        self.machine.add_transition('open', 'l0-l1-l2-o-', 'l0-l1-l2-o+')
        self.machine.add_transition('close', 'l0-l1-l2-o+', 'l0-l1-l2-o-')



    @property
    def actions(self):
        return self.machine.get_triggers(self.state)


    @classmethod
    def _init_states(cls):
        assert len(cls.locks) > 0

        lock_list = cartesian_product([cls.locks[0]], cls.lock_states)
        for i in range(1, len(cls.locks)):
            lock_list = cartesian_product(lock_list, cartesian_product([cls.locks[i]], cls.lock_states))

        door_list = cartesian_product([cls.doors[0]], cls.door_states)
        for i in range(1, len(cls.doors)):
            door_list = cartesian_product(door_list, cartesian_product([cls.doors[i]], cls.door_states))



        return cartesian_product(lock_list, door_list)


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
    simple.get_graph().draw('my_state_diagram.png', prog='dot')
