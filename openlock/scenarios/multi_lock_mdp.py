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
