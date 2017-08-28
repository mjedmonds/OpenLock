import gym
import numpy as np
from gym_lock.envs import ArmLockEnv
from gym_lock.common import Action

def create_state_entry(state, i):
    entry = [0] * len(col_label)
    entry[0] = i
    for name, val in state['OBJ_STATES'].items():
        entry[index_map[name]] = int(val)
    return entry

env = gym.make('arm_lock-v0')

action_script = [Action('push_perp', ('l2', 4)),    # try to unlock l2, but it doesn't budge!
                 Action('pull_perp', ('l2', 4)),    # try pulling l2 instead, still won't budge
                 Action('push_perp', ('l0', 4)),    # unlock l0
                 Action('push_perp', ('l1', 4)),    # unlock l1
                 Action('push_perp', ('l2', 4)),    # try to unlock l2, but it still doesn't budge!
                 Action('pull_perp', ('l2', 4)),    # try pulling l2 instead, it works
                 Action('push_perp', ('door', 4)),  # open the door
                 Action('pull_perp', ('l1', 4)),    # lock l1 (door locks too!)
                 Action('push_perp', ('l2', 4)),    # try to move l2 again
                 Action('pull_perp', ('l2', 4)),    # and it now doesn't work because l1 is locked!
                 Action('push_perp', ('l1', 4)),    # so let's re-unlock l1 (re-unlocks door!)
                 Action('pull_perp', ('door', 1)),  # close the door
                 Action('pull_perp', ('door', 1)),
                 Action('pull_perp', ('door', 1)),
                 Action('push_perp', ('l2', 4)),    # then re-lock l2
                 Action('pull_perp', ('l1', 4)),    # re-lock l1
                 Action('pull_perp', ('l0', 4))]    # re-lock l0


# setup .csv headers
col_label = []
col_label.append('frame')
for col_name in env.world_def.get_state()['OBJ_STATES']:
    col_label.append(col_name)
col_label.append('agent')
for col_name in env.action_space:
    col_label.append(col_name)

index_map = {name : idx for idx, name in enumerate(col_label)}
results = [col_label]

i = 0
while (True):
    # if env.viewer.desired_config:
    #     # TODO: messy interface for stepping, add adapter or simplyify
    #     # TODO: add ability to move base
    #     obs, rew, done, info = env.step(Action('goto', env.viewer.desired_config))
    #     # TODO: abstraction
    #     env.viewer.desired_config = None
    #     #print obs['FSM_STATE']
    # else:
    #     env.step(False)

    # append initial observation
    results.append(create_state_entry(env.reset(), i))
    print env.reset()['OBJ_STATES']
    print env.reset()['_FSM_STATE']

    for action in action_script:

        i += 1

        # create pre-observation entry
        entry = [0] * len(col_label)
        entry[0] = i
        # copy over previous state
        entry[1:index_map['agent']] = results[-1][1:index_map['agent']]

        # mark action idx
        if type(action.params[0]) is str:
            col = '{}_{}'.format(action.name, action.params[0])
        else:
            col = action.name

        entry[index_map[col]] = 1

        # append pre-observation entry
        results.append(entry)

        # take action
        obs, rew, done, info = env.step(action)
        # import time
        # print obs['OBJ_STATES']
        # print obs['_FSM_STATE']
        # time.sleep(5)

        # append post-observation entry to results list
        i += 1
        results.append(create_state_entry(obs, i))

    np.savetxt('results.csv', results, delimiter=',', fmt='%s')
    exit()


        # # fluent_statuses = [1 if obj == True for obj
        # env.step(Action('move_end_frame', (-5, 0, 0)))
        # env.step(Action('rest', None))

    # env.step(Action('rest', None))
    # for i in range(0, 100000):
    #     env.step(False)
