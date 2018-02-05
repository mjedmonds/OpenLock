import gym
import numpy as np
from gym_lock.envs import ArmLockEnv
from gym_lock.common import Action
import causality.causal_planner
import causality.common
import time

def create_state_entry(state, i, col_label, index_map):
    entry = [0] * len(col_label)
    entry[0] = i
    for name, val in state['OBJ_STATES'].items():
        entry[index_map[name]] = int(val)
    return entry

def main():

    env = gym.make('arm_lock-v0')

    # build causal planner and get the known and unknown action sequences
    data_dir = './scenario_outputs/action_reversal/'
    trial_name = 'ex1_extended'
    demonstration_file = data_dir + trial_name + '.csv'
    perceptual_file = data_dir + 'output_node_' + trial_name + '.mat'

    causal_planner = causality.causal_planner.load_trial(demonstration_file, perceptual_file)

    # setup .csv headers
    col_label = []
    col_label.append('frame')
    for col_name in env.world_def.get_internal_state()['OBJ_STATES']:
        col_label.append(col_name)
    col_label.append('agent')
    for col_name in env.action_space:
        col_label.append(col_name)

    index_map = {name : idx for idx, name in enumerate(col_label)}
    results = [col_label]

    i = 0

    # append initial observation
    results.append(create_state_entry(env.reset(), i, col_label, index_map))

    print 'Hello! Welcome to the game!'

    print '''See that door on your right? It is the vertical vertical on your right, with the
             red circle (the door hinge) and black circle (it's lock). That is your only escape.'''

    print '''To open it, you must manipulate the three locks (the rectangles above, below, and
             to your left). Their behavior is unknown! You'll know that you unlocked the door
             when the black circle goes away'''

    print 'ready...'
    print 'set...'
    print 'go!'

    env.render()

    obs = env.reset()
    print obs['OBJ_STATES']
    print obs['_FSM_STATE']

    causal_planner, env, results, i = explore_fluent_space(causal_planner, env, index_map, results, col_label, i)

    # update the shortest paths with the computed paths
    causal_planner.compute_shortest_action_seqs()

    np.savetxt('results.csv', results, delimiter=',', fmt='%s')


def explore_fluent_space(causal_planner, env, index_map, results, col_label, i):
    # hypothesize possible plans to unobserved fluent
    possible_complete_plans = causal_planner.compute_possible_complete_action_seqs()

    for unobserved_fluent in possible_complete_plans.keys():
        possible_action_seq_list = possible_complete_plans[unobserved_fluent]

        for possible_action_seq in possible_action_seq_list:
            for action in possible_action_seq:
                action_str = causal_planner.action_labels[action]
                if action_str in env.action_map:
                    exec_action = env.action_map[action_str]
                    i += 1

                    row_entry = produce_row_entry(index_map, results, col_label, i, exec_action)

                    # append pre-observation entry
                    results.append(row_entry)

                    # take action
                    obs, rew, done, info = env.step(exec_action)
                    # import time
                    print obs['OBJ_STATES']
                    print obs['_FSM_STATE']
                    # #time.sleep(5)

                    # append post-observation entry to results list
                    i += 1
                    new_state = create_state_entry(obs, i, col_label, index_map)
                    results.append(new_state)
                else:
                    raise ValueError('whoops that is not a valid action!')

            # determine if fluent state matches the unobserved state (action sequence is a success), and save successful paths as known_paths
            unobserved_fluent_vec = causality.common.delinearize_fluent_vec(unobserved_fluent, causal_planner.n_fluents)
            final_fluent_vec = np.array(new_state[1:5])
            # we reached the desired state
            if np.array_equal(unobserved_fluent_vec, final_fluent_vec):
                # add the action sequence to the known paths
                if unobserved_fluent not in causal_planner.known_action_seqs.keys():
                    causal_planner.known_action_seqs[unobserved_fluent] = [possible_action_seq]
                else:
                    causal_planner.known_action_seqs[unobserved_fluent].append(possible_action_seq)

            # reset the environment for next possible sequence
            obs = env.reset()
            print('env reset')
            print obs['OBJ_STATES']
            print obs['_FSM_STATE']

            time.sleep(1)

    return causal_planner, env, results, i


def produce_row_entry(index_map, results, col_label, i, action):
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

    return entry

if __name__ == '__main__':
    main()