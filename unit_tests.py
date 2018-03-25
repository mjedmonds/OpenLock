
import gym
import random
import copy
import sys
import jsonpickle
from session_manager import SessionManager
from agents.agent import Agent
from gym_lock.settings_scenario import select_scenario
from logger import SubjectWriter
import gym_lock.rewards as rewards


class ActionTest:
    def __init__(self, name, reward=None):
        self.name = name
        self.reward = reward

    def __eq__(self, other):
        return self.name == other.name and self.reward == other.reward

    def __str__(self):
        return self.name + ',' + str(self.reward)

    def __repr__(self):
        return str(self)


def test_ce3(manager):
    scenario_name = 'CE3'
    trials_to_verify = ['trial1',
                        'trial2',
                        'trial3',
                        'trial4',
                        'trial5',
                        'trial6']
    test_scenario(manager, scenario_name, trials_to_verify)


def test_ce4(manager):
    scenario_name = 'CE4'
    trials_to_verify = ['trial7',
                        'trial8',
                        'trial9',
                        'trial10',
                        'trial11']
    test_scenario(manager, scenario_name, trials_to_verify)


def test_cc3(manager):
    scenario_name = 'CC3'
    trials_to_verify = ['trial1',
                        'trial2',
                        'trial3',
                        'trial4',
                        'trial5',
                        'trial6']
    test_scenario(manager, scenario_name, trials_to_verify)


def test_cc4(manager):
    scenario_name = 'CC4'
    trials_to_verify = ['trial7',
                        'trial8',
                        'trial9',
                        'trial10',
                        'trial11']
    test_scenario(manager, scenario_name, trials_to_verify)


def test_scenario(manager, scenario_name, trials_to_verify):
    scenario = select_scenario(scenario_name, use_physics=True)
    manager.update_scenario(scenario)
    manager.env.use_physics = True

    trial_selected = manager.run_trial_common_setup(scenario_name=scenario_name,
                                                    action_limit=3,
                                                    attempt_limit=5)
    manager.env.reset()

    solutions = scenario.solutions

    for trial in trials_to_verify:
        trial_selected = manager.run_trial_common_setup(scenario_name=scenario_name,
                                                        action_limit=3,
                                                        attempt_limit=5,
                                                        specified_trial=trial)
        for solution in solutions:
            execute_solution(manager, solution)
            manager.env.reset()

        assert(manager.agent.logger.cur_trial.success is True)


def execute_solution(manager, action_seq):
    prev_num_solutions = len(manager.agent.logger.cur_trial.completed_solutions)
    execute_action_seq(manager, action_seq)
    assert(len(manager.agent.logger.cur_trial.completed_solutions) > prev_num_solutions)


def execute_action_seq(manager, action_seq):
    for action_log in action_seq:
        action = manager.env.action_map[action_log.name]
        state, reward, done, opt = manager.env.step(action)
        manager.update_acks()


def verify_file_output_matches(manager):
    pass


def verify_simulator_fsm_match(manager, num_attempts_per_scenario):
    scenarios_to_test = ['CE3',
                         'CE4',
                         'CC3',
                         'CC4']

    for scenario_name in scenarios_to_test:
        scenario = select_scenario(scenario_name, use_physics=True)
        manager.update_scenario(scenario)
        manager.env.use_physics = True

        trial_selected = manager.run_trial_common_setup(scenario_name=scenario_name,
                                                        action_limit=3,
                                                        attempt_limit=num_attempts_per_scenario)

        manager.env.reset()
        for i in range(num_attempts_per_scenario):
            action_idx = random.randrange(len(manager.env.action_map))
            action = manager.env.action_map[manager.env.action_space[action_idx]]
            manager.env.step(action)

            manager.verify_fsm_matches_simulator(manager.env.observation_space)


def test_rewards(manager):

    data_dir = './unit-test-output/rewards'

    scenarios_to_test = ['CE3',
                         'CE4',
                         'CC3',
                         'CC4']

    reward_functions = [
        'basic',
        'change_state',
        'unique_solutions',
        'change_state_unique_solutions',
        'negative_immovable_unique_solutions',
        'negative_immovable',
        'negative_immovable_partial_action_seq',
        'negative_immovable_negative_repeat',
        'negative_immovable_solution_multiplier',
        'negative_immovable_partial_action_seq_solution_multiplier',
        'negative_change_state_partial_action_seq_solution_multiplier',
    ]

    action_seqs_ce = [
        # all three actions do nothing
        [ActionTest('push_inactive0'), ActionTest('push_inactive1'), ActionTest('push_inactive0')],
        # one action moves one lever
        [ActionTest('push_l2'), ActionTest('push_inactive1'), ActionTest('push_inactive0')],
        [ActionTest('push_l1'), ActionTest('push_inactive1'), ActionTest('push_inactive0')],
        # move two levers
        [ActionTest('push_l2'), ActionTest('push_l1'), ActionTest('push_inactive0')],
        [ActionTest('push_l1'), ActionTest('push_l2'), ActionTest('push_inactive0')],
        # unlock the door but don't open
        [ActionTest('push_l2'), ActionTest('push_l0'), ActionTest('push_inactive0')],
        [ActionTest('push_l2'), ActionTest('push_l0'), ActionTest('push_inactive0')],
        [ActionTest('push_l1'), ActionTest('push_l0'), ActionTest('push_inactive0')],
        [ActionTest('push_l1'), ActionTest('push_l0'), ActionTest('push_inactive0')],
        # repeated actions
        [ActionTest('push_l0'), ActionTest('push_l0'), ActionTest('push_inactive0')],
        [ActionTest('push_l1'), ActionTest('push_l1'), ActionTest('push_inactive0')],
        # push 3 levers
        [ActionTest('push_l2'), ActionTest('push_l0'), ActionTest('push_l1')],
        [ActionTest('push_l1'), ActionTest('push_l0'), ActionTest('push_l2')],
        # open the door (repeat solutions)
        [ActionTest('push_l2'), ActionTest('push_l0'), ActionTest('push_door')],
        [ActionTest('push_l2'), ActionTest('push_l0'), ActionTest('push_door')],
        [ActionTest('push_l1'), ActionTest('push_l0'), ActionTest('push_door')],
        [ActionTest('push_l1'), ActionTest('push_l0'), ActionTest('push_door')],
    ]

    action_seqs_cc = [
        # all three actions do nothing
        [ActionTest('push_inactive0'), ActionTest('push_inactive1'), ActionTest('push_inactive0')],
        # one action moves one lever
        [ActionTest('push_l0'), ActionTest('push_inactive1'), ActionTest('push_inactive0')],
        [ActionTest('push_l0'), ActionTest('push_inactive1'), ActionTest('push_inactive0')],
        # move two levers
        [ActionTest('push_l0'), ActionTest('push_inactive0'), ActionTest('push_l2')],
        [ActionTest('push_l0'), ActionTest('push_inactive0'), ActionTest('push_l1')],
        # unlock the door but don't open
        [ActionTest('push_l0'), ActionTest('push_l1'), ActionTest('push_inactive0')],
        [ActionTest('push_l0'), ActionTest('push_l1'), ActionTest('push_inactive0')],
        [ActionTest('push_l0'), ActionTest('push_l2'), ActionTest('push_inactive0')],
        [ActionTest('push_l0'), ActionTest('push_l2'), ActionTest('push_inactive0')],
        # repeated actions
        [ActionTest('push_l0'), ActionTest('push_l0'), ActionTest('push_inactive0')],
        [ActionTest('push_l1'), ActionTest('push_l1'), ActionTest('push_inactive0')],
        # push 3 levers
        [ActionTest('push_l0'), ActionTest('push_l1'), ActionTest('push_l2')],
        [ActionTest('push_l0'), ActionTest('push_l2'), ActionTest('push_l1')],
        # open the door
        [ActionTest('push_l0'), ActionTest('push_l1'), ActionTest('push_door')],
        [ActionTest('push_l0'), ActionTest('push_l1'), ActionTest('push_door')],
        [ActionTest('push_l0'), ActionTest('push_l2'), ActionTest('push_door')],
        [ActionTest('push_l0'), ActionTest('push_l2'), ActionTest('push_door')],
    ]

    for scenario_name in scenarios_to_test:
        scenario_data_dir = data_dir + '/' + scenario_name
        scenario = select_scenario(scenario_name, use_physics=True)
        manager.update_scenario(scenario)
        manager.env.use_physics = True



        if scenario_name == 'CE3' or scenario_name == 'CE4':
            action_seqs = action_seqs_ce
        if scenario_name == 'CC3' or scenario_name == 'CC4':
            action_seqs = action_seqs_cc

        for reward_function in reward_functions:
            trial_selected = manager.run_trial_common_setup(scenario_name=scenario_name,
                                                        action_limit=3,
                                                        attempt_limit=10000)
            manager.env.reset()

            reward_filepath = scenario_data_dir + '/' + reward_function + '.json'
            rewards = []
            i = 0
            for action_seq in action_seqs:
                action_seq_rewards = run_reward_test(manager, action_seq, reward_function)
                rewards.append(action_seq_rewards)
                manager.agent.logger.cur_trial.add_attempt()
                i += 1

            save_reward_file(reward_filepath, rewards, action_seqs)
            reward_file = load_reward_file(reward_filepath)
            assert(reward_file == rewards)

            manager.run_trial_common_finish(trial_selected, False)


def save_reward_file(path, rewards, action_seqs):
    assert(len(rewards) == len(action_seqs))

    # ans = raw_input('Confirm you want to overwrite saved rewards by entering \'y\': ')
    # if ans != 'y':
    #     print('Exiting...')
    #     sys.exit(0)

    json_str = jsonpickle.encode(rewards)
    SubjectWriter.pretty_write(path, json_str)


def load_reward_file(path):
    with open(path, 'r') as f:
        content = f.read()
        rewards = jsonpickle.decode(content)

        return rewards


def run_reward_test(manager, action_seq, reward_function):
    manager.env.reward_mode = reward_function
    rewards = []
    for action_test in action_seq:
        action = manager.env.action_map[action_test.name]
        next_state, reward, done, opt = manager.env.step(action)

        action_test.reward = reward
        rewards.append(action_test)

        manager.update_acks()

    print 'Rewards: {}'.format(str(rewards))
    return rewards


def main():

    env = gym.make('arm_lock-v0')

    params = {'data_dir': '../OpenLockUnitTests'}

    # create session/trial/experiment manager
    agent = Agent(params['data_dir'])
    agent.setup_subject()
    manager = SessionManager(env, agent, params)

    test_ce3(manager)
    test_cc3(manager)
    test_cc4(manager)
    test_ce4(manager)

    # todo: implement verifying file output (json) against a known, correct output
    verify_file_output_matches(manager)

    verify_simulator_fsm_match(manager, 100)

    test_rewards(manager)

    print 'All tests passed'



if __name__ == '__main__':
    main()