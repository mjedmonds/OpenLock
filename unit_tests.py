
import gym
import random
import sys
import jsonpickle
from gym_lock.settings_scenario import select_scenario
from logger_env import SubjectWriter

# NOTE: to run this code, the OpenLockAgents must be in your PYTHONPATH
from agent import Agent


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


def test_ce3(env):
    scenario_name = 'CE3'
    trials_to_verify = ['trial1',
                        'trial2',
                        'trial3',
                        'trial4',
                        'trial5',
                        'trial6']
    test_scenario(env, scenario_name, trials_to_verify)


def test_ce4(env):
    scenario_name = 'CE4'
    trials_to_verify = ['trial7',
                        'trial8',
                        'trial9',
                        'trial10',
                        'trial11']
    test_scenario(env, scenario_name, trials_to_verify)


def test_cc3(env):
    scenario_name = 'CC3'
    trials_to_verify = ['trial1',
                        'trial2',
                        'trial3',
                        'trial4',
                        'trial5',
                        'trial6']
    test_scenario(env, scenario_name, trials_to_verify)


def test_cc4(env):
    scenario_name = 'CC4'
    trials_to_verify = ['trial7',
                        'trial8',
                        'trial9',
                        'trial10',
                        'trial11']
    test_scenario(env, scenario_name, trials_to_verify)


def test_scenario(agent, scenario_name, trials_to_verify):
    scenario = select_scenario(scenario_name, use_physics=True)
    agent.env.update_scenario(scenario)
    agent.env.use_physics = True

    trial_selected = agent.setup_trial(scenario_name=scenario_name,
                                       action_limit=3,
                                       attempt_limit=5)
    agent.env.reset()

    solutions = scenario.solutions

    for trial in trials_to_verify:
        trial_selected = agent.setup_trial(scenario_name=scenario_name,
                                           action_limit=3,
                                           attempt_limit=5,
                                           specified_trial=trial)
        for solution in solutions:
            execute_solution(agent, solution)
            agent.env.reset()

        assert(agent.logger.cur_trial.success is True)

        agent.finish_trial(trial_selected, False)


def execute_solution(agent, action_seq):
    prev_num_solutions = len(agent.logger.cur_trial.completed_solutions)
    execute_action_seq(agent, action_seq)
    assert(len(agent.logger.cur_trial.completed_solutions) > prev_num_solutions)


def execute_action_seq(agent, action_seq):
    for action_log in action_seq:
        action = agent.env.action_map[action_log.name]
        state, reward, done, opt = agent.env.step(action)
        agent.finish_action()


def verify_file_output_matches(env):
    pass


def verify_simulator_fsm_match(agent, num_attempts_per_scenario):
    scenarios_to_test = ['CE3',
                         'CE4',
                         'CC3',
                         'CC4']

    for scenario_name in scenarios_to_test:
        scenario = select_scenario(scenario_name, use_physics=True)
        agent.env.update_scenario(scenario)
        agent.env.use_physics = True

        trial_selected = agent.setup_trial(scenario_name=scenario_name,
                                           action_limit=3,
                                           attempt_limit=num_attempts_per_scenario)

        agent.env.reset()
        for i in range(num_attempts_per_scenario):
            action_idx = random.randrange(len(agent.env.action_map))
            action = agent.env.action_map[agent.env.action_space[action_idx]]
            agent.env.step(action)

            agent.verify_fsm_matches_simulator(agent.env.observation_space)

            agent.finish_action()

        agent.finish_trial(trial_selected, False)


def test_rewards(agent):

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
        agent.env.update_scenario(scenario)
        agent.env.use_physics = True

        if scenario_name == 'CE3' or scenario_name == 'CE4':
            action_seqs = action_seqs_ce
        if scenario_name == 'CC3' or scenario_name == 'CC4':
            action_seqs = action_seqs_cc

        for reward_function in reward_functions:
            trial_selected = agent.setup_trial(scenario_name=scenario_name,
                                               action_limit=3,
                                               attempt_limit=10000)
            agent.env.reset()

            reward_filepath = scenario_data_dir + '/' + reward_function + '.json'
            rewards = []
            i = 0
            for action_seq in action_seqs:
                action_seq_rewards = run_reward_test(agent, action_seq, reward_function)
                rewards.append(action_seq_rewards)
                agent.logger.cur_trial.add_attempt()
                i += 1

            # uncomment to save the rewards to a file
            #save_reward_file(reward_filepath, rewards, action_seqs)

            reward_file = load_reward_file(reward_filepath)
            if rewards != reward_file:
                mismatches = [i for i in reward_file if rewards[i] != reward_file[i]]
                reward_file_mismatches = [reward_file[i] for i in mismatches]
                rewards_mismatches = [rewards[i] for i in mismatches]
                assert_err = 'Reward does not match in {} reward function. Received reward of {}. Expected reward of {}'.format(reward_function, rewards_mismatches, reward_file_mismatches)
                assert reward_file == rewards, assert_err

            agent.finish_trial(trial_selected, False)


def save_reward_file(path, rewards, action_seqs):
    assert(len(rewards) == len(action_seqs))

    ans = eval(input('Confirm you want to overwrite saved rewards by entering \'y\': '))
    if ans != 'y':
        print('Exiting...')
        sys.exit(0)

    json_str = jsonpickle.encode(rewards)
    SubjectWriter.pretty_write(path, json_str)


def load_reward_file(path):
    with open(path, 'r') as f:
        content = f.read()
        rewards = jsonpickle.decode(content)

        return rewards


def run_reward_test(agent, action_seq, reward_function):
    agent.env.reward_mode = reward_function
    rewards = []
    for action_test in action_seq:
        action = agent.env.action_map[action_test.name]
        next_state, reward, done, opt = agent.env.step(action)

        action_test.reward = reward
        rewards.append(action_test)

        agent.finish_action()

    print('Rewards: {}'.format(str(rewards)))
    return rewards


def main():

    env = gym.make('arm_lock-v0')

    params = {'data_dir': '../OpenLockUnitTests'}

    # create session/trial/experiment manager
    agent = Agent(params, env)
    agent.setup_subject()

    print('Starting unit tests.')

    print('Testing CE3.')
    test_ce3(agent)
    print('Testing CC3')
    test_cc3(agent)
    print('Testing CC4')
    test_cc4(agent)
    print('Testing CE4.')
    test_ce4(agent)

    # todo: implement verifying file output (json) against a known, correct output
    verify_file_output_matches(agent)

    print('Verifying physics simulator and FSM output matches.')
    verify_simulator_fsm_match(agent, 100)

    print('Verifying rewards match saved values.')
    # bypass physics sim
    agent.env.use_physics = False
    test_rewards(agent)

    print('All tests passed')


if __name__ == '__main__':
    main()
