
import gym
import numpy as np
import random
from gym_lock.session_manager import SessionManager
from gym_lock.settings_scenario import select_scenario


def test_ce3(env, manager):
    scenario_name = 'CE3'
    trials_to_verify = ['trial1',
                        'trial2',
                        'trial3',
                        'trial4',
                        'trial5',
                        'trial6']
    test_scenario(env, manager, scenario_name, trials_to_verify)


def test_ce4(env, manager):
    scenario_name = 'CE4'
    trials_to_verify = ['trial7',
                        'trial8',
                        'trial9',
                        'trial10',
                        'trial11']
    test_scenario(env, manager, scenario_name, trials_to_verify)


def test_cc3(env, manager):
    scenario_name = 'CC3'
    trials_to_verify = ['trial1',
                        'trial2',
                        'trial3',
                        'trial4',
                        'trial5',
                        'trial6']
    test_scenario(env, manager, scenario_name, trials_to_verify)


def test_cc4(env, manager):
    scenario_name = 'CC4'
    trials_to_verify = ['trial7',
                        'trial8',
                        'trial9',
                        'trial10',
                        'trial11']
    test_scenario(env, manager, scenario_name, trials_to_verify)


def test_scenario(env, manager, scenario_name, trials_to_verify):
    scenario = select_scenario(scenario_name, use_physics=True)
    manager.update_scenario(scenario)
    env.use_physics = True

    trial_selected = manager.run_trial_common_setup(scenario_name=scenario_name,
                                                    action_limit=3,
                                                    attempt_limit=5)
    env.reset()

    solutions = scenario.solutions

    for trial in trials_to_verify:
        trial_selected = manager.run_trial_common_setup(scenario_name=scenario_name,
                                                        action_limit=3,
                                                        attempt_limit=5,
                                                        specified_trial=trial)
        for solution in solutions:
            execute_solution(solution, env)
            env.reset()

        assert(env.logger.cur_trial.success is True)


def execute_solution(action_seq, env):
    prev_num_solutions = len(env.logger.cur_trial.completed_solutions)
    execute_action_seq(action_seq, env)
    assert(len(env.logger.cur_trial.completed_solutions) > prev_num_solutions)


def execute_action_seq(action_seq, env):
    for action_log in action_seq:
        action = env.action_map[action_log.name]
        state, reward, trial_finished, opt = env.step(action)


def verify_simulator_fsm_match(env, manager, num_attempts_per_scenario):
    scenarios_to_test = ['CE3',
                         'CE4',
                         'CC3',
                         'CC4']

    for scenario_name in scenarios_to_test:
        scenario = select_scenario(scenario_name, use_physics=True)
        manager.update_scenario(scenario)
        env.use_physics = True

        trial_selected = manager.run_trial_common_setup(scenario_name=scenario_name,
                                                        action_limit=3,
                                                        attempt_limit=num_attempts_per_scenario)

        env.reset()
        for i in range(num_attempts_per_scenario):
            action_idx = random.randrange(len(env.action_map))
            action = env.action_map[env.action_space[action_idx]]
            env.step(action)

            manager.verify_fsm_matches_simulator(env.observation_space)


def main():

    env = gym.make('arm_lock-v0')

    params = {'data_dir': '../OpenLockUnitTests'}

    # create session/trial/experiment manager
    manager = SessionManager(env, params, human=False)
    verify_simulator_fsm_match(env, manager, 100)
    test_ce3(env, manager)
    test_cc3(env, manager)
    test_cc4(env, manager)
    test_ce4(env, manager)



if __name__ == '__main__':
    main()