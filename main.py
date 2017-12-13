
import gym
import numpy as np
from gym_lock.settings_render import select_scenario
from gym_lock.session_manager import SessionManager

def exit_handler(signum, frame):
   print 'saving results.csv'
   np.savetxt('results.csv', env.results, delimiter=',', fmt='%s')
   exit()


if __name__ == '__main__':

    # PARAMETERS: todo: make these command-line arguments

    # general params
    # training params
    params = {
        'data_dir': '../OpenLockResults/subjects',
        'num_trials': 6,
        'scenario_name': 'CE3',
        'attempt_limit': 10,
        'action_limit': 3,
        'test_scenario_name': 'CE4',
        'test_attempt_limit': 10,
        'test_action_limit': 4
    }

    scenario = select_scenario(params['scenario_name'])
    env = gym.make('arm_lock-v0')
    # create session/trial/experiment manager
    manager = SessionManager(env, params)
    manager.update_scenario(scenario)

    for trial_num in range(0, params['num_trials']):
        manager.run_trial(params['scenario_name'], params['action_limit'], params['attempt_limit'])

    # testing trial
    print "INFO: STARTING TESTING TRIAL"
    scenario = select_scenario(params['test_scenario_name'])
    manager.update_scenario(scenario)
    manager.run_trial(params['test_scenario_name'], params['test_action_limit'], params['test_attempt_limit'])

    raw_input('Press space to continue')
    # record results

