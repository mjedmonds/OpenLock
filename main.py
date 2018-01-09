
import gym
import time
import sys

from gym_lock.settings_scenario import select_scenario, select_random_scenarios
from gym_lock.session_manager import SessionManager
from gym_lock.settings_trial import PARAMS

# def exit_handler(signum, frame):
#    print 'saving results.csv'
#    np.savetxt('results.csv', env.results, delimiter=',', fmt='%s')
#    exit()

def run_specific_trial_and_scenario(manager, scenario_name, trial_name, action_limit, attempt_limit):
    scenario = select_scenario(scenario_name)
    manager.update_scenario(scenario)
    manager.set_action_limit(action_limit)
    manager.run_trial_human(scenario_name, action_limit, attempt_limit, specified_trial=trial_name)
    manager.finish_subject()
    manager.write_results()
    sys.exit(0)

if __name__ == '__main__':

    # PARAMETERS: todo: make these command-line arguments

    # general params
    # training params
    params = PARAMS['CE4']

    # this section randomly selects a testing and training scenario
    # train_scenario_name, test_scenario_name = select_random_scenarios()
    # params['train_scenario_name'] = train_scenario_name
    # params['test_scenario_name'] = test_scenario_name

    scenario = select_scenario(params['train_scenario_name'])
    env = gym.make('arm_lock-v0')
    # create session/trial/experiment manager
    manager = SessionManager(env, params)
    manager.update_scenario(scenario)
    manager.set_action_limit(params['train_action_limit'])

    # used for debugging, runs a specific scenario & trial
    # run_specific_trial_and_scenario(manager, 'CC3', 'trial5', params['train_action_limit'], params['train_attempt_limit'])

    for trial_num in range(0, params['num_train_trials']):
        manager.run_trial_human(params['train_scenario_name'], params['train_action_limit'], params['train_attempt_limit'])

    # testing trial
    # print "INFO: STARTING TESTING TRIAL"
    if params['test_scenario_name'] is not None:
        scenario = select_scenario(params['test_scenario_name'])
        manager.update_scenario(scenario)
        manager.set_action_limit(params['test_action_limit'])
        # run testing trial with specified trial7
        manager.run_trial_human(params['test_scenario_name'], params['test_action_limit'], params['test_attempt_limit'], specified_trial='trial7')

    manager.finish_subject()
    manager.write_results()
    print "Thank you for participating."
    time.sleep(3)



