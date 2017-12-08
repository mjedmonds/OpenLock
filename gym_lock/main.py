
import gym
import numpy as np
from gym_lock.settings_render import select_scenario
from gym_lock.setttings_trial import select_random_trial, select_trial


def exit_handler(signum, frame):
   print 'saving results.csv'
   np.savetxt('results.csv', env.results, delimiter=',', fmt='%s')
   exit()


def get_trial(name, completed_trials):
    if name != 'CE4' and name != 'CC4':
        # select a random trial and add it to the scenario
        trial, configs, opt_params = select_random_trial(completed_trials, 1, 6)
    else:
        trial, configs, opt_params = select_trial('trial7')

    return trial, configs, opt_params


if __name__ == '__main__':
    scenario_name = 'CE4'
    scenario = select_scenario(scenario_name)
    env = gym.make('arm_lock-v0')
    # tell the environemnt what scenario is currently used
    env.scenario = scenario

    num_trials = 6
    attempt_limit = 10
    action_limit = 3
    completed_trials = []
    env.action_limit = action_limit

    for trial_num in range(0, num_trials):
        env.attempt_count = 0
        trial_selected, lever_configs, lever_opt_params = get_trial(scenario_name, completed_trials)
        scenario.set_lever_configs(lever_configs, lever_opt_params)

        obs = env.reset()

        while env.attempt_count < attempt_limit:
            env.render()

        # record results

    # simple.fsmm.observable_fsm.get_graph().draw('observable_diagram.png', prog='dot')
    # simple.fsmm.latent_fsm.get_graph().draw('latent_diagram.png', prog='dot')