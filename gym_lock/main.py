
import gym
import numpy as np
from gym_lock.settings_render import select_scenario
from gym_lock.setttings_trial import select_random_trial, select_trial


def exit_handler(signum, frame):
   print 'saving results.csv'
   np.savetxt('results.csv', env.results, delimiter=',', fmt='%s')
   exit()


if __name__ == '__main__':
    scenario_name = 'CE4'
    scenario = select_scenario(scenario_name)
    env = gym.make('arm_lock-v0')
    # tell the environemnt what scenario is currently used
    env.scenario = scenario

    completed_trials = []
    if scenario_name != 'CE4' and scenario_name != 'CC4':
        # select a random trial and add it to the scenario
        trial_selected, lever_configs, lever_opt_params = select_random_trial(completed_trials, 1, 6)
    else:
        trial_selected, lever_configs, lever_opt_params = select_trial('trial7')
    scenario.set_lever_configs(lever_configs, lever_opt_params)

    env.reset()
    env.render()

    obs = env.reset()
    print obs['OBJ_STATES']
    print obs['_FSM_STATE']

    while (True):
        env.render()

    # simple.fsmm.observable_fsm.get_graph().draw('observable_diagram.png', prog='dot')
    # simple.fsmm.latent_fsm.get_graph().draw('latent_diagram.png', prog='dot')