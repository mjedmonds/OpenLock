
import gym
import numpy as np
from gym_lock.settings_render import CURRENT_SCENARIO, select_scenario


def exit_handler(signum, frame):
   print 'saving results.csv'
   np.savetxt('results.csv', env.results, delimiter=',', fmt='%s')
   exit()


if __name__ == '__main__':
    select_scenario('CC4')
    scenario = CURRENT_SCENARIO
    env = gym.make('arm_lock-v0')

    env.render()

    obs = env.reset()
    print obs['OBJ_STATES']
    print obs['_FSM_STATE']

    while (True):
        env.render()

    # simple.fsmm.observable_fsm.get_graph().draw('observable_diagram.png', prog='dot')
    # simple.fsmm.latent_fsm.get_graph().draw('latent_diagram.png', prog='dot')