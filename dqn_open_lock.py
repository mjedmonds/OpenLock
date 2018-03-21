# -*- coding: utf-8 -*-
import gym
import numpy as np
import os
import sys
import json
from matplotlib import pyplot as plt

# MUST IMPORT FROM gym_lock to properly register the environment
from gym_lock.session_manager import SessionManager
from gym_lock.settings_trial import PARAMS, IDX_TO_PARAMS
from gym_lock.settings_scenario import select_scenario
from gym_lock.common import plot_rewards, plot_rewards_trial_switch_points
from gym_lock.envs.arm_lock_env import ObservationSpace
from dqn_agent import DQNAgent, DDQNAgent

EPISODES = 1000


def run_trials(manager, env, agent, trial_count, num_iters, num_trials, scenario_name, action_limit, attempt_limit, use_dynamic_epsilon, dynamic_epsilon_max, dynamic_epsilon_decay, testing, fig=None):
    # train over multiple iterations over all trials
    for iter_num in range(num_iters):
        manager.completed_trials = []
        for trial_num in range(0, num_trials):
            manager, env, agent = run_single_trial(manager,
                                                   env,
                                                   agent,
                                                   trial_num,
                                                   iter_num,
                                                   scenario_name,
                                                   action_limit,
                                                   attempt_limit,
                                                   use_dynamic_epsilon,
                                                   dynamic_epsilon_max,
                                                   dynamic_epsilon_decay,
                                                   testing, 
                                                   fig=fig)
            trial_count += 1

    return manager, env, agent, trial_count


def run_single_trial(manager, env, agent, trial_num, iter_num, scenario_name, action_limit, attempt_limit, use_dynamic_epsilon=False, dynamic_max=None, dynamic_decay=None, testing=False, fig=None):
    print agent.epsilon_decay
    agent = manager.run_trial_dqn(agent=agent,
                                  scenario_name=scenario_name,
                                  action_limit=action_limit,
                                  attempt_limit=attempt_limit,
                                  trial_count=trial_num,
                                  iter_num=iter_num,
                                  testing=testing,
                                  fig=fig)
    manager.finish_trial(manager.env.logger, manager.writer, human=False, agent=agent)
    print 'One trial complete for subject {}'.format(env.logger.subject_id)
    # reset the epsilon after each trial (to allow more exploration)
    if use_dynamic_epsilon:
        agent.update_dynamic_epsilon(agent.epsilon_min, dynamic_max, dynamic_decay)
    return manager, env, agent


# trains the transfer case and trains multiple transfer cases
def train_transfer_test_transfer(manager, env, agent, params, fig=None):
    # train all training cases/trials
    trial_count = 0
    manager, env, agent, trial_count = run_trials(manager, env, agent, trial_count, params['train_num_iters'], params['train_num_trials'], params['train_scenario_name'], params['train_action_limit'], params['train_attempt_limit'], params['use_dynamic_epsilon'], params['dynamic_epsilon_max'], params['dynamic_epsilon_decay'], testing=False, fig=fig)

    plot_rewards(agent.rewards, agent.epsilons, manager.writer.subject_path + '/training_rewards.png')
    plot_rewards_trial_switch_points(agent.rewards, agent.epsilons, agent.trial_switch_points, manager.writer.subject_path + '/training_rewards_switch_points.png', plot_xticks=False)
    agent.test_start_reward_idx = len(agent.rewards)
    agent.test_start_trial_count = trial_count

    agent.save_model(manager.writer.subject_path + '/models', '/training_final.h5')

    # testing trial
    # print "INFO: STARTING TESTING TRIAL"
    if params['test_scenario_name'] is not None:

        # setup testing trial
        scenario = select_scenario(params['test_scenario_name'], use_physics=params['use_physics'])
        manager.update_scenario(scenario)
        manager.set_action_limit(params['test_action_limit'])
        env.observation_space = ObservationSpace(len(scenario.levers), append_solutions_remaining=False)

        manager, env, agent, trial_count = run_trials(manager, env, agent, trial_count, params['test_num_iters'], params['test_num_trials'], params['test_scenario_name'], params['test_action_limit'], params['test_attempt_limit'], params['use_dynamic_epsilon'], params['dynamic_epsilon_max'], params['dynamic_epsilon_decay'], testing=True)

        plot_rewards(agent.rewards[agent.test_start_reward_idx:], agent.epsilons[agent.test_start_reward_idx:], manager.writer.subject_path + '/testing_rewards.png', width=6, height=6)
        agent.save_model(manager.writer.subject_path + '/models', '/testing_final.h5')

    return manager, env, agent


def train_single_trial(manager, env, agent, params, fig=None):
    manager, env, agent = run_single_trial(manager, env, agent,
                                           trial_num=0,
                                           iter_num=0,
                                           scenario_name=params['train_scenario_name'],
                                           action_limit=params['train_action_limit'],
                                           attempt_limit=params['train_attempt_limit'],
                                           fig=fig)
    plot_rewards(agent.rewards, agent.epsilons, manager.writer.subject_path + '/training_rewards.png') 
    plot_rewards_trial_switch_points(agent.rewards, agent.epsilons, agent.trial_switch_points, manager.writer.subject_path + '/training_rewards_switch_points.png', plot_xticks=False)
    agent.save_model(manager.writer.subject_path + '/models', '/training_final.h5')
    return manager, env, agent


def create_reward_fig():
    # creating the figure
    fig = plt.figure()
    fig.set_size_inches(12, 6)
    plt.ion()
    plt.show()
    return fig


def main():
    # general params
    # training params
    if len(sys.argv) < 2:
        # general params
        # training params
        # PICK ONE and comment others
        params = PARAMS['CE3-CE4']
        # params = PARAMS['CE3-CC4']
        # params = PARAMS['CC3-CE4']
        # params = PARAMS['CC3-CC4']
        # params = PARAMS['CE4']
        # params = PARAMS['CC4']
    else:
        setting = sys.argv[1]
        params = PARAMS[IDX_TO_PARAMS[int(setting) - 1]]
        print('training_scenario: {}, testing_scenario: {}'.format(params['train_scenario_name'],
                                                                   params['test_scenario_name']))
        params['reward_mode'] = sys.argv[2]
    print(params['reward_mode'])
    human_decay_mean = 0.7429 # from human data
    human_decay_median = 0.5480 # from human data

    # RL specific settings
    params['use_physics'] = False
    params['full_attempt_limit'] = True # run to the full attempt limit, regardless of whether or not all solutions were found
    params['train_num_iters'] = 100
    params['test_num_iters'] = 10
    # params['epsilon_decay'] = 0.9955
    # params['epsilon_decay'] = 0.9999
    params['epsilon_decay'] = 0.99999
    params['dynamic_epsilon_decay'] = 0.9955
    params['dynamic_epsilon_max'] = 0.5
    params['use_dynamic_epsilon'] = True
    params['test_num_trials'] = 5

    params['data_dir'] = '../OpenLockRLResults/subjects'
    params['train_attempt_limit'] = 300
    params['test_attempt_limit'] = 300
    params['gamma'] = 0.8    # discount rate
    params['epsilon'] = 1.0  # exploration rate
    params['epsilon_min'] = 0.00
    params['learning_rate'] = 0.0005
    params['batch_size'] = 64

    # SINGLE TRIAL TRAINING
    #params['train_attempt_limit'] = 30000
    #params['epsilon_decay'] = 0.99995
    #params['use_dynamic_epsilon'] = False

    # dummy settings
    # params['num_training_iters'] = 10
    # params['num_testing_iters'] = 10
    # params['train_attempt_limit'] = 30
    # params['test_attempt_limit'] = 30

    # human comparison settings
    # params['num_training_iters'] = 1
    # params['num_testing_iters'] = 1
    # params['train_attempt_limit'] = 300000
    # params['test_attempt_limit'] = 300000
    # params['epsilon_decay'] = human_decay_mean
    # params['dynamic_epsilon_decay'] = human_decay_mean
    # params['dynamic_epsilon_max'] = 1
    # params['use_dynamic_epsilon'] = True

    scenario = select_scenario(params['train_scenario_name'], use_physics=params['use_physics'])

    env = gym.make('arm_lock-v0')

    env.use_physics = params['use_physics']

    # create session/trial/experiment manager
    manager = SessionManager(env, params)
    manager.update_scenario(scenario)
    trial_selected = manager.run_trial_common_setup(scenario_name=params['train_scenario_name'],
                                                    action_limit=params['train_action_limit'],
                                                    attempt_limit=params['train_attempt_limit'])



    # set up observation space
    env.observation_space = ObservationSpace(len(scenario.levers), append_solutions_remaining=False)

    # set reward mode
    env.reward_mode = params['reward_mode']
    print 'Reward mode: {}'.format(env.reward_mode)

    env.full_attempt_limit = True

    state_size = env.observation_space.multi_discrete.shape[0]
    action_size = len(env.action_space)
    # agent = DQNAgent(state_size, action_size, params)
    agent = DDQNAgent(state_size, action_size, params)
    env.reset()
    fig = create_reward_fig()

    # MULTI-TRIAL TRAINING, TESTING
    # runs through all training trials and testing trials
    manager, env, agent = train_transfer_test_transfer(manager, env, agent, params, fig)

    # SINGLE TRIAL TRAINING
    #manager, env, agent = train_single_trial(manager, env, agent, params, fig)

    manager.finish_subject(manager.env.logger, manager.writer, human=False, agent=agent)
    print 'Training & testing complete for subject {}'.format(env.logger.subject_id)


def replot_training_results(path):
    agent_json = json.load(open(path))
    agent_folder = os.path.dirname(path)
    plot_rewards(agent_json['rewards'], agent_json['epsilons'], agent_folder + '/reward_plot.png')


if __name__ == "__main__":
    # agent_path = '../OpenLockRLResults/negative_immovable_partial_seq/2014838386/2014838386_agent.json'
    # replot_training_results(agent_path)
    main()


