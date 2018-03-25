# -*- coding: utf-8 -*-
import gym
import os
import sys
import json
from matplotlib import pyplot as plt

# MUST IMPORT FROM gym_lock to properly register the environment
from session_manager import SessionManager
from gym_lock.settings_trial import PARAMS, IDX_TO_PARAMS
from gym_lock.settings_scenario import select_scenario
from gym_lock.common import plot_rewards, plot_rewards_trial_switch_points
from gym_lock.envs.arm_lock_env import ObservationSpace
from agents.dqn_agent import DDQNAgent

EPISODES = 1000


def run_trials(manager, trial_count, num_iters, num_trials, scenario_name, action_limit, attempt_limit, use_dynamic_epsilon, dynamic_epsilon_max, dynamic_epsilon_decay, test_trial, fig=None):
    # train over multiple iterations over all trials
    for iter_num in range(num_iters):
        manager.completed_trials = []
        for trial_num in range(0, num_trials):
            manager = run_single_trial(manager,
                                       trial_num,
                                       iter_num,
                                       scenario_name,
                                       action_limit,
                                       attempt_limit,
                                       use_dynamic_epsilon,
                                       dynamic_epsilon_max,
                                       dynamic_epsilon_decay,
                                       test_trial,
                                       fig=fig)
            trial_count += 1

    return manager, trial_count


def run_single_trial(manager, trial_num, iter_num, scenario_name, action_limit, attempt_limit, use_dynamic_epsilon=False, dynamic_max=None, dynamic_decay=None, test_trial=False, fig=None):
    manager.run_trial_dqn(scenario_name=scenario_name,
                          action_limit=action_limit,
                          attempt_limit=attempt_limit,
                          trial_count=trial_num,
                          iter_num=iter_num,
                          test_trial=test_trial,
                          fig=fig)
    print 'One trial complete for subject {}'.format(manager.agent.logger.subject_id)
    # reset the epsilon after each trial (to allow more exploration)
    if use_dynamic_epsilon:
        manager.agent.update_dynamic_epsilon(manager.agent.epsilon_min, dynamic_max, dynamic_decay)
    return manager


# trains the transfer case and trains multiple transfer cases
def train_transfer_test_transfer(manager, fig=None):
    # train all training cases/trials
    params = manager.params
    trial_count = 0
    manager, trial_count = run_trials(manager, trial_count, params['train_num_iters'], params['train_num_trials'], params['train_scenario_name'], params['train_action_limit'], params['train_attempt_limit'], params['use_dynamic_epsilon'], params['dynamic_epsilon_max'], params['dynamic_epsilon_decay'], test_trial=False, fig=fig)

    plot_rewards(manager.agent.rewards, manager.agent.epsilons, manager.agent.writer.subject_path + '/training_rewards.png')
    plot_rewards_trial_switch_points(manager.agent.rewards, manager.agent.epsilons, manager.agent.trial_switch_points, manager.agent.writer.subject_path + '/training_rewards_switch_points.png', plot_xticks=False)
    manager.agent.test_start_reward_idx = len(manager.agent.rewards)
    manager.agent.test_start_trial_count = trial_count

    manager.agent.save_weights(manager.agent.writer.subject_path + '/models', '/training_final.h5')

    # testing trial
    # print "INFO: STARTING TESTING TRIAL"
    if params['test_scenario_name'] is not None:

        # setup testing trial
        scenario = select_scenario(params['test_scenario_name'], use_physics=params['use_physics'])
        manager.update_scenario(scenario)
        manager.set_action_limit(params['test_action_limit'])
        manager.env.observation_space = ObservationSpace(len(scenario.levers), append_solutions_remaining=False)

        manager, trial_count = run_trials(manager, trial_count, params['test_num_iters'], params['test_num_trials'], params['test_scenario_name'], params['test_action_limit'], params['test_attempt_limit'], params['use_dynamic_epsilon'], params['dynamic_epsilon_max'], params['dynamic_epsilon_decay'], test_trial=True)

        plot_rewards(manager.agent.rewards[manager.agent.test_start_reward_idx:], manager.agent.epsilons[manager.agent.test_start_reward_idx:], manager.agent.writer.subject_path + '/testing_rewards.png', width=6, height=6)
        manager.agent.save_weights(manager.agent.writer.subject_path + '/models', '/testing_final.h5')

    return manager


def train_single_trial(manager, params, fig=None):
    manager = run_single_trial(manager,
                               trial_num=0,
                               iter_num=0,
                               scenario_name=params['train_scenario_name'],
                               action_limit=params['train_action_limit'],
                               attempt_limit=params['train_attempt_limit'],
                               fig=fig)
    plot_rewards(manager.agent.rewards, manager.agent.epsilons, manager.agent.writer.subject_path + '/training_rewards.png')
    plot_rewards_trial_switch_points(manager.agent.rewards, manager.agent.epsilons, manager.agent.trial_switch_points, manager.agent.writer.subject_path + '/training_rewards_switch_points.png', plot_xticks=False)
    manager.agent.save_model(manager.writer.subject_path + '/models', '/training_final.h5')
    return manager


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
    params['train_num_iters'] = 10
    params['test_num_iters'] = 10
    params['train_attempt_limit'] = 30
    params['test_attempt_limit'] = 30

    # human comparison settings
    # params['train_num_iters'] = 1
    # params['test_num_iters'] = 1
    # params['train_attempt_limit'] = 300000
    # params['test_attempt_limit'] = 300000
    # params['epsilon_decay'] = human_decay_mean
    # params['dynamic_epsilon_decay'] = human_decay_mean
    # params['dynamic_epsilon_max'] = 1
    # params['use_dynamic_epsilon'] = True

    scenario = select_scenario(params['train_scenario_name'], use_physics=params['use_physics'])

    # setup initial env
    env = gym.make('arm_lock-v0')

    env.use_physics = params['use_physics']
    env.full_attempt_limit = True

    # set up observation space
    env.observation_space = ObservationSpace(len(scenario.levers), append_solutions_remaining=False)

    # set reward mode
    env.reward_mode = params['reward_mode']
    print 'Reward mode: {}'.format(env.reward_mode)

    agent = DDQNAgent(1, 1, params)

    # create session/trial/experiment manager
    # TODO: passing a fake agent here is a hack
    manager = SessionManager(env, agent, params)
    manager.update_scenario(scenario)
    trial_selected = manager.run_trial_common_setup(scenario_name=params['train_scenario_name'],
                                                    action_limit=params['train_action_limit'],
                                                    attempt_limit=params['train_attempt_limit'])

    # setup agent
    state_size = manager.env.observation_space.multi_discrete.shape[0]
    action_size = len(manager.env.action_space)

    # agent = DQNAgent(state_size, action_size, params)
    agent = DDQNAgent(state_size, action_size, params)
    # update agent to be a properly initialized agent
    # TODO: this is also a hack
    manager.agent = agent

    manager.env.reset()
    fig = create_reward_fig()

    # MULTI-TRIAL TRAINING, TESTING
    # runs through all training trials and testing trials
    manager = train_transfer_test_transfer(manager, fig)

    # SINGLE TRIAL TRAINING
    #manager, env, agent = train_single_trial(manager, env, agent, params, fig)

    manager.agent.finish_subject()
    print 'Training & testing complete for subject {}'.format(manager.agent.logger.subject_id)


def replot_training_results(path):
    agent_json = json.load(open(path))
    agent_folder = os.path.dirname(path)
    plot_rewards(agent_json['rewards'], agent_json['epsilons'], agent_folder + '/reward_plot.png')


if __name__ == "__main__":
    # agent_path = '../OpenLockRLResults/negative_immovable_partial_seq/2014838386/2014838386_agent.json'
    # replot_training_results(agent_path)
    main()


