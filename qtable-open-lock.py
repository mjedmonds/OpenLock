#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import random
import gym
import numpy as np
import sys
import os
import json
from distutils.dir_util import copy_tree

# MUST IMPORT FROM gym_lock to properly register the environment
from gym_lock.session_manager import SessionManager
from gym_lock.settings_trial import PARAMS, IDX_TO_PARAMS
from gym_lock.settings_scenario import select_scenario
from gym_lock.common import plot_rewards, plot_rewards_trial_switch_points


class OpenLockQ(object):

    def __init__(self, num_states, num_actions, params):
        '''
        Initialize the Q value
        Args:
            num_states: the number of states
            num_actions: total number of actions
            discount: the discount factor
            lr: the learning rate
            epsilon: the epsilon greedy factor
        '''

        self.Q = np.zeros((num_states, num_actions))
        self.state_size = num_states
        self.action_size = num_actions
        self.discount = params['gamma']
        self.lr = params['learning_rate']
        self.epsilon = params['epsilon']
        self.EPS_DECAY = params['epsilon_decay']
        self.START = params['epsilon']
        self.END = params['epsilon_min']
        self.trial_switch_points = []
        self.average_trial_rewards = []
        self.STEPS = 0

    def update(self, s, a, r, s_p):
        '''
        Update the Q value
        Args:
            s: current state
            a: current action
            r: the reward
            s_p: the next state
        '''

        self.Q[s, a] += self.lr * (r + self.discount * np.max(self.Q[s_p, :]) -
                                   self.Q[s, a])

    def action(self, s, train=True):
        '''
        Choose an action to take on state s
        Args:
            s: the state
            train: if train or test
        Ret:
            act: the chosen action to take
        '''

        if not train:
            act = np.argmax(self.Q[s, :])
        else:
            epsilon = np.random.uniform()
            if epsilon >= self.epsilon:
                dist_s = self.Q[s, :]
                # breaking ties randomly
                act = np.random.choice(np.where(dist_s == dist_s.max())[0])
            else:
                act = np.random.randint(low=0, high=5)
        return act

    def update_epsilon(self):
        '''
        Update epsilon
        '''
        self.STEPS += 1
        self.epsilon = self.END + (self.START - self.END) * \
                       np.exp(-1. * self.STEPS / self.EPS_DECAY)

        return


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

    params['use_physics'] = False
    params['num_training_iters'] = 100
    params['num_testing_iters'] = 100
    params['epsilon_decay'] = 0.9955
    params['num_testing_trials'] = 5

    # RL specific settings
    params['data_dir'] = '../OpenLockRLResults/subjects'
    params['train_attempt_limit'] = 300
    params['test_attempt_limit'] = 300
    params['gamma'] = 0.8    # discount rate
    params['epsilon'] = 1.0  # exploration rate
    params['epsilon_min'] = 0.01
    params['learning_rate'] = 0.001
    params['batch_size'] = 64

    scenario = select_scenario(params['train_scenario_name'], use_physics=params['use_physics'])

    env = gym.make('arm_lock-v0')

    env.use_physics = params['use_physics']

    # create session/trial/experiment manager
    manager = SessionManager(env, params, human=False)
    manager.update_scenario(scenario)
    trial_selected = manager.run_trial_common_setup(scenario_name=params['train_scenario_name'],
                                                    action_limit=params['train_action_limit'],
                                                    attempt_limit=params['train_attempt_limit'])

    # copy the entire code base; this is unnecessary but prevents worrying about a particular
    # source code version when trying to reproduce exact parameters
    copy_tree('./', manager.writer.subject_path + '/src/')

    # set up observation space
    env.observation_space = ObservationSpace(len(scenario.levers), append_solutions_remaining=False)

    # set reward mode
    env.reward_mode = params['reward_mode']
    print 'Reward mode: {}'.format(env.reward_mode)

    state_size = env.observation_space.multi_discrete.shape
    action_size = len(env.action_space)
    # agent = DQNAgent(state_size, action_size, params)
    agent = OpenLockQ(state_size, action_size, params)
    env.reset()

    trial_count = 0

    env.human_agent = False
    # train over multiple iterations over all trials
    for iter_num in range(params['num_training_iters']):
        manager.completed_trials = []
        for trial_num in range(0, params['num_train_trials']):
            agent = manager.run_trial_qtable(agent=agent,
                                          scenario_name=params['train_scenario_name'],
                                          action_limit=params['train_action_limit'],
                                          attempt_limit=params['train_attempt_limit'],
                                          trial_count=trial_num,
                                          iter_num=iter_num)

            # reset the epsilon after each trial (to allow more exploration)
            agent.epsilon = 0.5
            trial_count += 1


if __name__ == '__main__':
    main()

    num_episodes = 10000
    discount = env.discount
    reward_history = []
    avg_reward = []
    eval_history = []
    eval_step = []

    for i in range(num_episodes):
        s = env.s0()
        cnt = 0
        reward = 0
        while not env.terminate():
            a = policy.action(s, train=True)
            s_p, r = env.step(s, a)
            policy.update(s, a, r, s_p)
            s = s_p
            reward += r * (discount ** cnt)
            cnt += 1
        policy.update_epsilon()
        reward_history.append(reward)
        avg_reward.append(np.mean(reward_history))
        if i % 10 == 0:
            s = env.s0()
            cnt = 0
            reward = 0
            while not env.terminate():
                a = policy.action(s, train=False)
                s_p, r = env.step(s, a)
                s = s_p
                reward += r * (discount ** cnt)
                cnt += 1
            eval_step.append(i)
            eval_history.append(reward)

    fig, axarr = plt.subplots(1, 3)
    axarr[0].plot(reward_history)
    axarr[1].plot(eval_step, eval_history)
    axarr[2].plot(avg_reward)
    plt.show()