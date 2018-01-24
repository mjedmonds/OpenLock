# -*- coding: utf-8 -*-
import random
import gym
import numpy as np
import sys
import os
import json
from distutils.dir_util import copy_tree
from matplotlib import pyplot as plt
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam

# MUST IMPORT FROM gym_lock to properly register the environment
from gym_lock.session_manager import SessionManager
from gym_lock.settings_trial import PARAMS, IDX_TO_PARAMS
from gym_lock.settings_scenario import select_scenario
from gym_lock.space_manager import ObservationSpace, ActionSpace
from gym_lock.common import plot_rewards, plot_rewards_trial_switch_points


EPISODES = 1000


class DQNAgent:
    def __init__(self, state_size, action_size, epsilon_decay, reward_mode, num_training_iters, params):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.8    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.1
        self.epsilon_decay = epsilon_decay
        self.learning_rate = 0.001
        self.epsilons = []
        self.rewards = []
        self.trial_switch_points = []
        self.average_trial_rewards = []
        self.weights = [
                        ('dense', 128),
                        # ('dropout', 0.5),
                        ('dense', 128),
                        # ('dropout', 0.5)
                        ('dense', 128),
                        # ('dropout', 0.5)
                        ('dense', 128),
                        ]
        self.epsilon_save_rate = 100
        self.batch_size = 64
        self.num_training_iters = num_training_iters
        self.train_attempt_limit = params['train_attempt_limit']
        self.train_action_limit = params['train_action_limit']
        self.test_attempt_limit = params['test_attempt_limit']
        self.test_action_limit = params['test_action_limit']
        self.reward_mode = reward_mode
        self.model = self._build_model()


    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        # first layer
        model.add(Dense(self.weights[0][1], input_dim=self.state_size, activation='relu'))
        # add other layers
        for i in range(1, len(self.weights)):
            if self.weights[i][0] == 'dense':
                model.add(Dense(self.weights[i][1], activation='relu'))
            if self.weights[i][0] == 'dropout':
                model.add(Dropout(self.weights[i][1]))
        # output layer
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self):
        minibatch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_reward(self, reward):
        self.epsilons.append(self.epsilon)
        self.rewards.append(reward)

    def load(self, name):
        self.model.load_weights(name)

    def save_model(self, save_dir, filename):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.save(save_dir + '/' + filename)

    def save(self, name):
        self.model.save_weights(name)


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
        reward_mode = sys.argv[2]

    use_physics = False
    num_training_iters = 100
    epsilon_decay = 0.99999

    # RL specific settings
    params['data_dir'] = '../OpenLockRLResults/subjects'
    params['train_attempt_limit'] = 300
    params['test_attempt_limit'] = 300

    scenario = select_scenario(params['train_scenario_name'], use_physics=use_physics)

    env = gym.make('arm_lock-v0')

    env.use_physics = use_physics

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
    env.reward_mode = reward_mode
    print 'Reward mode: {}'.format(env.reward_mode)

    state_size = env.observation_space.multi_discrete.shape
    action_size = len(env.action_space)
    agent = DQNAgent(state_size, action_size, epsilon_decay, reward_mode, num_training_iters, params)
    env.reset()

    # train over multiple iterations over all trials
    for iter_num in range(num_training_iters):
        manager.completed_trials = []
        for trial_num in range(0, params['num_train_trials']):
            agent = manager.run_trial_dqn(agent=agent,
                                          scenario_name=params['train_scenario_name'],
                                          action_limit=params['train_action_limit'],
                                          attempt_limit=params['train_attempt_limit'],
                                          trial_count=trial_num,
                                          iter_num=iter_num)

    plot_rewards(agent.rewards, agent.epsilons, manager.writer.subject_path + '/training_rewards.png')
    plot_rewards_trial_switch_points(agent.rewards, agent.epsilons, agent.trial_switch_points, manager.writer.subject_path + '/training_rewards_switch_points.png', plot_xticks=False)
    agent.test_start_idx = len(agent.rewards)
    agent.save_model(manager.writer.subject_path + '/models', '/training_final.h5')

    # testing trial
    # print "INFO: STARTING TESTING TRIAL"
    if params['test_scenario_name'] is not None:
        scenario = select_scenario(params['test_scenario_name'], use_physics=use_physics)
        manager.update_scenario(scenario)
        manager.set_action_limit(params['test_action_limit'])
        # run testing trial with specified trial7
        agent = manager.run_trial_dqn(agent=agent,
                                      scenario_name=params['test_scenario_name'],
                                      action_limit=params['test_action_limit'],
                                      attempt_limit=params['test_attempt_limit'],
                                      trial_count=params['num_train_trials'] + 1,
                                      iter_num=0,
                                      specified_trial='trial7',
                                      testing=True)

        plot_rewards(agent.rewards[agent.test_start_idx:], agent.epsilons[agent.test_start_idx:], manager.writer.subject_path + '/testing_rewards.png', width=6, height=6)
    agent.save_model(manager.writer.subject_path + '/models', '/testing_final.h5')
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


