# -*- coding: utf-8 -*-
import random
import gym
import numpy as np
import sys
from matplotlib import pyplot as plt
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
# MUST IMPORT FROM gym_lock to properly register the environment
from gym_lock.session_manager import SessionManager
from gym_lock.settings_trial import PARAMS, IDX_TO_PARAMS
from gym_lock.settings_scenario import select_scenario
from gym_lock.space_manager import ObservationSpace, ActionSpace

EPISODES = 1000


class DQNAgent:
    def __init__(self, state_size, action_size, epsilon_decay, num_training_iters=None):
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
        self.weights = [64, 64]
        self.epsilon_save_rate = 100
        self.batch_size = 64
        self.num_training_iters = num_training_iters
        self.model = self._build_model()


    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        # first layer
        model.add(Dense(self.weights[0], input_dim=self.state_size, activation='relu'))
        # add other layers
        for i in range(1, len(self.weights)):
            model.add(Dense(self.weights[i], activation='relu'))
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

    def save(self, name):
        self.model.save_weights(name)

    def plot_rewards(self, filename):
        plt.clf()
        last = 0
        assert len(self.epsilons) == len(self.rewards)
        r, = plt.plot(self.rewards, color='red', linestyle='-', label='reward')
        e, = plt.plot(self.epsilons, color='blue', linestyle='--', alpha=0.5, label='epsilon')
        plt.legend([r, e])
        plt.savefig(filename)


if __name__ == "__main__":

    reward_mode = 'basic'
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
        params = PARAMS[IDX_TO_PARAMS[int(setting)-1]]
        print('training_scenario: {}, testing_scenario: {}'.format(params['train_scenario_name'], params['test_scenario_name']))
        reward_mode = sys.argv[2]

    use_physics = False
    num_training_iters = 1000
    epsilon_decay = 0.99999

    # RL specific settings
    params['data_dir'] = '../OpenLockRLResults/subjects'
    params['train_attempt_limit'] = 30
    params['test_attempt_limit'] = 30

    scenario = select_scenario(params['train_scenario_name'], use_physics=use_physics)

    env = gym.make('arm_lock-v0')

    env.use_physics = use_physics

    # create session/trial/experiment manager
    manager = SessionManager(env, params, human=False)
    manager.update_scenario(scenario)
    trial_selected = manager.run_trial_common_setup(scenario_name=params['train_scenario_name'],
                                                    action_limit=params['train_action_limit'],
                                                    attempt_limit=params['train_attempt_limit'])

    # set up observation space
    obs_space = ObservationSpace(len(scenario.levers))
    env.observation_space = obs_space.multi_discrete

    env.reward_mode = reward_mode
    print 'Reward mode: {}'.format(env.reward_mode)

    state_size = obs_space.multi_discrete.shape
    action_size = len(env.action_space)
    agent = DQNAgent(state_size, action_size, epsilon_decay, num_training_iters)
    env.reset()

    # train over multiple iterations over all trials
    for iter_num in range(num_training_iters):
        manager.completed_trials = []
        for trial_num in range(0, params['num_train_trials']):
            agent = manager.run_trial_computer(agent=agent,
                                               obs_space=obs_space,
                                               scenario_name=params['train_scenario_name'],
                                               action_limit=params['train_action_limit'],
                                               attempt_limit=params['train_attempt_limit'],
                                               trial_count=trial_num,
                                               iter_num=iter_num)

    agent.plot_rewards(manager.writer.subject_path + '/training_rewards.png')

    # testing trial
    # print "INFO: STARTING TESTING TRIAL"
    if params['test_scenario_name'] is not None:
        scenario = select_scenario(params['test_scenario_name'], use_physics=use_physics)
        manager.update_scenario(scenario)
        manager.set_action_limit(params['test_action_limit'])
        # run testing trial with specified trial7
        agent = manager.run_trial_computer(agent=agent,
                                           obs_space=obs_space,
                                           scenario_name=params['test_scenario_name'],
                                           action_limit=params['test_action_limit'],
                                           attempt_limit=params['test_attempt_limit'],
                                           trial_count=params['num_train_trials'] + 1,
                                           iter_num=0)

    manager.finish_subject(manager.env.logger, manager.writer, human=False, agent=agent)
    print 'Training complete for subject {}'.format(env.logger.subject_id)
    sys.exit(0)
    # agent.load("./save/cartpole-dqn.h5")
    # done = False
    # batch_size = 32
    #
    # for e in range(EPISODES):
    #     env.reset()
    #     state = obs_space.create_discrete_observation_from_state(env)
    #     state = np.reshape(state, [1, state_size])
    #     for time in range(500):
    #         # env.render()
    #         action = agent.act(state)
    #         next_state, reward, done, _ = env.step(action)
    #         reward = reward if not done else -10
    #         next_state = np.reshape(next_state, [1, state_size])
    #         agent.remember(state, action, reward, next_state, done)
    #         state = next_state
    #         if done:
    #             print("episode: {}/{}, score: {}, e: {:.2}"
    #                   .format(e, EPISODES, time, agent.epsilon))
    #             break
    #     if len(agent.memory) > batch_size:
    #         agent.replay(batch_size)
    #     # if e % 10 == 0:
    #     #     agent.save("./save/cartpole-dqn.h5")
