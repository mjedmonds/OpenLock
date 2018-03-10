
import numpy as np
import random
import os

from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras import backend as K


class Agent(object):
    def __init__(self, state_size, action_size, params):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=200000)
        self.gamma = params['gamma']    # discount rate
        self.epsilon = params['epsilon']  # exploration rate
        self.epsilon_min = params['epsilon_min']
        self.epsilon_decay = params['epsilon_decay']
        self.learning_rate = params['learning_rate']
        self.epsilons = []
        self.rewards = []
        self.trial_rewards = []
        self.trial_switch_points = []
        self.average_trial_rewards = []
        self.batch_size = params['batch_size']
        self.train_num_iters = params['train_num_iters']
        self.train_attempt_limit = params['train_attempt_limit']
        self.train_action_limit = params['train_action_limit']
        self.test_attempt_limit = params['test_attempt_limit']
        self.test_action_limit = params['test_action_limit']
        self.reward_mode = params['reward_mode']

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def save_reward(self, reward, trial_reward):
        self.epsilons.append(self.epsilon)
        self.rewards.append(reward)
        self.trial_rewards.append(trial_reward)

    # update the epsilon after every trial once it drops below epsilon_threshold
    def update_dynamic_epsilon(self, epsilon_threshold, new_epsilon, new_epsilon_decay):
        if self.epsilon < epsilon_threshold:
            self.epsilon = new_epsilon
            self.epsilon_decay = new_epsilon_decay

    def save_model(self, save_dir, filename):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.save(save_dir + '/' + filename)

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


class DQNAgent(Agent):
    def __init__(self, state_size, action_size, params):
        super(DQNAgent, self).__init__(state_size, action_size, params)
        self.weights = [
                        ('dense', 128),
                        # ('dropout', 0.5),
                        ('dense', 128),
                        # ('dropout', 0.5)
                        ('dense', 128),
                        # ('dropout', 0.5)
                        ('dense', 128),
                        ]
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


class DDQNAgent(Agent):
    def __init__(self, state_size, action_size, params):
        super(DDQNAgent, self).__init__(state_size, action_size, params)
        self.weights = [
                        ('dense', 128),
                        # ('dropout', 0.5),
                        ('dense', 128),
                        ('dense', 128),
                        ('dense', 128),
                        ]
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _huber_loss(self, target, prediction):
        # sqrt(1+error^2)-1
        error = prediction - target
        return K.mean(K.sqrt(1+K.square(error))-1, axis=-1)

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        # input layer
        model.add(Dense(self.weights[0][1], input_dim=self.state_size, activation='relu'))
        # add other layers
        for i in range(1, len(self.weights)):
            if self.weights[i][0] == 'dense':
                model.add(Dense(self.weights[i][1], activation='relu'))
            if self.weights[i][0] == 'dropout':
                model.add(Dropout(self.weights[i][1]))
        # output layer
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss=self._huber_loss,
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())

    def replay(self):
        minibatch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = self.model.predict(state)
            if done:
                target[0][action] = reward
            else:
                a = self.model.predict(next_state)[0]
                t = self.target_model.predict(next_state)[0]
                target[0][action] = reward + self.gamma * t[np.argmax(a)]
            self.model.fit(state, target, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
