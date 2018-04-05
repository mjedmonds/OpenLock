
import numpy as np
import random
import os
from agents.agent import Agent

from Sum_tree import SumTree
from collections import deque
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, Embedding
from keras.optimizers import Adam
from keras import backend as K


class DAgent(Agent):
    def __init__(self, state_size, action_size, params):
        super(DAgent, self).__init__(params['data_dir'])
        super(DAgent, self).setup_subject(human=False)

        self.params = params

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

    def finish_subject(self, strategy='Deep Q-Learning', transfer_strategy='Deep Q-Learning'):
        super(DAgent, self).finish_subject(strategy, transfer_strategy)

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

    def save_weights(self, save_dir, filename):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.save(save_dir + '/' + filename)

    def save_agent(self, save_dir, testing, iter_num, trial_count, attempt_count):
        if testing:
            save_str = '/agent_test_i_' + str(iter_num) + '_t' + str(trial_count) + '_a' + str(attempt_count) + '.h5'
        else:
            save_str = '/agent_i_' + str(iter_num) + '_t' + str(trial_count) + '_a' + str(attempt_count) + '.h5'
        self.save_weights(save_dir, save_str)

    # load Keras weights (.h5)
    def load(self, name):
        self.model.load_weights(name)

    # save Keras weights (.h5)
    def save(self, name):
        self.model.save_weights(name)


class DQNAgent(DAgent):
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

    def finish_subject(self, strategy='DQN', transfer_strategy='DQN'):
        super(DAgent, self).finish_subject(strategy, transfer_strategy)

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


class DDQNAgent(DAgent):
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

    def finish_subject(self, strategy='DDQN', transfer_strategy='DDQN'):
        super(DAgent, self).finish_subject(strategy, transfer_strategy)

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

class DDQN_PRIORITY_Agent(DAgent):
    def __init__(self, state_size, action_size, params, capacity = 200000):
        super(DDQN_PRIORITY_Agent, self).__init__(state_size, action_size, params)
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
        self.memory = Memory(memory_capacity= capacity)

    def finish_subject(self, strategy='DDQN_PRIORITY', transfer_strategy='DDQN_PRIORITY'):
        super(DAgent, self).finish_subject(strategy, transfer_strategy)

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

    def _remember(self, state, action, reward, next_state, done):
        self.memory.store([state, action, reward, next_state, done])

    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())

    def replay(self):
        [weights, index, data] = self.memory.sample(self.batch_size)

        x = [] # state list
        y = [] # target list
        # get error to update priority
        for i in range(len(data)):
            [state, action, reward, next_state, done] = data[i]
            target = self.model.predict(state)
            target_prev = target[0][action]
            if done:
                target[0][action] = reward
            else:
                a = self.model.predict(next_state)[0]
                t = self.target_model.predict(next_state)[0]
                target[0][action] = reward + self.gamma * t[np.argmax(a)]
            errors = abs(target_prev - target[0][action])
            # update priority
            self.memory.update_priority(errors, index[i])
            x.append(state)
            y.append(target)
        x = np.array(x).squeeze()
        y = np.array(y).squeeze()
        self.model.fit(x, y, epochs=1, verbose=0,sample_weight= weights)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

class Memory(object):
    def __init__(self, memory_capacity):
        self.sum_tree = SumTree(memory_capacity)
        self.abs_upper_err = 1.
        self.epsilon = 0.01 # avoid zero priority
        self.alpha = 0.6 # convert TD error to priority
        self.beta = 0.4 # sampling
        self.delta_beta_each_sampling = 0.001

    def store(self, transition):
        # find max priority from sum_tree
        max_priority = np.max(self.sum_tree.tree[-self.sum_tree.capacity:])
        if max_priority == 0:
            max_priority = self.abs_upper_err # if priority is zero, set a low value on it
        self.sum_tree.add(max_priority, transition)

    def sample(self, n):
        # initialize the parameter
        index_array = np.empty([n,],dtype=np.int64)
        memory_array = np.empty([n,], dtype= object)
        weight_array = np.empty([n])

        # sample transition
        p_seg = self.sum_tree.total()/n
        # increase beta
        self.beta = np.min([1, self.beta + self.delta_beta_each_sampling])

        for i in range(n):
            seg = np.random.uniform(p_seg*i, p_seg*(i+1))
            index, p, data = self.sum_tree.get(seg)
            prob = p/self.sum_tree.total()
            # compute importance samplng weight
            weight_array[i] = (n*prob)**(-self.beta)
            index_array[i] = index
            memory_array[i] = data
        return weight_array, index_array, memory_array

    def update_priority(self, TD_error, index):
        TD_error = TD_error+self.epsilon
        clip_error = np.minimum(TD_error, self.abs_upper_err)
        p = clip_error**self.alpha
        self.sum_tree.update(index, p)
