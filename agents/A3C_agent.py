
import numpy as np
import random
import os
from agents.agent import Agent

from collections import deque
from A3C_LSTM.ac_network import AC_Network
import tensorflow as tf
import scipy.signal

class ActorCriticAgent(Agent):
    def __init__(self, state_size, action_size, name, params):
        super(ActorCriticAgent, self).__init__(params['data_dir'])
        super(ActorCriticAgent, self).setup_subject(human=False)

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
        self.name = name
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
        super(ActorCriticAgent, self).finish_subject(strategy, transfer_strategy)

    def save_reward(self, reward, trial_reward):
        self.epsilons.append(self.epsilon)
        self.rewards.append(reward)
        self.trial_rewards.append(trial_reward)

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

class A3CAgent(ActorCriticAgent):
    def __init__(self, state_size, action_size, name, params):
        super(A3CAgent, self).__init__(state_size, action_size, name, params)
        self.local_AC = self._build_model()
        self.name = name

    def finish_subject(self, strategy='A3C_LSTM', transfer_strategy='A3C_LSTM'):
        super(DAgent, self).finish_subject(strategy, transfer_strategy)

    def _build_model(self):
        cell_unit = 256
        trainer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        return AC_Network(self.state_size, self.action_size , self.name, trainer, cell_unit)

    def update_dynamic_epsilon(self, epsilon_threshold, new_epsilon, new_epsilon_decay):
        if self.epsilon < epsilon_threshold:
            self.epsilon = new_epsilon
            self.epsilon_decay = new_epsilon_decay

    def train(self, rollout, sess, gamma, r, REWARD_FACTOR):
        rollout = np.array(rollout)
        states = rollout[:, 0]
        actions = rollout[:, 1]
        rewards = rollout[:, 2]
        values = rollout[:, 5]

        # Here we take the rewards and values from the rollout, and use them to
        # generate the advantage and discounted returns.
        rewards_list = np.asarray(rewards.tolist() + [r]) * REWARD_FACTOR
        discounted_rewards = self.discounting(rewards_list, gamma)[:-1]

        # Advantage estimation
        # JS, P Moritz, S Levine, M Jordan, P Abbeel,
        # "High-dimensional continuous control using generalized advantage estimation."
        # arXiv preprint arXiv:1506.02438 (2015).
        values_list = np.asarray(values.tolist() + [r]) * REWARD_FACTOR
        advantages = rewards + gamma * values_list[1:] - values_list[:-1]
        discounted_advantages = self.discounting(advantages, gamma)

        # Update the global network using gradients from loss
        # Generate network statistics to periodically save
        # sess.run(self.local_AC.reset_state_op)

        rnn_state = self.local_AC.state_init
        feed_dict = {self.local_AC.target_v: discounted_rewards,
                     self.local_AC.inputs: np.vstack(states),
                     self.local_AC.actions: np.vstack(actions),
                     self.local_AC.advantages: discounted_advantages,
                     self.local_AC.state_in[0]: rnn_state[0],
                     self.local_AC.state_in[1]: rnn_state[1]}
        v_l, p_l, e_l, g_n, v_n, _ = sess.run([self.local_AC.value_loss,
                                               self.local_AC.policy_loss,
                                               self.local_AC.entropy,
                                               self.local_AC.grad_norms,
                                               self.local_AC.var_norms,
                                               self.local_AC.apply_grads],
                                              feed_dict=feed_dict)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        return v_l / len(rollout), p_l / len(rollout), e_l / len(rollout), g_n, v_n

    # Copies one set of variables to another.
    # Used to set worker network parameters to those of global network.
    def update_target_graph(self, from_scope, to_scope):
        from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
        to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

        op_holder = []
        for from_var, to_var in zip(from_vars, to_vars):
            op_holder.append(to_var.assign(from_var))
        return op_holder

    # Weighted random selection returns n_picks random indexes.
    # the chance to pick the index i is give by the weight weights[i].
    def weighted_pick(self, weights, n_picks,epsilon = 0.005):
        if np.random.rand(1) > epsilon:

            t = np.cumsum(weights)
            s = np.sum(weights)
            index = np.searchsorted(t, np.random.rand(n_picks) * s)
        else:
            index = random.randrange(self.action_size)
        return index

    # Discounting function used to calculate discounted returns.
    def discounting(self, x, gamma):
        return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

    # Normalization of inputs and outputs
    def norm(self, x, upper, lower=0.):
        return (x - lower) / max((upper - lower), 1e-12)



