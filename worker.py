import tensorflow as tf
import numpy as np
import gym
from ac_network import AC_Network

from session_manager import SessionManager
from gym_lock.settings_scenario import select_scenario
from agents.A3C_agent import A3CAgent

# Size of mini batches to run training on


class Worker():
    def __init__(self, name, s_size, a_size, trainer, model_path, global_episodes, env_name, seed, test, cell_units, params, testing_trial=False):
        self.name = "worker_" + str(name)
        self.number = name
        self.model_path = model_path
        self.trainer = trainer
        self.global_episodes = global_episodes
        self.increment = self.global_episodes.assign_add(1)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_mean_values = []
        self.summary_writer = tf.summary.FileWriter("train_" + str(self.number))
        self.is_test = test

        self.s_size = s_size
        self.a_size = a_size
        self.params = params
        self.MINI_BATCH = 30
        self.REWARD_FACTOR = 0.001

        # Create the local copy of the network and the tensorflow op to copy global parameters to local network
        #self.update_local_ops = update_target_graph('global', self.name)

        self.testing_trial = testing_trial
        if not self.testing_trial:
            self.scenario_name = params['train_scenario_name']
            self.attempt_limit = params['train_attempt_limit']
        else:
            self.scenario_name = params['test_scenario_name']
            self.attempt_limit = params['test_attempt_limit']

        self.scenario = select_scenario(self.scenario_name, params['use_physics'])
        env = gym.make(env_name)

        agent = A3CAgent(self.s_size, self.a_size,self.name,self.params)
        self.manager = SessionManager(env, agent,params,random_seed=seed)
        self.manager.update_scenario(self.scenario)
        self.manager.env.reward_mode = params['reward_mode']
        self.manager.env.use_physics = params['use_physics']
        self.trial_count = 0
        self.manager.env.seed(seed)

    def get_env(self):
        return self.manager.env

    def work(self, gamma, sess, coord, saver):
        self.manager.run_trial_a3c(sess = sess, global_episodes = self.global_episodes,
                                   number = self.number, testing_trial = self.testing_trial,
                                   params = self.params, coord = coord,attempt_limit = self.attempt_limit,
                                   scenario_name = self.scenario_name,trial_count = self.trial_count,is_test = self.is_test,
                                   a_size = self.a_size,MINI_BATCH = self.MINI_BATCH,gamma = gamma,
                                   episode_rewards = self.episode_rewards,episode_lengths = self.episode_lengths,
                                   episode_mean_values = self.episode_mean_values,summary_writer = self.summary_writer,
                                   name = self.name, saver = saver, model_path = self.model_path,REWARD_FACTOR = self.REWARD_FACTOR)


