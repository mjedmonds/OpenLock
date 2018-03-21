
import time
import os
import copy
import numpy as np

from gym_lock.settings_trial import select_random_trial, select_trial
from gym_lock.envs.arm_lock_env import ObservationSpace
import logger
from gym_lock.common import show_rewards


class SessionManager():

    env = None
    writer = None
    params = None
    completed_trials = []

    def __init__(self, env, params):
        self.env = env
        self.params = params

        self.env.logger.use_physics = env.use_physics
        self.completed_trials = []

    # code to run before human and computer trials
    def run_trial_common_setup(self, scenario_name, action_limit, attempt_limit, specified_trial=None, multithreaded=False):
        # setup trial
        self.env.attempt_count = 0
        self.env.attempt_limit = attempt_limit
        self.set_action_limit(action_limit)
        # select trial
        if specified_trial is None:
            trial_selected, lever_configs = self.get_trial(scenario_name, self.completed_trials)
            if trial_selected is None:
                if not multithreaded:
                    print('WARNING: no more trials available. Resetting completed_trials.')
                self.completed_trials = []
                trial_selected, lever_configs = self.get_trial(scenario_name, self.completed_trials)
        else:
            trial_selected, lever_configs = select_trial(specified_trial)
        self.env.scenario.set_lever_configs(lever_configs)

        self.env.logger.add_trial(trial_selected, scenario_name, self.env.scenario.solutions)
        self.env.logger.cur_trial.add_attempt()

        if not multithreaded:
            print "INFO: New trial. There are {} unique solutions remaining.".format(len(self.env.scenario.solutions))

        self.env.reset()

        return trial_selected

    # code to run after both human and computer trials
    def run_trial_common_finish(self, trial_selected):
        # todo: detect whether or not all possible successful paths were uncovered
        self.env.logger.finish_trial()
        self.completed_trials.append(copy.deepcopy(trial_selected))

    # code to run a human subject
    def run_trial_human(self, scenario_name, action_limit, attempt_limit, specified_trial=None, verify=False):
        self.env.human_agent = True
        trial_selected = self.run_trial_common_setup(scenario_name, action_limit, attempt_limit, specified_trial)

        obs_space = None
        while self.env.attempt_count < attempt_limit and self.env.logger.cur_trial.success is False:
            self.env.render(self.env)
            # used to verify simulator and fsm states are always the same (they should be)
            if verify:
                obs_space = self.verify_fsm_matches_simulator(obs_space)

        self.run_trial_common_finish(trial_selected)

    # code to run a computer trial
    def run_trial_dqn(self, agent, scenario_name, action_limit, attempt_limit, trial_count, iter_num, testing=False, specified_trial=None, fig=None, fig_update_rate=100):
        self.env.human_agent = False
        trial_selected = self.run_trial_common_setup(scenario_name, action_limit, attempt_limit, specified_trial)

        state = self.env.reset()
        state = np.reshape(state, [1, agent.state_size])

        save_dir = self.writer.subject_path + '/models'

        print('scenario_name: {}, trial_count: {}, trial_name: {}'.format(scenario_name, trial_count, trial_selected))

        trial_reward = 0
        attempt_reward = 0

        while True:
            # end if attempt limit reached
            if self.env.attempt_count >= attempt_limit:
                break
            # trial is success and not forcing agent to use all attempts
            elif self.params['full_attempt_limit'] is False and self.env.logger.cur_trial.success is True:
                break

            # self.env.render()

            action_idx = agent.act(state)
            # convert idx to Action object (idx -> str -> Action)
            action = self.env.action_map[self.env.action_space[action_idx]]
            next_state, reward, done, opt = self.env.step(action)

            next_state = np.reshape(next_state, [1, agent.state_size])

            # THIS OVERRIDES done coming from the environment based on whether or not
            # we are allowing the agent to move to the next trial after finding all solutions
            if self.params['full_attempt_limit'] and self.env.attempt_count < attempt_limit:
                done = False

            agent.remember(state, action_idx, reward, next_state, done)
            # agent.remember(state, action_idx, trial_reward, next_state, done)
            # self.env.render()

            trial_reward += reward
            attempt_reward += reward
            state = next_state

            if opt['env_reset']:
                self.print_update(iter_num, trial_count, scenario_name, self.env.attempt_count, self.env.attempt_limit, attempt_reward, trial_reward, agent.epsilon)
                print(self.env.logger.cur_trial.attempt_seq[-1].action_seq)
                self.env.logger.cur_trial.attempt_seq[-1].reward = attempt_reward
                agent.save_reward(attempt_reward, trial_reward)
                attempt_reward = 0

                # update figure
                if fig is not None and self.env.attempt_count % fig_update_rate == 0:
                    show_rewards(agent.rewards, agent.epsilons, fig)

            # save agent's model
            # if self.env.attempt_count % (self.env.attempt_limit/2) == 0 or self.env.attempt_count == self.env.attempt_limit or self.env.logger.cur_trial.success is True:
            if self.env.attempt_count == 0 or self.env.attempt_count == self.env.attempt_limit:
                agent.save_agent(agent, save_dir, testing, iter_num, trial_count)

            # replay to learn
            if len(agent.memory) > agent.batch_size:
                agent.replay()

        self.dqn_trial_sanity_checks(agent)

        self.env.logger.cur_trial.trial_reward = trial_reward
        self.run_trial_common_finish(trial_selected)
        agent.trial_switch_points.append(len(agent.rewards))
        agent.average_trial_rewards.append(trial_reward / self.env.attempt_count)

        return agent



    # code to run a computer trial
    def run_trial_qtable(self, agent, scenario_name, action_limit, attempt_limit, trial_count, iter_num, testing=False, specified_trial=None):
        self.env.human_agent = False
        trial_selected = self.run_trial_common_setup(scenario_name, action_limit, attempt_limit, specified_trial)

        state = self.env.reset()
        state = np.reshape(state, [1, agent.state_size])

        save_dir = self.writer.subject_path + '/models'

        print('scenario_name: {}, trial_count: {}, trial_name: {}'.format(scenario_name, trial_count, trial_selected))

        trial_reward = 0
        attempt_reward = 0
        while self.env.attempt_count < attempt_limit and self.env.logger.cur_trial.success is False:
            # self.env.render()

            action_idx = agent.action(state, train=True)
            # convert idx to Action object (idx -> str -> Action)
            action = self.env.action_map[self.env.action_space[action_idx]]
            next_state, reward, done, opt = self.env.step(action)

            next_state = np.reshape(next_state, [1, agent.state_size])

            agent.update(state, action_idx, reward, next_state)
            # self.env.render()
            trial_reward += reward
            attempt_reward += reward
            state = next_state

            if done:
                agent.update_epsilon()

            if opt['env_reset']:
                self.print_update(iter_num, trial_count, scenario_name, self.env.attempt_count, self.env.attempt_limit, attempt_reward, trial_reward, agent.epsilon)
                print(self.env.logger.cur_trial.attempt_seq[-1].action_seq)
                agent.save_reward(attempt_reward, trial_reward)
                attempt_reward = 0

        self.run_trial_common_finish(trial_selected)
        agent.trial_switch_points.append(len(agent.rewards))
        agent.average_trial_rewards.append(trial_reward / attempt_limit)

        return agent

    def update_scenario(self, scenario):
        self.env.scenario = scenario
        self.env.observation_space = ObservationSpace(len(scenario.levers))

    def set_action_limit(self, action_limit):
        self.env.action_limit = action_limit

    def print_update(self, iter_num, trial_num, scenario_name, episode, episode_max, a_reward, t_reward, epsilon):
        print("ID: {}, iter {}, trial {}, scenario {}, episode: {}/{}, attempt_reward {}, trial_reward {}, e: {:.2}".format(self.env.logger.subject_id, iter_num, trial_num, scenario_name, episode, episode_max, a_reward, t_reward, epsilon))

    def verify_fsm_matches_simulator(self, obs_space):
        if obs_space is None:
            obs_space = ObservationSpace(len(self.env.world_def.get_levers()))
        state, labels = obs_space.create_discrete_observation_from_simulator(self.env)
        fsm_state, fsm_labels = obs_space.create_discrete_observation_from_fsm(self.env)
        try:
            assert(state == fsm_state)
            assert(labels == fsm_labels)
        except AssertionError:
            print 'FSM does not match simulator data'
            print state
            print fsm_state
            print labels
            print fsm_labels
        return obs_space

    def dqn_trial_sanity_checks(self, agent):
        try:
            if len(agent.trial_switch_points) > 0:
                assert(len(self.env.logger.cur_trial.attempt_seq) == len(agent.rewards) - agent.trial_switch_points[-1])
                reward_agent = agent.rewards[agent.trial_switch_points[-1]:]
            else:
                assert(len(self.env.logger.cur_trial.attempt_seq) == len(agent.rewards))
                reward_agent = agent.rewards[:]
            reward_seq = []
            for attempt in self.env.logger.cur_trial.attempt_seq:
                reward_seq.append(attempt.reward)
            assert(reward_seq == reward_agent)

            if len(self.env.logger.trial_seq) > 0:
                reward_seq_prev = []
                for attempt in self.env.logger.trial_seq[-1].attempt_seq:
                    reward_seq_prev.append(attempt.reward)
                assert(reward_seq != reward_seq_prev)

        except AssertionError:
            print('reward len does not match attempt len')

    @staticmethod
    def get_trial(name, completed_trials=None):
        # select a random trial and add it to the scenario
        if name != 'CE4' and name != 'CC4':
            # trials 1-6 have 3 levers for CC3/CE3
            trial, configs = select_random_trial(completed_trials, 1, 6)
        else:
            # trials 7-11 have 4 levers for CC4/CE4
            trial, configs = select_random_trial(completed_trials, 7, 11)

        return trial, configs