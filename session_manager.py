
import time
import os
import copy
import numpy as np
import gym_lock.common as common

from gym_lock.settings_trial import select_random_trial, select_trial
from gym_lock.envs.arm_lock_env import ObservationSpace
import logger
from gym_lock.common import show_rewards


class SessionManager():

    env = None
    writer = None
    params = None
    completed_trials = []

    def __init__(self, env, agent, params):
        self.env = env
        self.params = params
        self.agent = agent

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
        self.env.observation_space = ObservationSpace(len(self.env.scenario.levers))
        self.env.solutions = self.env.scenario.solutions
        self.agent.logger.add_trial(trial_selected, scenario_name, self.env.scenario.solutions)
        self.agent.logger.cur_trial.add_attempt()

        if not multithreaded:
            print "INFO: New trial. There are {} unique solutions remaining.".format(len(self.env.scenario.solutions))

        self.env.reset()

        return trial_selected

    # code to run after both human and computer trials
    def run_trial_common_finish(self, trial_selected, test_trial):
        # todo: detect whether or not all possible successful paths were uncovered
        self.agent.finish_trial(test_trial)
        self.completed_trials.append(copy.deepcopy(trial_selected))
        self.env.completed_solutions = []
        self.env.cur_action_seq = []

    # code to run a human subject
    def run_trial_human(self, scenario_name, action_limit, attempt_limit, specified_trial=None, verify=False, test_trial=False):
        self.env.human_agent = True
        trial_selected = self.run_trial_common_setup(scenario_name, action_limit, attempt_limit, specified_trial)

        obs_space = None
        while self.env.attempt_count < attempt_limit and self.env.determine_trial_success() is False:
            self.env.render(self.env)
            # acknowledge any acks that may have occurred (action executed, attempt ended, etc)
            env_reset = self.update_acks()
            # used to verify simulator and fsm states are always the same (they should be)
            if verify:
                obs_space = self.verify_fsm_matches_simulator(obs_space)

        self.run_trial_common_finish(trial_selected, test_trial)

    # code to run a computer trial
    def run_trial_dqn(self, scenario_name, action_limit, attempt_limit, trial_count, iter_num, test_trial=False, specified_trial=None, fig=None, fig_update_rate=100):
        self.env.human_agent = False
        trial_selected = self.run_trial_common_setup(scenario_name, action_limit, attempt_limit, specified_trial)

        state = self.env.reset()
        state = np.reshape(state, [1, self.agent.state_size])

        save_dir = self.agent.writer.subject_path + '/models'

        print('scenario_name: {}, trial_count: {}, trial_name: {}'.format(scenario_name, trial_count, trial_selected))

        trial_reward = 0
        attempt_reward = 0

        while True:
            # end if attempt limit reached
            if self.env.attempt_count >= attempt_limit:
                break
            # trial is success and not forcing agent to use all attempts
            elif self.params['full_attempt_limit'] is False and self.agent.logger.cur_trial.success is True:
                break

            # self.env.render()

            action_idx = self.agent.act(state)
            # convert idx to Action object (idx -> str -> Action)
            action = self.env.action_map[self.env.action_space[action_idx]]
            next_state, reward, done, opt = self.env.step(action)
            """See the temperature of solutions
            count1 = 0
            reward_i = 0
            for i in range(len(manager.agent.memory)):
                action_i = manager.agent.memory[i][1]
                action_j = manager.env.action_map[manager.env.action_space[action_i]]
                reward_j, success = determine_reward(manager.env, action_j, 'negative_change_state_partial_action_seq_solution_multiplier')
                count1+=1
                reward_i += reward_j
            num_ = len(manager.env.completed_solutions)
            print "solution: ", num_, " reward_ave: ", reward_i/count1
            """



            next_state = np.reshape(next_state, [1, self.agent.state_size])

            # THIS OVERRIDES done coming from the environment based on whether or not
            # we are allowing the agent to move to the next trial after finding all solutions
            if self.params['full_attempt_limit'] and self.env.attempt_count < attempt_limit:
                done = False


            self.agent.remember(state, action_idx, reward, next_state, done,)
            # agent.remember(state, action_idx, trial_reward, next_state, done)
            # self.env.render()

            env_reset = self.update_acks()
            trial_reward += reward
            attempt_reward += reward
            state = next_state

            if env_reset:
                self.print_update(iter_num, trial_count, scenario_name, self.env.attempt_count, self.env.attempt_limit, attempt_reward, trial_reward, self.agent.epsilon)
                print(self.agent.logger.cur_trial.attempt_seq[-1].action_seq)
                self.agent.logger.cur_trial.attempt_seq[-1].reward = attempt_reward
                self.agent.save_reward(attempt_reward, trial_reward)
                attempt_reward = 0

                # update figure
                if fig is not None and self.env.attempt_count % fig_update_rate == 0:
                    show_rewards(self.agent.rewards, self.agent.epsilons, fig)

            # save agent's model
            # if self.env.attempt_count % (self.env.attempt_limit/2) == 0 or self.env.attempt_count == self.env.attempt_limit or self.env.logger.cur_trial.success is True:
            if self.env.attempt_count == 0 or self.env.attempt_count == self.env.attempt_limit:
                self.agent.save_agent(save_dir, test_trial, iter_num, trial_count, self.env.attempt_count)

            # replay to learn
            if len(self.agent.memory) > self.agent.batch_size:
                self.agent.replay()

        self.dqn_trial_sanity_checks()

        self.agent.logger.cur_trial.trial_reward = trial_reward
        self.run_trial_common_finish(trial_selected, test_trial=test_trial)
        self.agent.trial_switch_points.append(len(self.agent.rewards))
        self.agent.average_trial_rewards.append(trial_reward / self.env.attempt_count)

    # code to run a computer trial ddqn priority memory replay
    def run_trial_ddqn_prority(self, scenario_name, action_limit, attempt_limit, trial_count, iter_num, test_trial=False, specified_trial=None, fig=None, fig_update_rate=100):
        self.env.human_agent = False
        trial_selected = self.run_trial_common_setup(scenario_name, action_limit, attempt_limit, specified_trial)

        state = self.env.reset()
        state = np.reshape(state, [1, self.agent.state_size])

        save_dir = self.agent.writer.subject_path + '/models'

        print('scenario_name: {}, trial_count: {}, trial_name: {}'.format(scenario_name, trial_count, trial_selected))

        trial_reward = 0
        attempt_reward = 0
        train_step = 0
        while True:
            # end if attempt limit reached
            if self.env.attempt_count >= attempt_limit:
                break
            # trial is success and not forcing agent to use all attempts
            elif self.params['full_attempt_limit'] is False and self.agent.logger.cur_trial.success is True:
                break

            # self.env.render()

            action_idx = self.agent.act(state)
            # convert idx to Action object (idx -> str -> Action)
            action = self.env.action_map[self.env.action_space[action_idx]]
            next_state, reward, done, opt = self.env.step(action)

            next_state = np.reshape(next_state, [1, self.agent.state_size])

            # THIS OVERRIDES done coming from the environment based on whether or not
            # we are allowing the agent to move to the next trial after finding all solutions
            if self.params['full_attempt_limit'] and self.env.attempt_count < attempt_limit:
                done = False

            self.agent._remember(state, action_idx, reward, next_state, done)
            # agent.remember(state, action_idx, trial_reward, next_state, done)
            # self.env.render()

            env_reset = self.update_acks()
            trial_reward += reward
            attempt_reward += reward
            state = next_state

            if env_reset:
                self.print_update(iter_num, trial_count, scenario_name, self.env.attempt_count, self.env.attempt_limit, attempt_reward, trial_reward, self.agent.epsilon)
                print(self.agent.logger.cur_trial.attempt_seq[-1].action_seq)
                self.agent.logger.cur_trial.attempt_seq[-1].reward = attempt_reward
                self.agent.save_reward(attempt_reward, trial_reward)
                attempt_reward = 0

                # update figure
                if fig is not None and self.env.attempt_count % fig_update_rate == 0:
                    show_rewards(self.agent.rewards, self.agent.epsilons, fig)

            # save agent's model
            # if self.env.attempt_count % (self.env.attempt_limit/2) == 0 or self.env.attempt_count == self.env.attempt_limit or self.env.logger.cur_trial.success is True:
            if self.env.attempt_count == 0 or self.env.attempt_count == self.env.attempt_limit:
                self.agent.save_agent(save_dir, test_trial, iter_num, trial_count, self.env.attempt_count)

            # replay to learn
            if train_step > self.agent.batch_size:
                self.agent.replay()
            train_step += 1
        self.dqn_trial_sanity_checks()

        self.agent.logger.cur_trial.trial_reward = trial_reward
        self.run_trial_common_finish(trial_selected, test_trial=test_trial)
        self.agent.trial_switch_points.append(len(self.agent.rewards))
        self.agent.average_trial_rewards.append(trial_reward / self.env.attempt_count)

    # code to run a computer trial using q-tables (UNFINISHED)
    def run_trial_qtable(self, scenario_name, action_limit, attempt_limit, trial_count, iter_num, testing=False, specified_trial=None):
        self.env.human_agent = False
        trial_selected = self.run_trial_common_setup(scenario_name, action_limit, attempt_limit, specified_trial)

        state = self.env.reset()
        state = np.reshape(state, [1, self.agent.state_size])

        save_dir = self.writer.subject_path + '/models'

        print('scenario_name: {}, trial_count: {}, trial_name: {}'.format(scenario_name, trial_count, trial_selected))

        trial_reward = 0
        attempt_reward = 0
        while self.env.attempt_count < attempt_limit and self.agent.logger.cur_trial.success is False:
            # self.env.render()

            action_idx = self.agent.action(state, train=True)
            # convert idx to Action object (idx -> str -> Action)
            action = self.env.action_map[self.env.action_space[action_idx]]
            next_state, reward, done, opt = self.env.step(action)

            next_state = np.reshape(next_state, [1, self.agent.state_size])

            self.agent.update(state, action_idx, reward, next_state)
            # self.env.render()
            trial_reward += reward
            attempt_reward += reward
            state = next_state

            if done:
                self.agent.update_epsilon()

            if opt['env_reset']:
                self.print_update(iter_num, trial_count, scenario_name, self.env.attempt_count, self.env.attempt_limit, attempt_reward, trial_reward, agent.epsilon)
                print(self.agent.logger.cur_trial.attempt_seq[-1].action_seq)
                self.agent.save_reward(attempt_reward, trial_reward)
                attempt_reward = 0

        self.run_trial_common_finish(trial_selected)
        self.agent.trial_switch_points.append(len(self.agent.rewards))
        self.agent.average_trial_rewards.append(trial_reward / attempt_limit)

    def update_scenario(self, scenario):
        self.env.scenario = scenario
        self.env.solutions = scenario.solutions
        self.env.completed_solutions = []
        self.env.cur_action_seq = []

    def update_attempt(self):
        reset = False
        # above the allowed number of actions, need to increment the attempt count and reset the simulator
        if self.env.action_count >= self.env.action_limit:

            self.env.attempt_count += 1

            attempt_success = self.env.determine_unique_solution()
            if attempt_success:
                self.env.completed_solutions.append(self.env.cur_action_seq)

            self.agent.logger.cur_trial.finish_attempt(attempt_success=attempt_success, results=self.env.results)

            # update the user about their progress
            trial_finished, pause = self.update_user(attempt_success, multithreaded=False)

            # pauses if the human user unlocked the door but didn't push on the door
            if self.env.use_physics and self.env.human_agent and pause:
                # pause for 4 sec to allow user to view lock
                t_end = time.time() + 4
                while time.time() < t_end:
                    self.env.render(self.env)
                    self.env.update_state_machine()

            # reset attempt if the trial isn't finished or if we are running to the full
            # attempt limit. If the full attempt is not used, trial will advance
            # after finding all solutions
            if not trial_finished or self.env.full_attempt_limit is not False:
                self.add_attempt()

            self.env.reset()
            reset = True

        return reset

    def update_user(self, attempt_success, multithreaded=False):
        pause = False
        num_solutions_remaining = len(self.env.solutions) - len(self.env.completed_solutions)
        # continue or end trial
        if self.env.determine_trial_success():
            if not multithreaded:
                print "INFO: You found all of the solutions. "
            # todo: should we mark that the trial is finished even though the attempt_limit
            # todo: may not be reached?
            trial_finished = True
            pause = True            # pause if they open the door
        elif self.env.attempt_count < self.env.attempt_limit:
            # alert user to the number of solutions remaining
            if attempt_success is True:
                if not multithreaded:
                    print "INFO: You found a solution. There are {} unique solutions remaining.".format(num_solutions_remaining)
                pause = True            # pause if they open the door
            else:
                if not multithreaded and self.env.human_agent:
                    print "INFO: Ending attempt. Action limit reached. There are {} unique solutions remaining. You have {} attempts remaining.".format(num_solutions_remaining, self.env.attempt_limit - self.env.attempt_count)
                # pause if the door lock is missing and the agent is a human
                if self.env.human_agent and self.env.get_state()['OBJ_STATES']['door_lock'] == common.ENTITY_STATES['DOOR_UNLOCKED']:
                    pause = True
            trial_finished = False
        else:
            if not multithreaded:
                print "INFO: Ending trial. Attempt limit reached. You found {} unique solutions".format(len(self.env.completed_solutions))
            trial_finished = True

        return trial_finished, pause

    def update_acks(self):
        env_reset = False
        if not self.env.action_ack:
            if self.agent.logger.cur_trial.cur_attempt is None:
                print 'cur_attempt is none...shouldnt be'
            self.agent.logger.cur_trial.cur_attempt.add_action(self.env.action.name, self.env.action.start_time)
            self.env.action_ack = True
        if not self.env.action_finish_ack:
            self.agent.logger.cur_trial.cur_attempt.finish_action(self.env.action.end_time)
            self.env.action_finish_ack = True
            env_reset = self.update_attempt()

        return env_reset

    def set_action_limit(self, action_limit):
        self.env.action_limit = action_limit

    def add_attempt(self):
        self.env.cur_action_seq = []
        self.agent.logger.cur_trial.add_attempt()

    def print_update(self, iter_num, trial_num, scenario_name, episode, episode_max, a_reward, t_reward, epsilon):
        print("ID: {}, iter {}, trial {}, scenario {}, episode: {}/{}, attempt_reward {}, trial_reward {}, e: {:.2}".format(self.agent.subject_id, iter_num, trial_num, scenario_name, episode, episode_max, a_reward, t_reward, epsilon))

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

    def dqn_trial_sanity_checks(self):
        try:
            if len(self.agent.trial_switch_points) > 0:
                assert(len(self.agent.logger.cur_trial.attempt_seq) == len(self.agent.rewards) - self.agent.trial_switch_points[-1])
                reward_agent = self.agent.rewards[self.agent.trial_switch_points[-1]:]
            else:
                assert(len(self.agent.logger.cur_trial.attempt_seq) == len(self.agent.rewards))
                reward_agent = self.agent.rewards[:]
            reward_seq = []
            for attempt in self.agent.logger.cur_trial.attempt_seq:
                reward_seq.append(attempt.reward)
            assert(reward_seq == reward_agent)

            if len(self.agent.logger.trial_seq) > 0:
                reward_seq_prev = []
                for attempt in self.agent.logger.trial_seq[-1].attempt_seq:
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