
import time
import os
import copy
import numpy as np
import gym_lock.common as common

from gym_lock.settings_trial import select_random_trial, select_trial, get_trial
from gym_lock.envs.arm_lock_env import ObservationSpace
import logger
from gym_lock.common import show_rewards
from matplotlib import pyplot as plt
import tensorflow as tf

class SessionManager():
    """
    Manage an environment and an agent in order to enforce a common routine across all scenarios and agents.

    The class provides an interface for different sessions with different environments and agents
    to operate the same way from a high level perspective. It has an environment and an agent
    as class variables, which it coordinates using several methods, some to be used for all
    scenarios and agents, and others for running trials with specific agents.
    """

    env = None
    writer = None
    params = None
    completed_trials = []

    def __init__(self, env, agent=None, params=None, random_seed = None):
        """
        Set the instance's env, agent, and params. Initialize completed_trials to the empty list.

        :param env: environment managed by this session (e.g. an object returned by gym.make())
        :param agent: agent managed by this session (some Agent object)
        :param params: dictionary of other parameters
        :param random_seed: default: None
        """
        self.env = env
        self.params = params
        self.agent = agent
        self.random_seed = random_seed

        self.completed_trials = []

    # code to run before human and computer trials
    def run_trial_common_setup(self, scenario_name, action_limit, attempt_limit, specified_trial=None, multithreaded=False):
        """
        Set the env class variables and select a trial (specified if provided, otherwise a random trial from the scenario name).

        This method should be called before running human and computer trials.
        Returns the trial selected (string).

        :param scenario_name: name of scenario (e.g. those defined in settings_trial.PARAMS)
        :param action_limit: number of actions permitted
        :param attempt_limit: number of attempts permitted
        :param specified_trial: optional specified trial. If none, get_trial is used to select trial
        :param multithreaded:
        :return: the selected_trial as returned by get_trial or select_trial
        """
        # setup trial
        self.env.attempt_count = 0
        self.env.attempt_limit = attempt_limit
        self.set_action_limit(action_limit)
        # select trial
        if specified_trial is None:
            trial_selected, lever_configs = get_trial(scenario_name, self.completed_trials)
            if trial_selected is None:
                if not multithreaded:
                    print('WARNING: no more trials available. Resetting completed_trials.')
                self.completed_trials = []
                trial_selected, lever_configs = get_trial(scenario_name, self.completed_trials)
        else:
            trial_selected, lever_configs = select_trial(specified_trial)

        self.env.scenario.set_lever_configs(lever_configs)
        self.env.observation_space = ObservationSpace(len(self.env.scenario.levers))
        self.env.solutions = self.env.scenario.solutions
        self.agent.logger.add_trial(trial_selected, scenario_name, self.env.scenario.solutions, self.random_seed)
        self.agent.logger.cur_trial.add_attempt()

        if not multithreaded:
            print("INFO: New trial. There are {} unique solutions remaining.".format(len(self.env.scenario.solutions)))

        self.env.reset()

        return trial_selected

    # code to run after both human and computer trials
    def run_trial_common_finish(self, trial_selected, test_trial):
        """
        Reset variables after finishing trial and call agent.finish_trial(). Add finished trial to completed_trials.

        :param trial_selected: trial to add to completed_trials
        :param test_trial: trial used for agent.finish_trial()
        :return: Nothing
        """
        # todo: detect whether or not all possible successful paths were uncovered
        self.agent.finish_trial(test_trial)
        self.completed_trials.append(copy.deepcopy(trial_selected))
        self.env.completed_solutions = []
        self.env.cur_action_seq = []

    # code to run a human subject
    def run_trial_human(self, scenario_name, action_limit, attempt_limit, specified_trial=None, verify=False, test_trial=False):
        """
        Run trial for a human subject.

        :param scenario_name: name of scenario (e.g. those defined in settings_trial.PARAMS)
        :param action_limit: number of actions permitted
        :param attempt_limit: number of attempts permitted
        :param specified_trial: optional specified trial
        :param verify: flag to indicate whether or not to call verify_fsm_matches_simulator()
        :param test_trial: default: False
        :return: Nothing
        """
        self.env.human_agent = True
        trial_selected = self.run_trial_common_setup(scenario_name, action_limit, attempt_limit, specified_trial)

        obs_space = None
        while not self.determine_human_trial_finished(attempt_limit):
            self.env.render(self.env)
            # acknowledge any acks that may have occurred (action executed, attempt ended, etc)
            env_reset = self.finish_action()
            # used to verify simulator and fsm states are always the same (they should be)
            if verify:
                obs_space = self.verify_fsm_matches_simulator(obs_space)

        self.run_trial_common_finish(trial_selected, test_trial)

    # code to run a computer trial
    def run_trial_dqn(self, scenario_name, action_limit, attempt_limit, trial_count, iter_num, test_trial=False, specified_trial=None, fig=None, fig_update_rate=100):
        """
        Run a computer trial.

        :param scenario_name:
        :param action_limit:
        :param attempt_limit:
        :param trial_count:
        :param iter_num:
        :param test_trial:
        :param specified_trial:
        :param fig:
        :param fig_update_rate:
        :return:
        """
        self.env.human_agent = False
        trial_selected = self.run_trial_common_setup(scenario_name, action_limit, attempt_limit, specified_trial)

        state = self.env.reset()
        state = np.reshape(state, [1, self.agent.state_size])

        save_dir = self.agent.writer.subject_path + '/models'

        print('scenario_name: {}, trial_count: {}, trial_name: {}'.format(scenario_name, trial_count, trial_selected))

        trial_reward = 0
        attempt_reward = 0

        while not self.determine_computer_trial_finished(attempt_limit):

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

            self.agent.remember(state, action_idx, reward, next_state, done)
            # agent.remember(state, action_idx, trial_reward, next_state, done)
            # self.env.render()

            env_reset = self.finish_action()
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
        """
        Run a computer trial (ddqn priority memory replay).

        :param scenario_name:
        :param action_limit:
        :param attempt_limit:
        :param trial_count:
        :param iter_num:
        :param test_trial:
        :param specified_trial:
        :param fig:
        :param fig_update_rate:
        :return:
        """
        self.env.human_agent = False
        trial_selected = self.run_trial_common_setup(scenario_name, action_limit, attempt_limit, specified_trial)

        state = self.env.reset()
        state = np.reshape(state, [1, self.agent.state_size])

        save_dir = self.agent.writer.subject_path + '/models'

        print(('scenario_name: {}, trial_count: {}, trial_name: {}'.format(scenario_name, trial_count, trial_selected)))

        trial_reward = 0
        attempt_reward = 0
        train_step = 0
        while not self.determine_computer_trial_finished(attempt_limit):

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

            env_reset = self.finish_action()
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

    # code for a3c
    def run_trial_a3c(self, sess, global_episodes, number, testing_trial, params, coord,attempt_limit,
                      scenario_name, trial_count,is_test,a_size,MINI_BATCH,gamma,episode_rewards,episode_lengths,episode_mean_values,
                      summary_writer, name, saver, model_path, REWARD_FACTOR, fig=None):
        """
        Run a trial for a3c.

        :param sess:
        :param global_episodes:
        :param number:
        :param testing_trial:
        :param params:
        :param coord:
        :param attempt_limit:
        :param scenario_name:
        :param trial_count:
        :param is_test:
        :param a_size:
        :param MINI_BATCH:
        :param gamma:
        :param episode_rewards:
        :param episode_lengths:
        :param episode_mean_values:
        :param summary_writer:
        :param name:
        :param saver:
        :param model_path:
        :param REWARD_FACTOR:
        :param fig:
        :return:
        """
        self.env.human_agent = False
        episode_count = sess.run(global_episodes)
        increment = global_episodes.assign_add(1)
        total_steps = 0
        print("Starting worker " + str(number))
        with sess.as_default(), sess.graph.as_default():

            sess.run(self.agent.update_target_graph('global', name))
            episode_buffer = []
            episode_mini_buffer = []
            episode_values = []
            episode_states = []
            episode_reward = 0
            attempt_reward = 0
            episode_step_count = 0

            if not testing_trial:
                trial_selected = self.run_trial_common_setup(params['train_scenario_name'],
                                                             params['train_action_limit'],
                                                             params['train_attempt_limit'],
                                                             multithreaded=True)

            else:
                trial_selected = self.run_trial_common_setup(params['test_scenario_name'],
                                                             params['test_action_limit'],
                                                             params['test_attempt_limit'],
                                                             specified_trial='trial7', multithreaded=True)
            if name == 'worker_0':
                print('scenario_name: {}, trial_count: {}, trial_name: {}'.format(scenario_name, trial_count, trial_selected))
            terminal = False
            state = self.env.reset()
            rnn_state = self.agent.local_AC.state_init

            while not coord.should_stop():
                # end if attempt limit reached
                sess.run(self.agent.update_target_graph('global', name))
                if self.env.attempt_count >= attempt_limit or (self.params['full_attempt_limit'] is False and self.agent.logger.cur_trial.success is True):

                    episode_buffer = []
                    episode_mini_buffer = []
                    episode_values = []
                    episode_states = []
                    episode_reward = 0
                    attempt_reward = 0
                    episode_step_count = 0

                    if not testing_trial:
                        trial_selected = self.run_trial_common_setup(params['train_scenario_name'],
                                                                             params['train_action_limit'],
                                                                             params['train_attempt_limit'],
                                                                             multithreaded=True)

                    else:
                        trial_selected = self.run_trial_common_setup(params['test_scenario_name'],
                                                                             params['test_action_limit'],
                                                                             params['test_attempt_limit'],
                                                                             specified_trial='trial7', multithreaded=True)
                    if name == 'worker_0':
                        print('scenario_name: {}, trial_count: {}, trial_name: {}'.format(scenario_name, trial_count, trial_selected))
                    terminal = False
                    state = self.env.reset()
                    rnn_state = self.agent.local_AC.state_init

                # Run an episode
                while not terminal and not self.determine_computer_trial_finished(attempt_limit):

                    episode_states.append(state)
                    if is_test:
                        self.env.render()

                    # Get preferred action distribution
                    a_dist, v, rnn_state = sess.run(
                        [self.agent.local_AC.policy, self.agent.local_AC.value, self.agent.local_AC.state_out],
                        feed_dict={self.agent.local_AC.inputs: [state],
                                   self.agent.local_AC.state_in[0]: rnn_state[0],
                                   self.agent.local_AC.state_in[1]: rnn_state[1]})

                    a0 = self.agent.weighted_pick(a_dist[0], 1, self.agent.epsilon)  # Use stochastic distribution sampling
                    #if is_test:
                    #    a0 = np.argmax(a_dist[0])  # Use maximum when testing
                    a = np.zeros(a_size)
                    a[a0] = 1
                    action_idx = np.argmax(a)
                    action = self.env.action_map[self.env.action_space[action_idx]]

                    next_state, reward, terminal, opt = self.env.step(action)
                    episode_reward += reward
                    attempt_reward += reward


                    # THIS OVERRIDES done coming from the environment based on whether or not
                    # we are allowing the agent to move to the next trial after finding all solutions
                    if self.params['full_attempt_limit'] and self.env.attempt_count < attempt_limit:
                        terminal = False


                    episode_buffer.append([state,a,reward,next_state,terminal,v[0,0]])
                    episode_mini_buffer.append([state,a,reward,next_state,terminal,v[0,0]])

                    episode_values.append(v[0,0])


                    # Train on mini batches from episode
                    if len(episode_mini_buffer) == MINI_BATCH :
                        v1 = sess.run([self.agent.local_AC.value],
                                      feed_dict={self.agent.local_AC.inputs: [state],
                                                 self.agent.local_AC.state_in[0]: rnn_state[0],
                                                 self.agent.local_AC.state_in[1]: rnn_state[1]})
                        v_l, p_l, e_l, g_n, v_n = self.agent.train(episode_mini_buffer, sess, gamma, v1[0][0], REWARD_FACTOR)
                        episode_mini_buffer = []

                    env_reset = self.update_attempt(multithread = True)
                    state = next_state
                    total_steps += 1
                    episode_step_count += 1

                    if env_reset:
                        self.agent.logger.cur_trial.attempt_seq[-1].reward = attempt_reward
                        self.agent.save_reward(attempt_reward, episode_reward)
                        attempt_reward = 0

                self.agent.logger.cur_trial.trial_reward = attempt_limit
                self.run_trial_common_finish(trial_selected, test_trial=False)
                self.agent.trial_switch_points.append(len(self.agent.rewards))
                self.agent.average_trial_rewards.append(attempt_reward / self.env.attempt_count)

                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_step_count)

                if episode_count % 100 == 0 and not episode_count % 1000 == 0 and not is_test:
                    mean_reward = np.mean(episode_rewards[-5:])
                    mean_length = np.mean(episode_lengths[-5:])
                    mean_value = np.mean(episode_mean_values[-5:])
                    summary = tf.Summary()

                    # summary.text.add(tag='Scenario name', simple_value=str(self.env.scenario.name))
                    # summary.text.add(tag='trial count', simple_value=str(trial_count))
                    # summary.text.add(tag='trial name', simple_value=str(trial_selected))
                    summary.value.add(tag='Perf/Reward', simple_value=float(mean_reward))
                    summary.value.add(tag='Perf/Length', simple_value=float(mean_length))
                    summary.value.add(tag='Perf/Value', simple_value=float(mean_value))
                    summary.value.add(tag='Losses/Value Loss', simple_value=float(v_l))
                    summary.value.add(tag='Losses/Policy Loss', simple_value=float(p_l))
                    summary.value.add(tag='Losses/Entropy', simple_value=float(e_l))
                    summary.value.add(tag='Losses/Grad Norm', simple_value=float(g_n))
                    summary.value.add(tag='Losses/Var Norm', simple_value=float(v_n))
                    summary_writer.add_summary(summary, episode_count)

                    summary_writer.flush()

                if name == 'worker_0':
                    if episode_count % 20 == 0 and not is_test:
                        saver.save(sess, model_path + '/model-' + str(episode_count) + '.cptk')
                    # creating the figure
                    if episode_count % 20 == 0 :
                        saver.save(sess, model_path + '/model-' + str(episode_count) + '.cptk')
                        if fig == None:
                            fig = plt.figure()
                            fig.set_size_inches(12, 6)
                        else:
                            show_rewards(self.agent.rewards, self.agent.epsilons, fig)
                    if (episode_count) % 1 == 0:
                        print("| Reward: " + str(episode_reward), " | Episode", episode_count, " | Epsilon", self.agent.epsilon)
                    sess.run(increment)  # Next global episode

                self.agent.update_dynamic_epsilon(self.agent.epsilon_min, params['dynamic_epsilon_max'], params['dynamic_epsilon_decay'])

                episode_count += 1
                trial_count +=1


    # code to run a computer trial using q-tables (UNFINISHED)
    def run_trial_qtable(self, scenario_name, action_limit, attempt_limit, trial_count, iter_num, testing=False, specified_trial=None):
        """
        Run a computer trial using q-tables.

        :param scenario_name:
        :param action_limit:
        :param attempt_limit:
        :param trial_count:
        :param iter_num:
        :param testing:
        :param specified_trial:
        :return:
        """
        self.env.human_agent = False
        trial_selected = self.run_trial_common_setup(scenario_name, action_limit, attempt_limit, specified_trial)

        state = self.env.reset()
        state = np.reshape(state, [1, self.agent.state_size])

        save_dir = self.writer.subject_path + '/models'

        print('scenario_name: {}, trial_count: {}, trial_name: {}'.format(scenario_name, trial_count, trial_selected))

        trial_reward = 0
        attempt_reward = 0
        while not self.determine_computer_trial_finished(attempt_limit):
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

    def determine_computer_trial_finished(self, attempt_limit):
        # end if attempt limit reached
        if self.env.attempt_count >= attempt_limit:
            return True
        # trial is success and not forcing agent to use all attempts
        elif self.params['full_attempt_limit'] is False and self.agent.logger.cur_trial.success is True:
            return True
        return False

    def determine_human_trial_finished(self, attempt_limit):
        if self.env.attempt_count >= attempt_limit or self.env.determine_trial_success():
            return True
        return False

    def update_scenario(self, scenario):
        """
        Set the environment's scenario to the specified scenario.

        :param scenario: new scenario to use
        :return: Nothing
        """
        self.env.scenario = scenario
        self.env.solutions = scenario.solutions
        self.env.completed_solutions = []
        self.env.cur_action_seq = []

    def update_attempt(self, multithread=False):
        """
        Check if the action count has reached the action limit.
        If so, increment the environment attempt_count and call the env.reset() method.

        :param multithread: default: False
        :return: boolean representing whether or not the environment was reset
        """
        reset = False
        # above the allowed number of actions, need to increment the attempt count and reset the simulator
        if self.env.action_count >= self.env.action_limit:

            self.env.attempt_count += 1

            attempt_success = self.env.determine_unique_solution()
            if attempt_success:
                self.env.completed_solutions.append(self.env.cur_action_seq)

            self.agent.logger.cur_trial.finish_attempt(attempt_success=attempt_success, results=self.env.results)

            # update the user about their progress
            all_solutions_found, pause = self.update_user(attempt_success, multithreaded=multithread)

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
            if not all_solutions_found or self.env.full_attempt_limit is not False:
                self.add_attempt()

            self.env.reset()
            reset = True

        return reset

    def update_user(self, attempt_success, multithreaded=False):
        """
        Print update to the user.
        Either all solutions have been found, there are solutions remaining, or the user has
        reached the attempt limit and the trial is over without finding all solutions.

        :param attempt_success:
        :param multithreaded:
        :return: two booleans, the first representing whether the all solutions have been found (trial is finished), the second representing whether the simulator should pause (for when the user opened the door).
        """
        pause = False
        num_solutions_remaining = len(self.env.solutions) - len(self.env.completed_solutions)
        # continue or end trial
        if self.env.determine_trial_success():
            if not multithreaded:
                print("INFO: You found all of the solutions. ")
            # todo: should we mark that the trial is finished even though the attempt_limit
            # todo: may not be reached?
            all_solutions_found = True
            pause = True            # pause if they open the door
        elif self.env.attempt_count < self.env.attempt_limit:
            # alert user to the number of solutions remaining
            if attempt_success is True:
                if not multithreaded:
                    print("INFO: You found a solution. There are {} unique solutions remaining.".format(num_solutions_remaining))
                pause = True            # pause if they open the door
            else:
                if not multithreaded and self.env.human_agent:
                    print("INFO: Ending attempt. Action limit reached. There are {} unique solutions remaining. You have {} attempts remaining.".format(num_solutions_remaining, self.env.attempt_limit - self.env.attempt_count))
                # pause if the door lock is missing and the agent is a human
                if self.env.human_agent and self.env.get_state()['OBJ_STATES']['door_lock'] == common.ENTITY_STATES['DOOR_UNLOCKED']:
                    pause = True
            all_solutions_found = False
        else:
            if not multithreaded:
                print("INFO: Ending trial. Attempt limit reached. You found {} unique solutions".format(len(self.env.completed_solutions)))
            all_solutions_found = True

        return all_solutions_found, pause

    def finish_action(self, multithread=False):
        """
        Log an action and determine whether or not to reset by calling update_attempt.

        :param multithread:
        :return: env_reset: False or update_attempt()
        """
        env_reset = False
        # acknowledge action
        if not self.env.action_ack:
            # if the current attempt is empty, add a new one and reset
            if self.agent.logger.cur_trial.cur_attempt is None:
                #print 'cur_attempt is none...shouldnt be'
                self.env.cur_action_seq = []
                self.agent.logger.cur_trial.add_attempt()
                self.env.reset()
                return True
            self.agent.logger.cur_trial.cur_attempt.add_action(self.env.action.name, self.env.action.start_time)
            self.env.action_ack = True
        if not self.env.action_finish_ack:
            self.agent.logger.cur_trial.cur_attempt.finish_action(self.env.results, self.env.action.end_time)
            self.env.action_finish_ack = True
            env_reset = self.update_attempt(multithread=multithread)

        return env_reset

    def set_action_limit(self, action_limit):
        """
        Set self.env.action_limit.

        :param action_limit: new self.env.action_limit
        :return: Nothing
        """
        self.env.action_limit = action_limit

    def add_attempt(self):
        """
        Log the attempt and reset env.cur_action_seq.

        :return: Nothing
        """
        self.env.cur_action_seq = []
        self.agent.logger.cur_trial.add_attempt()

    def print_update(self, iter_num, trial_num, scenario_name, episode, episode_max, a_reward, t_reward, epsilon):
        """
        Print ID, iteration number, trial number, scenario, episode, attempt_reward, trial_reward, epsilon.

        :param iter_num:
        :param trial_num:
        :param scenario_name:
        :param episode:
        :param episode_max:
        :param a_reward:
        :param t_reward:
        :param epsilon:
        :return: Nothing
        """
        print("ID: {}, iter {}, trial {}, scenario {}, episode: {}/{}, attempt_reward {}, trial_reward {}, e: {:.2}".format(self.agent.subject_id, iter_num, trial_num, scenario_name, episode, episode_max, a_reward, t_reward, epsilon))

    def verify_fsm_matches_simulator(self, obs_space):
        """
        Ensure that the simulator data matches the FSM.

        :param obs_space:
        :return: obs_space
        """
        if obs_space is None:
            obs_space = ObservationSpace(len(self.env.world_def.get_levers()))
        state, labels = obs_space.create_discrete_observation_from_simulator(self.env)
        fsm_state, fsm_labels = obs_space.create_discrete_observation_from_fsm(self.env)
        try:
            assert(state == fsm_state)
            assert(labels == fsm_labels)
        except AssertionError:
            print('FSM does not match simulator data')
            print(state)
            print(fsm_state)
            print(labels)
            print(fsm_labels)
        return obs_space

    def dqn_trial_sanity_checks(self):
        """
        Used by dqn & ddqn trials to make sure attempt_seq and reward_seq are valid.

        :return: Nothing
        """
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

