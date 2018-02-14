
import time
import os
import copy
import numpy as np

from gym_lock.settings_trial import select_random_trial, select_trial
from gym_lock.envs.arm_lock_env import ObservationSpace
from gym_lock import logger


class SessionManager():

    env = None
    writer = None
    params = None
    completed_trials = []

    def __init__(self, env, params, human=True):
        self.env = env
        self.params = params

        # logger is stored in the environment - change if possible
        self.env.logger, self.writer = self.setup_subject(params['data_dir'], human)

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
            self.env.render()
            # used to verify simulator and fsm states are always the same (they should be)
            if verify:
                obs_space = self.verify_fsm_matches_simulator(obs_space)

        self.run_trial_common_finish(trial_selected)

    # code to run a computer trial
    def run_trial_dqn(self, agent, scenario_name, action_limit, attempt_limit, trial_count, iter_num, testing=False, specified_trial=None):
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

            if self.env.logger.cur_trial.success:
                print('found both solutions')

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

            self.save_agent(agent, save_dir, testing, iter_num, trial_count)

            # replay to learn
            if len(agent.memory) > agent.batch_size:
                agent.replay()

        self.dqn_trial_sanity_checks(agent)

        self.env.logger.cur_trial.trial_reward = trial_reward
        self.run_trial_common_finish(trial_selected)
        agent.trial_switch_points.append(len(agent.rewards))
        agent.average_trial_rewards.append(trial_reward / self.env.attempt_count)

        return agent

    def save_agent(self, agent, save_dir, testing, iter_num, trial_count):
        # save model
        # if self.env.attempt_count % (self.env.attempt_limit/2) == 0 or self.env.attempt_count == self.env.attempt_limit or self.env.logger.cur_trial.success is True:
        if self.env.attempt_count == 0 or self.env.attempt_count == self.env.attempt_limit:
            if testing:
                save_str = '/agent_test_i_' + str(iter_num) + '_t' + str(trial_count) + '_a' + str(self.env.attempt_count) + '.h5'
            else:
                save_str = '/agent_i_' + str(iter_num) + '_t' + str(trial_count) + '_a' + str(self.env.attempt_count) + '.h5'
            agent.save_model(save_dir,  save_str)

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
    def write_results(logger, writer, agent=None):
        writer.write(logger, agent)

    @staticmethod
    def finish_subject(logger, writer, human=True, agent=None):
        logger.finish(time.time())
        if human:
            strategy = SessionManager.prompt_strategy()
            transfer_strategy = SessionManager.prompt_transfer_strategy()
        else:
            strategy = 'RL'
            transfer_strategy = 'RL'
        logger.strategy = strategy
        logger.transfer_strategy = transfer_strategy

        SessionManager.write_results(logger, writer, agent)

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

    @staticmethod
    def prompt_participant_id():
        while True:
            try: 
                participant_id = int(raw_input('Please enter the participant ID (ask the RA for this): '))
            except ValueError:
                print 'Please enter an integer for the participant ID'
                continue
            else:
                return participant_id

    @staticmethod
    def prompt_age():
        while True:
            try:
                age = int(raw_input('Please enter your age: '))
            except ValueError:
                print 'Please enter your age as an integer'
                continue
            else:
                return age

    @staticmethod
    def prompt_gender():
        while True:
            gender = raw_input('Please enter your gender (\'M\' for male, \'F\' for female, or \'O\' for other): ')
            if gender == 'M' or gender == 'F' or gender == 'O':
                return gender
            else:
                continue

    @staticmethod
    def prompt_handedness():
        while True:
            handedness = raw_input('Please enter your handedness (\'right\' for right-handed or \'left\' for left-handed): ')
            if handedness == 'right' or handedness == 'left':
                return handedness
            else:
                continue

    @staticmethod
    def prompt_eyewear():
        while True:
            eyewear = raw_input('Please enter \'yes\' if you wear glasses or contacts or \'no\' if you do not wear glasses or contacts: ')
            if eyewear == 'yes' or eyewear == 'no':
                return eyewear
            else:
                continue

    @staticmethod
    def prompt_major():
        major = raw_input('Please enter your major: ')
        return major

    @staticmethod
    def prompt_subject():
        print 'Welcome to OpenLock!'
        participant_id = SessionManager.prompt_participant_id()
        age = SessionManager.prompt_age()
        gender = SessionManager.prompt_gender()
        handedness = SessionManager.prompt_handedness()
        eyewear = SessionManager.prompt_eyewear()
        major = SessionManager.prompt_major()
        return participant_id, age, gender, handedness, eyewear, major

    @staticmethod
    def prompt_strategy():
        strategy = raw_input('Did you develop any particular technique or strategy to solve the problem? If so, what was your technique/strategy? ')
        return strategy

    @staticmethod
    def prompt_transfer_strategy():
        transfer_strategy = raw_input('If you used a particular technique/strategy, did you find that it also worked when the number of colored levers increased from 3 to 4? ')
        return transfer_strategy

    @staticmethod
    def make_subject_dir(data_path):
        subject_id = str(hash(time.time()))
        subject_path = data_path + '/' + subject_id
        while True:
            # make sure directory does not exist
            if not os.path.exists(subject_path):
                os.makedirs(subject_path)
                return subject_id, subject_path
            else:
                subject_id = str(hash(time.time()))
                subject_path = data_path + '/' + subject_id
                continue

    @staticmethod
    def setup_subject(data_path, human=True):
        # human agent
        if human:
            participant_id, age, gender, handedness, eyewear, major = SessionManager.prompt_subject()
            # age, gender, handedness, eyewear = ['25', 'M', 'right', 'no']
        # robot agent
        else:
            age = -1
            gender = 'robot'
            handedness = 'none'
            eyewear = 'no'
            major = 'robotics'
            participant_id = -1

        subject_id, subject_path = SessionManager.make_subject_dir(data_path)

        print "Starting trials for subject {}".format(subject_id)
        sub_logger = logger.SubjectLog(subject_id=subject_id,
                                       participant_id=participant_id,
                                       age=age,
                                       gender=gender,
                                       handedness=handedness,
                                       eyewear=eyewear,
                                       major=major,
                                       start_time=time.time())
        sub_writer = logger.SubjectWriter(subject_path)
        return sub_logger, sub_writer


