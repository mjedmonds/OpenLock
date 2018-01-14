
import time
import os
import copy

from gym_lock.settings_trial import select_random_trial, select_trial
import logger


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

    # code to run before human and computer trials
    def run_trial_common_setup(self, scenario_name, action_limit, attempt_limit, specified_trial=None):
        # setup trial
        self.env.attempt_count = 0
        self.env.attempt_limit = attempt_limit
        self.set_action_limit(action_limit)
        # select trial
        if specified_trial is None:
            trial_selected, lever_configs = self.get_trial(scenario_name, self.completed_trials)
        else:
            trial_selected, lever_configs = select_trial(specified_trial)
        self.env.scenario.set_lever_configs(lever_configs)

        self.env.logger.add_trial(trial_selected, scenario_name, self.env.scenario.solutions)
        self.env.logger.cur_trial.add_attempt()

        print "INFO: New trial. There are {} unique solutions remaining.".format(len(self.env.scenario.solutions))

        self.env.reset()

        return trial_selected

    # code to run after both human and computer trials
    def run_trial_common_finish(self, trial_selected):
        # todo: detect whether or not all possible successful paths were uncovered
        self.env.logger.finish_trial()
        self.completed_trials.append(copy.deepcopy(trial_selected))

    # code to run a human subject
    def run_trial_human(self, scenario_name, action_limit, attempt_limit, specified_trial=None):
        self.env.human_agent = True
        trial_selected = self.run_trial_common_setup(scenario_name, action_limit, attempt_limit, specified_trial)

        while self.env.attempt_count < attempt_limit and self.env.logger.cur_trial.success is False:
            self.env.render()

        self.run_trial_common_finish(trial_selected)

    # code to run a computer trial
    def run_trial_computer(self, agent, obs_space, scenario_name, action_limit, attempt_limit, trial_count, specified_trial=None):
        self.env.human_agent = False
        trial_selected = self.run_trial_common_setup(scenario_name, action_limit, attempt_limit, specified_trial)

        prev_state = None
        cum_reward = 0
        agent_save_dir = '../OpenLockRLAgents'
        while self.env.attempt_count < attempt_limit and self.env.logger.cur_trial.success is False:
            # self.env.render()
            state = obs_space.create_discrete_observation_from_state(self.env.world_def)[0]

            # only allow agent to act after discrete state change
            if state is not prev_state:
                prev_state = state
                action_idx = agent.act(state)
                # convert idx to Action object (idx -> str -> Action)
                action = self.env.action_map[self.env.action_space[action_idx]]
                state, reward, done, _ = self.env.step(action)
                state = obs_space.create_discrete_observation_from_state(self.env.world_def)[0]
                agent.remember(prev_state, action, reward, state, done)
                cum_reward += reward
                if done:
                    print("trial {}, scenario {}, episode: {}/{}, reward {}, e: {:.2}".format(trial_count, scenario_name, self.env.attempt_count, self.env.attempt_limit, cum_reward, agent.epsilon))
                    print(self.env.logger.cur_trial.attempt_seq[-1].action_seq)
                    cum_reward = 0
                    # break
                # save agent every 10000 attempts
                if self.env.attempt_count % 10000 == 0:
                    save_dir = self.writer.subject_path + '/models'
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    agent.save(save_dir + '/agent_t' + str(trial_count) + '_a' + str(self.env.attempt_count) + '.h5')

        self.run_trial_common_finish(trial_selected)

        return agent

    def update_scenario(self, scenario):
        self.env.scenario = scenario

    def set_action_limit(self, action_limit):
        self.env.action_limit = action_limit

    @staticmethod
    def write_results(logger, writer):
        writer.write(logger)

    @staticmethod
    def finish_subject(logger, writer):
        logger.finish(time.time())
        strategy = SessionManager.prompt_strategy()
        transfer_strategy = SessionManager.prompt_transfer_strategy()
        logger.strategy = strategy
        logger.transfer_strategy = transfer_strategy

        SessionManager.write_results(logger, writer)

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
        age = SessionManager.prompt_age()
        gender = SessionManager.prompt_gender()
        handedness = SessionManager.prompt_handedness()
        eyewear = SessionManager.prompt_eyewear()
        major = SessionManager.prompt_major()
        return age, gender, handedness, eyewear, major

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
            age, gender, handedness, eyewear, major = SessionManager.prompt_subject()
            # age, gender, handedness, eyewear = ['25', 'M', 'right', 'no']
        # robot agent
        else:
            age = -1
            gender = 'robot'
            handedness = 'none'
            eyewear = 'no'
            major = 'robotics'

        subject_id, subject_path = SessionManager.make_subject_dir(data_path)

        print "Starting trials for subject {}".format(subject_id)
        sub_logger = logger.SubjectLog(subject_id=subject_id,
                                       age=age,
                                       gender=gender,
                                       handedness=handedness,
                                       eyewear=eyewear,
                                       major=major,
                                       start_time=time.time())
        sub_writer = logger.SubjectWriter(subject_path)
        return sub_logger, sub_writer


