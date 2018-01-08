
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

    def __init__(self, env, params, computer=False):
        self.env = env
        self.params = params

        # logger is stored in the environment - change if possible
        self.env.logger, self.writer = self.setup_subject(params['data_dir'], computer)

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
        trial_selected = self.run_trial_common_setup(scenario_name, action_limit, attempt_limit, specified_trial)

        while self.env.attempt_count < attempt_limit and self.env.logger.cur_trial.success is False:
            self.env.render()

        self.run_trial_common_finish(trial_selected)

    # code to run a computer trial
    def run_trial_computer(self, scenario_name, action_limit, attempt_limit):

        self.run_trial_common_finish(trial_selected)

    def update_scenario(self, scenario):
        self.env.scenario = scenario

    def set_action_limit(self, action_limit):
        self.env.action_limit = action_limit

    def write_results(self):
        self.writer.write(self.env.logger)

    def finish_subject(self):
        self.env.logger.finish(time.time())

    @staticmethod
    def get_trial(name, completed_trials=None):
        if name != 'CE4' and name != 'CC4':
            # select a random trial and add it to the scenario
            trial, configs = select_random_trial(completed_trials, 1, 6)
        else:
            trial, configs = select_trial('trial7')

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
    def prompt_subject():
        print 'Welcome to OpenLock!'
        age = SessionManager.prompt_age()
        gender = SessionManager.prompt_gender()
        handedness = SessionManager.prompt_handedness()
        eyewear = SessionManager.prompt_eyewear()
        return age, gender, handedness, eyewear

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
    def setup_subject(data_path, computer=False):
        # human agent
        if not computer:
            age, gender, handedness, eyewear = SessionManager.prompt_subject()
            # age, gender, handedness, eyewear = ['25', 'M', 'right', 'no']
            subject_id, subject_path = SessionManager.make_subject_dir(data_path)
        # robot agent
        else:
            age = -1
            gender = 'robot'
            handedness = 'none'
            eyewear = 'no'
            subject_id, subject_path = SessionManager.make_subject_dir(data_path)

        print "Starting trials for subject {}".format(subject_id)
        sub_logger = logger.SubjectLog(subject_id=subject_id,
                                       age=age,
                                       gender=gender,
                                       handedness=handedness,
                                       eyewear=eyewear,
                                       start_time=time.time())
        sub_writer = logger.SubjectWriter(subject_path)
        return sub_logger, sub_writer
