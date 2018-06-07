
from shutil import copytree, ignore_patterns
import time

from logger import SubjectLogger, SubjectWriter


# base class for all agents; each agent has a logger
class Agent(object):
    """
    Manage the agent's internals (e.g. neural network for DQN/DDQN, or q-table for Q-table)
    and maintain a logger to record the outcomes the agent achieves.
    """
    def __init__(self, data_path):
        """
        Initialize logger, writer, subject_id, human to None; data_path to data_path.

        :param data_path: path to directory to write log files to
        """
        self.logger = None
        self.writer = None
        self.subject_id = None
        self.data_path = data_path
        self.human = False

    # default args are for non-human agent
    def setup_subject(self, human=False, participant_id=-1, age=-1, gender='robot', handedness='none', eyewear='no', major='robotics', random_seed = None):
        """
        Set internal variables for subject, initialize logger, and create a copy of the code base for reproduction.

        :param human: True if human agent, default: False
        :param participant_id: default: -1
        :param age: default: -1
        :param gender: default: 'robot'
        :param handedness: default: 'none'
        :param eyewear: default: 'no'
        :param major: default: 'robotics'
        :param random_seed: default: None
        :return: Nothing
        """
        self.human = human
        self.writer = SubjectWriter(self.data_path)
        self.subject_id = self.writer.subject_id

        print("Starting trials for subject {}. Saving to {}".format(self.subject_id, self.writer.subject_path))
        self.logger = SubjectLogger(subject_id=self.subject_id,
                                    participant_id=participant_id,
                                    age=age,
                                    gender=gender,
                                    handedness=handedness,
                                    eyewear=eyewear,
                                    major=major,
                                    start_time=time.time(),
                                    random_seed= random_seed)

        # copy the entire code base; this is unnecessary but prevents worrying about a particular
        # source code version when trying to reproduce exact parameters
        copytree('./', self.writer.subject_path + '/src/', ignore=ignore_patterns('*.mp4',
                                                                                  '*.pyc',
                                                                                  '.git',
                                                                                  '.gitignore',
                                                                                  '.gitmodules'))

    def get_current_attempt_logged_actions(self, idx):
        results = self.logger.cur_trial.cur_attempt.results
        agent_idx = results[0].index('agent')
        actions = results[idx][agent_idx+1:len(results[idx])]
        action_labels = results[0][agent_idx+1:len(results[idx])]
        return actions, action_labels

    def get_current_attempt_logged_states(self, idx):
        results = self.logger.cur_trial.cur_attempt.results
        agent_idx = results[0].index('agent')
        # frame is stored in 0
        states = results[idx][1:agent_idx]
        state_labels = results[0][1:agent_idx]
        return states, state_labels

    def get_last_attempt_results(self):
        if self.logger.cur_trial is not None:
            trial = self.logger.cur_trial
        else:
            trial = self.logger.trial_seq[-1]
        if trial.cur_attempt is not None:
            return trial.cur_attempt.results
        else:
            return trial.attempt_seq[-1].results

    def write_results(self):
        """
        Log current agent state.

        :return: Nothing
        """
        self.writer.write(self.logger, self)

    def write_trial(self, test_trial=False, random_seed = None):
        """
        Log trial.

        :param test_trial: true if test trial, default: False
        :param random_seed: default: None
        :return: Nothing
        """
        self.writer.write_trial(self.logger, test_trial)

    def finish_trial(self, test_trial, random_seed):
        """
        Finish trial and log it.

        :param test_trial: true if test trial
        :param random_seed:
        :return:
        """
        self.logger.finish_trial()
        self.write_trial(test_trial, random_seed)

    def finish_subject(self, strategy, transfer_strategy):
        """
        Finish subject at current time, set strategy and transfer_strategy, call write_results().

        :param strategy:
        :param transfer_strategy:
        :return: Nothing
        """
        self.logger.finish(time.time())
        self.logger.strategy = strategy
        self.logger.transfer_strategy = transfer_strategy

        self.write_results()