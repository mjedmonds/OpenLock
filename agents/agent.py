
from shutil import copytree, ignore_patterns
import time

from logger import SubjectLogger, SubjectWriter


# base class for all agents; each agent has a logger
class Agent(object):
    def __init__(self, data_path):
        self.logger = None
        self.writer = None
        self.subject_id = None
        self.data_path = data_path
        self.human = False

    # default args are for non-human agent
    def setup_subject(self, human=False, participant_id=-1, age=-1, gender='robot', handedness='none', eyewear='no', major='robotics', random_seed = None):
        self.human = human
        self.writer = SubjectWriter(self.data_path)
        self.subject_id = self.writer.subject_id

        print("Starting trials for subject {}".format(self.subject_id))
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

    def write_results(self):
        self.writer.write(self.logger, self)

    def write_trial(self, test_trial=False, random_seed = None):
        self.writer.write_trial(self.logger, test_trial)

    def finish_trial(self, test_trial, random_seed):
        self.logger.finish_trial()
        self.write_trial(test_trial, random_seed)

    def finish_subject(self, strategy, transfer_strategy):
        self.logger.finish(time.time())
        self.logger.strategy = strategy
        self.logger.transfer_strategy = transfer_strategy

        self.write_results()