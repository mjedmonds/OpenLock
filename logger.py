
import time
import jsonpickle
import os
import json
import copy
import texttable
import sys


class ActionLog(object):
    """
    Represents an action for the purpose of logging. Actions have a name, start time, and end time.
    """
    start_time = None
    end_time = None
    name = None

    def __init__(self, name, start_time):
        """
        Create an action.

        :param name: Name of the action.
        :param start_time: Start time of the action.
        """
        self.name = name
        self.start_time = start_time

    def __eq__(self, other):
        """
        Check equality of this action's name with another by name.

        :param other: ActionLog object to compare this one with.
        :return: True if names are the same, False otherwise.
        """
        return self.name == other.name

    def __str__(self):
        """
        Get name of action.

        :return: self.name.
        """
        return self.name

    def __repr__(self):
        """
        Get string representation of action.

        :return: str(self), which returns self.name.
        """
        return str(self)

    def finish(self, end_time):
        self.end_time = end_time


class AttemptLog(object):
    """
    Represents an attempt for the purpose of logging.
    """
    attempt_num = None
    action_seq = []
    start_time = None
    success = False
    end_time = None
    cur_action = None
    results = None
    reward = None

    def __init__(self, attempt_num, start_time):
        """
        Create an attempt.

        :param attempt_num: Number of this attempt.
        :param start_time: Start time of this attempt.
        """
        self.attempt_num = attempt_num
        self.start_time = start_time
        self.reward = None

    def __eq__(self, other):
        """
        Check equality of this attempt with another by comparing action sequences.

        :param other: Other AttemptLog to compare this one with.
        :return: True if action_seq are equal, False otherwise.
        """
        return self.action_seq == other.action_seq

    def __str__(self):
        """
        Get string representation of attempt. Calls pretty_str_results().

        :return: Same as pretty_str_results().
        """
        return self.pretty_str_results()

    def add_action(self, name, t=None):
        """
        Add an action to this attempt. Set it to current action.

        :param name: Name of the action.
        :param t: Start time of the action.
        :return: Nothing.
        """
        if t is None:
            t = time.time()
        self.cur_action = ActionLog(name, t)

    def finish_action(self, results, t=None):
        """
        Finish the current action.

        :param results: Environment results.
        :param t: Finish time of the action.
        :return: Nothing
        """
        if t is None:
            t = time.time()
        self.cur_action.finish(t)
        self.results = results
        self.action_seq.append(copy.deepcopy(self.cur_action))
        self.cur_action = None

    def finish(self, success, results, end_time):
        """
        Finish the attempt.

        :param success: Success of the attempt.
        :param results: Environment results.
        :param end_time: Time of end of the attempt.
        :return: success (same as param).
        """

        self.success = success
        self.results = results
        self.end_time = end_time
        return self.success

    def pretty_str_results(self):
        """
        Print results in an ASCII table.

        :return: Results table as a string.
        """
        table = texttable.Texttable()
        col_labels = self.results[0]
        table.set_cols_align(['l' for i in range(len(col_labels))])
        content = [col_labels]
        content.extend(self.results[1:len(self.results)])
        table.add_rows(content)
        table.set_cols_width([12 for i in range(len(col_labels))])
        return table.draw()


class TrialLog(object):
    """
    Represents a trial for the purpose of logging.
    """
    attempt_seq = []
    success = False
    start_time = None
    end_time = None
    name = None
    scenario_name = None
    cur_attempt = None
    solutions = []
    completed_solutions = []
    solution_found = []
    random_seed = None

    def __init__(self, name, scenario_name, solutions, start_time, random_seed):
        """
        Create the trial.

        :param name: Name of the trial.
        :param scenario_name: Name of the scenario.
        :param solutions:
        :param start_time: Start time of the trial.
        :param random_seed:
        """
        self.name = name
        self.start_time = start_time
        self.scenario_name = scenario_name
        self.solutions = solutions
        self.success = False
        self.completed_solutions = []
        self.attempt_seq = []
        self.solution_found = []
        self.trial_reward = None
        self.random_seed = random_seed

    def add_attempt(self):
        """
        Add attempt to the trial with current time as start time. Set it to current attempt.

        :return: Nothing
        """
        self.cur_attempt = AttemptLog(len(self.attempt_seq), time.time())
        self.cur_attempt.action_seq = []

    def finish_attempt(self, attempt_success, results):
        """
        Mark the current attempt as finished.

        :param attempt_success: Whether or not the attempt is a success.
        :param results: Environment results.
        :return: Nothing.
        """
        # check to see if this attempt is a solution that has not been completed already
        if attempt_success:
            self.completed_solutions.append(self.cur_attempt.action_seq)
        self.solution_found.append(attempt_success)
        self.cur_attempt.finish(attempt_success, results, time.time())
        self.attempt_seq.append(copy.deepcopy(self.cur_attempt))
        self.success = len(self.solutions) == len(self.completed_solutions)
        self.cur_attempt = None

    def finish(self, end_time):
        """
        Finish the trial.

        :param end_time: Time that the trial ended.
        :return: True if all solutions were completed, False otherwise.
        """
        # if we have completed all solutions (solutions are only added if they are not repeats)
        self.success = len(self.solutions) == len(self.completed_solutions)
        self.end_time = end_time
        return self.success


class SubjectLogger(object):
    """
    Represents a subject for the purpose of logger.
    """
    subject_id = None
    age = None
    gender = None
    handedness = None
    eyewear = None
    major = None

    trial_seq = []
    cur_trial = None
    start_time = None
    end_time = None

    cur_scenario_name = None

    strategy = None
    random_seed = None


    def __init__(self, subject_id, participant_id, age, gender, handedness, eyewear, major, start_time, human=True, random_seed = None):
        """
        Create the subject.

        :param subject_id: Subject ID.
        :param participant_id: Participant ID.
        :param age: Age of subject.
        :param gender: Gender of subject.
        :param handedness: Handedness of subject.
        :param eyewear: Eyewear of subject (yes or no).
        :param major: Major of subject.
        :param start_time: Time that the subject starts.
        :param human: Whether subject is human, default True.
        :param random_seed: Default None.
        """
        self.subject_id = subject_id
        self.participant_id = participant_id
        self.start_time = start_time
        self.age = age
        self.gender = gender
        self.handedness = handedness
        self.eyewear = eyewear
        self.major = major
        self.human = human
        self.random_seed = random_seed

    def add_trial(self, trial_name, scenario_name, solutions, random_seed):
        """
        Set the current trial to a new TrialLog object.

        :param trial_name: Name of the trial.
        :param scenario_name: Name of the scenario.
        :param solutions: Solutions of trial.
        :param random_seed:
        :return: Nothing.
        """
        self.cur_trial = TrialLog(trial_name, scenario_name, solutions, time.time(), random_seed)

    def finish_trial(self):
        """
        Finish the current trial.

        :return: True if trial was successful, False otherwise.
        """
        success = self.cur_trial.finish(time.time())
        self.trial_seq.append(self.cur_trial)
        self.cur_trial = None
        return success

    def finish(self, end_time):
        """
        Set end time of the subject.

        :param end_time: End time of the subject.
        :return: Nothing.
        """
        self.end_time = end_time

# SubjectLogger used to be called SubjectLog, so we'll allow the pickler to
# properly instantiate the class
SubjectLog = SubjectLogger


class TerminalLogger:
    """
    Logs stdout output to agent's log
    """
    def __init__(self, logfile):
        self.stdout = sys.stdout
        self.log = open(logfile, 'a')

    def write(self, message):
        self.stdout.write(message)
        self.log.write(message)

    def flush(self):
        pass


class SubjectWriter:
    """
    Writes the log files for a subject.
    """
    subject_path = None

    def __init__(self, data_path):
        """
        Create a log file for the subject inside the data_path directory.

        :param data_path: Path to keep the log files in.
        """
        self.subject_id = str(hash(time.time()))
        self.subject_path = data_path + '/' + self.subject_id
        while True:
            # make sure directory does not exist
            if not os.path.exists(self.subject_path):
                os.makedirs(self.subject_path)
                break
            else:
                self.subject_id = str(hash(time.time()))
                self.subject_path = data_path + '/' + self.subject_id
                continue
        # setup writing stdout to file and to stdout
        self.terminal_logger = TerminalLogger(self.subject_path + '/' + self.subject_id + '_stdout.log')

    def write_trial(self, logger, test_trial=False):
        """
        Make trial directory, trial summary, and write trial summary.

        :param logger: SubjectLogger.
        :param test_trial: True if test trial, False otherwise. Default False.
        :return: Nothing.
        """
        i = len(logger.trial_seq)-1
        trial = logger.trial_seq[i]
        trial_str = jsonpickle.encode(trial)
        trial_dir = self.subject_path + '/trial' + str(i)
        # if test_trial:
        #     trial_dir = trial_dir + '_test'
        os.makedirs(trial_dir)
        trial_summary_filename = trial_dir + '/trial' + str(i) + '_summary.json'
        self.pretty_write(trial_summary_filename, trial_str)

    def write(self, logger, agent):
        """
        Write subject summary log.

        :param logger: SubjectLogger object.
        :param agent: Agent object.
        :return: Nothing.
        """
        subject_summary = jsonpickle.encode(logger)
        # json_results = self.JSONify_subject(logger)

        subject_summary_filename = self.subject_path + '/' + logger.subject_id +'_summary.json'
        self.pretty_write(subject_summary_filename, subject_summary)

        # write out the RL agent
        if agent is not None:
            agent_cpy = copy.copy(agent)
            agent_file_name = self.subject_path + '/' + logger.subject_id + '_agent.json'
            agent_str = jsonpickle.encode(agent_cpy, unpicklable=False)
            self.pretty_write(agent_file_name, agent_str)

    @staticmethod
    def pretty_write(filename, json_str):
        """
        Write json_str to filename with sort_keys=True, indents=4.

        :param filename: Name of file to be output.
        :param json_str: JSON str to write (e.g. from jsonpickle.encode()).
        :return: Nothing.
        """
        with open(filename, 'w') as outfile:
            # reencode to pretty print
            json_obj = json.loads(json_str)
            json_str = json.dumps(json_obj, indent=4, sort_keys=True)
            outfile.write(json_str)

            # results_dir = trial_dir + '/results'
            # os.makedirs(results_dir)
            # for j in range(len(trial.attempt_seq)):
            #     attempt = trial.attempt_seq[j]
            #     results = attempt.results
            #     np.savetxt(results_dir + '/results_attempt' + str(j) + '.csv', results, delimiter=',', fmt='%s')
    #
    # def JSONify_subject(self, subject):
    #     trial_jsons = []
    #     for trial in subject.trial_seq:
    #         trial_jsons.append( self.JSONify_trial(subject.trial))
    #     subject_json = jsonpickle.encode(subject)
    #     print trial_jsons
    #
    # def JSONify_trial(self, trial_seq):
    #     attempt_jsons = []
    #     for attempt in trial.attempt_seq:
    #         attempt_jsons.append(self.JSONify_attempt(attempt))
    #     trial_json = jsonpickle.encode(trial)
    #     return trial_json
    #
    # def JSONify_attempt(self, attempt):
    #     results_seq_str = jsonpickle.encode(attempt.results_seq)
    #     attempt.results_seq_str = results_seq_str
    #     return jsonpickle.encode(attempt)
    #
    # def JSONify_action(self, action):
    #     return jsonpickle.encode(action)

def obj_dict(obj):
    """
    Get object dict.

    :param obj: An object.
    :return: __dict__ of the object passed in.
    """
    return obj.__dict__
