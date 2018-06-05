
import time
import jsonpickle
import os
import json
import copy
import texttable


class ActionLog(object):
    start_time = None
    end_time = None
    name = None

    def __init__(self, name, start_time):
        self.name = name
        self.start_time = start_time

    def __eq__(self, other):
        return self.name == other.name

    def __str__(self):
        return self.name

    def __repr__(self):
        return str(self)

    def finish(self, end_time):
        self.end_time = end_time


class AttemptLog(object):
    attempt_num = None
    action_seq = []
    start_time = None
    success = False
    end_time = None
    cur_action = None
    results = None
    reward = None

    def __init__(self, attempt_num, start_time):
        self.attempt_num = attempt_num
        self.start_time = start_time
        self.reward = None

    def __eq__(self, other):
        return self.action_seq == other.action_seq

    def __str__(self):
        return self.pretty_print_results()

    def add_action(self, name, t=None):
        if t is None:
            t = time.time()
        self.cur_action = ActionLog(name, t)

    def finish_action(self, results, t=None):
        if t is None:
            t = time.time()
        self.cur_action.finish(t)
        self.results = results
        self.action_seq.append(copy.deepcopy(self.cur_action))
        self.cur_action = None

    def finish(self, success, results, end_time):

        self.success = success
        self.results = results
        self.end_time = end_time
        return self.success

    def pretty_print_results(self):
        table = texttable.Texttable()
        col_labels = self.results[0]
        table.set_cols_align(['l' for i in range(len(col_labels))])
        table.add_rows(self.results[1:len(self.results)])
        table.header(col_labels)
        table.set_cols_width([12 for i in range(len(col_labels))])
        return table.draw()


class TrialLog(object):
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

    def __init__(self, name, scenario_name, solutions, start_time):
        self.name = name
        self.start_time = start_time
        self.scenario_name = scenario_name
        self.solutions = solutions
        self.success = False
        self.completed_solutions = []
        self.attempt_seq = []
        self.solution_found = []
        self.trial_reward = None

    def add_attempt(self):
        self.cur_attempt = AttemptLog(len(self.attempt_seq), time.time())
        self.cur_attempt.action_seq = []

    def finish_attempt(self, attempt_success, results):
        # check to see if this attempt is a solution that has not been completed already
        if attempt_success:
            self.completed_solutions.append(self.cur_attempt.action_seq)
        self.solution_found.append(attempt_success)
        self.cur_attempt.finish(attempt_success, results, time.time())
        self.attempt_seq.append(copy.deepcopy(self.cur_attempt))
        self.success = len(self.solutions) == len(self.completed_solutions)
        self.cur_attempt = None

    def finish(self, end_time):
        # if we have completed all solutions (solutions are only added if they are not repeats)
        self.success = len(self.solutions) == len(self.completed_solutions)
        self.end_time = end_time
        return self.success


class SubjectLogger(object):
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

    def __init__(self, subject_id, participant_id, age, gender, handedness, eyewear, major, start_time, human=True):
        self.subject_id = subject_id
        self.participant_id = participant_id
        self.start_time = start_time
        self.age = age
        self.gender = gender
        self.handedness = handedness
        self.eyewear = eyewear
        self.major = major
        self.human = human

    def add_trial(self, trial_name, scenario_name, solutions):
        self.cur_trial = TrialLog(trial_name, scenario_name, solutions, time.time())

    def finish_trial(self):
        success = self.cur_trial.finish(time.time())
        self.trial_seq.append(self.cur_trial)
        self.cur_trial = None
        return success

    def finish(self, end_time):
        self.end_time = end_time

# SubjectLogger used to be called SubjectLog, so we'll allow the pickler to
# properly instantiate the class
SubjectLog = SubjectLogger


class SubjectWriter:
    subject_path = None

    def __init__(self, data_path):
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

    def write_trial(self, logger, test_trial=False):
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
        subject_summary = jsonpickle.encode(logger)
        # json_results = self.JSONify_subject(logger)

        subject_summary_filename = self.subject_path + '/' + logger.subject_id +'_summary.json'
        self.pretty_write(subject_summary_filename, subject_summary)

        # write out the RL agent
        if agent is not None:
            agent_cpy = copy.copy(agent)
            if hasattr(agent_cpy, 'memory'):
                del agent_cpy.memory
            if hasattr(agent_cpy, 'model'):
                del agent_cpy.model
            if hasattr(agent_cpy, 'target_model'):
                del agent_cpy.target_model
            agent_file_name = self.subject_path + '/' + logger.subject_id + '_agent.json'
            agent_str = jsonpickle.encode(agent_cpy, unpicklable=False)
            self.pretty_write(agent_file_name, agent_str)

    @staticmethod
    def pretty_write(filename, json_str):
        with open(filename, 'w') as outfile:
            outfile.write(json_str)

        with open(filename, 'r') as infile:
            ugly_json_str = json.load(infile)

        sort = True
        indents = 4

        with open(filename, 'w') as outfile:
            json.dump(ugly_json_str, outfile, sort_keys=sort, indent=indents)

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
    return obj.__dict__
