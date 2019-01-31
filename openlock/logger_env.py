import time
import jsonpickle
import os
import json
import copy
import texttable
import sys

from openlock.common import Action


class ActionLog(object):
    """
    Represents an action for the purpose of logging. Actions have a name, start time, and end time.
    """

    start_time = None
    end_time = None
    name = None
    reward = 0

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
        if isinstance(other, ActionLog):
            return self.name == other.name
        elif isinstance(other, str):
            return self.name == other
        else:
            raise TypeError("Unexpected object for ActionLog() equality operator")

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

    def finish(self, end_time=None):
        if end_time is None:
            end_time = time.time()
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
        self.reward = 0

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
        return copy.copy(self.cur_action)

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
        action = copy.copy(self.cur_action)

        self.cur_action = None
        return action

    def add_reward(self, reward):
        """
        Add the reward to the last action. Action must have called ActionLog.finish() before computing reward
        Because of this, we add/log the reward separately.
        :param reward: reward received from the previously executed action
        :return: Nothing
        """
        self.reward += reward
        self.action_seq[-1].reward = reward

    def finish(self, success, results, end_time):
        """
        Finish the attempt.

        :param success: Success of the attempt.
        :param results: Environment results.
        :param end_time: Time of end of the attempt.
        :param reward: reward received for this attempt
        :return Nothing
        """

        self.success = success
        self.results = results
        self.end_time = end_time

    def pretty_str_results(self):
        """
        Print results in an ASCII table.

        :return: Results table as a string.
        """
        table = texttable.Texttable()
        col_labels = self.results[0]
        table.set_cols_align(["l" for i in range(len(col_labels))])
        content = [col_labels]
        content.extend(self.results[1 : len(self.results)])
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

    def __init__(self, name, scenario_name, solutions, start_time):
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

    def add_attempt(self):
        """
        Add attempt to the trial with current time as start time. Set it to current attempt.

        :return: Nothing
        """
        self.cur_attempt = AttemptLog(len(self.attempt_seq), time.time())
        self.cur_attempt.action_seq = []
        self.cur_attempt.results = []

    def finish_attempt(self, results, action_seq=None):
        """
        Mark the current attempt as finished.

        :param attempt_success: Whether or not the attempt is a success.
        :param results: Environment results.
        :param reward: reward received for this attempt
        :return: Nothing.
        """
        if action_seq is None:
            action_seq = self.cur_attempt.action_seq
        # todo: refactor this; hacky way to deal
        action_seq_str = [str(x) for x in action_seq]
        solutions_str = [[str(x) for x in solution] for solution in self.solutions]
        completed_solutions_str = [[str(x) for x in solution] for solution in self.completed_solutions]
        # check to see if this attempt is a solution that has not been completed already
        if action_seq_str in solutions_str and action_seq_str not in completed_solutions_str:
            attempt_success = True
            self.completed_solutions.append(self.cur_attempt.action_seq)
        else:
            attempt_success = False
        self.solution_found.append(attempt_success)
        self.cur_attempt.finish(attempt_success, results, time.time())
        self.attempt_seq.append(copy.deepcopy(self.cur_attempt))
        self.success = len(self.solutions) == len(self.completed_solutions)
        self.cur_attempt = None
        return attempt_success

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

    @property
    def num_attempts_since_last_solution_found(self):
        count = 0
        for solution_found in reversed(self.solution_found):
            # skip solution we just found
            if count == 0:
                count += 1
                continue
            if solution_found is True:
                return count
            count += 1
        # if we hit here, we are at the first solution
        return count
