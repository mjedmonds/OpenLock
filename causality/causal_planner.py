import numpy as np
import common
import copy
import sys


class ActionSequence:
    def __init__(self, action_seq, action_labels=None):
        self.action_seq = action_seq
        if action_labels is not None:
            self.action_seq_str = self.to_string(action_labels)
        else:
            self.action_seq_str = None

    def __len__(self):
        return len(self.action_seq)

    def __iter__(self):
        return iter(self.action_seq)

    def to_string(self, action_labels):
        result = []
        for action_idx in self.action_seq:
            result.append(action_labels[action_idx])
        return result

    def concat(self, other):
        if self.action_seq_str is not None and other.action_seq_str is not None:
            return ActionSequence(self.action_seq + other.action_seq, self.action_seq_str + other.action_seq_str)
        else:
            return ActionSequence(self.action_seq + other.action_seq)

class CausalPlanner:

    def __init__(self, fluent_labels, action_labels, perceptual_model):
        self.fluent_labels = fluent_labels
        self.action_labels = action_labels
        self.perceptual_model = perceptual_model
        self.n_fluents = len(fluent_labels)
        self.known_action_seqs = None
        self.possible_action_seqs = None

    def compute_action_seqs(self, fluent_vecs, action_vecs):
        """
        determines the known action_seqs from the human demonstration. for every unreachable, unknown action_seq, compute closest transitions to the unreachable states.
         Then uses the perceptual model to propose action sequeneces to arrive at that fluent state
        :param fluent_vecs: fluent vectors from demonstration
        :param action_vecs: action vectors from demonstration
        :return: a dict of possible action_seqs to bring the sequence to the the unreachable states
        """

        self.known_action_seqs = self.extract_reachable_states_from_demonstration(fluent_vecs, action_vecs)

        # build shortest action seqs
        self.compute_shortest_action_seqs()

        reachable_fluent_states = self.known_action_seqs.keys()
        unreachable_fluent_states = [x for x in range(pow(2, self.n_fluents)) if x not in reachable_fluent_states]

        self.possible_action_seqs = self.compute_possible_action_seqs(unreachable_fluent_states)

    def shortest_known_action_seq(self, starting_action):
        known_starting_action_seqs = self.known_action_seqs[starting_action]

        shortest_len = sys.maxsize
        shortest_idx = -1
        for i in range(len(known_starting_action_seqs)):
            if len(known_starting_action_seqs[i]) < shortest_len:
                shortest_idx = i
                shortest_len = len(known_starting_action_seqs[i])

        return known_starting_action_seqs[shortest_idx]

    def compute_shortest_action_seqs(self):
        self.known_shortest_action_seqs = dict()
        for known_starting_fluent in self.known_action_seqs.keys():
            self.known_shortest_action_seqs[known_starting_fluent] = self.shortest_known_action_seq(known_starting_fluent)

    def compute_possible_complete_action_seqs(self):
        possible_complete_plans = dict()
        for unreachable_fluent_state in self.possible_action_seqs.keys():
            possible_tuples = self.possible_action_seqs[unreachable_fluent_state]

            # construct complete action sequences to execute. These will be used to check if the unreachable state can be reached
            possible_complete_action_seqs = []
            for possible_tuple in possible_tuples:
                starting_action = possible_tuple[0]
                possible_action_seqs_list = possible_tuple[1]
                known_shortest_action_seq = self.known_shortest_action_seqs[starting_action]

                possible_complete_action_seqs.extend(self.compute_complete_action_seqs(known_shortest_action_seq, possible_action_seqs_list))

            possible_complete_plans[unreachable_fluent_state] = possible_complete_action_seqs

        return possible_complete_plans

    def compute_complete_action_seqs(self, known_shortest_action_seq, possible_action_seqs_list):
        return [known_shortest_action_seq.concat(x) for x in possible_action_seqs_list]

    def extract_reachable_states_from_demonstration(self, fluents, actions):
        """
        determines all reachable states from a demonstration and the corresponding action sequences to reach each state.
        :param fluents: 2d-array of fluent states observed in the demonstration
        :param actions: 2d-array of actions executed in the demonstration
        :return: known_action_seqs: contains a linear index for each fluent state reachable and a list of action sequences capable of reaching the state
        """

        # setup fluent space
        fluent_space = common.tabulate(self.fluent_labels)

        # setup known action_seqs
        known_action_seqs = dict()
        action_seq = []
        for i in range(0, fluents.shape[0]):
            fluent_vec = fluents[i]

            if i == 0:
                self.initial_fluent_state = fluent_vec
                continue

            prev_action_val = actions[i - 1]

            lin_fluent_vec = common.linearize_fluent_vec(fluents[i])
            action_executed = np.where(prev_action_val > 0)[0]  # find action executed at last frame
            assert (action_executed.size <= 1), "More than one action in a single frame, should be impossible"
            if action_executed.size == 1:
                action_seq.append(action_executed[0])
            else:
                continue  # no action executed

            # add on the current action as a way to reach this fluent state
            if lin_fluent_vec in known_action_seqs.keys():
                known_action_seqs[lin_fluent_vec].append(ActionSequence(copy.copy(action_seq)))
            else:
                known_action_seqs[lin_fluent_vec] = [ActionSequence(copy.copy(action_seq))]

        return known_action_seqs

    def compute_possible_action_seqs(self, unreachable_fluent_states):
        """
        uses the known_action_seqs to compute closest transitions to the unreachable states, then uses the perceptual model to propose action sequeneces to arrive at that fluent state
        :param unreachable_fluent_states: list of linear fluent states that are unreachable
        :param known_action_seqs: list of known fluent states that are reachable and corresponding action_seqs to achieve them
        :param perceptual_model: the perceptually causal model
        :return: a dict of possible action_seqs to bring the sequence to the the unreachable states
                 each key is an unreachable state, each value is a tuple between a starting (known) fluent and a list of possible action sequences
        """

        # each key is an unreachable state, each value is a tuple between a starting (known) fluent and a list of possible action sequences
        unreachable_fluent_to_possible_action_seq = dict()
        for unreachable_fluent in unreachable_fluent_states:
            unreachable_fluent_vec = common.delinearize_fluent_vec(unreachable_fluent, self.n_fluents)

            # find the closest reachable fluents to unreachable fluent states
            distances, starting_fluents = self.compute_closest_fluents(unreachable_fluent_vec, self.known_action_seqs)

            possible_action_seqs = []
            # compute possible action_seqs using each starting reachable fluent
            for starting_fluent in starting_fluents:
                starting_fluent_vec = common.delinearize_fluent_vec(starting_fluent, self.n_fluents)

                possible_action_seqs.append((starting_fluent, self.compute_perceptual_action_seq(starting_fluent_vec, unreachable_fluent_vec)))

            unreachable_fluent_to_possible_action_seq[unreachable_fluent] = possible_action_seqs

        return unreachable_fluent_to_possible_action_seq

    def compute_perceptual_action_seq(self, starting_fluent_vec, unreachable_fluent_vec, max_depth=5):
        """
        computes possible action_seqs from each starting_fluent (known, reachable action_seqs) to the unreachable fluents using transitions in the perceptual model
        :param starting_fluent_vec: fluents that can be reached, sorted in order from their distance to the unreachable_fleunt_vec
        :param unreachable_fluent_vec: the fluent vector that cannot be reached
        :return: perceptual_action_seq: the shortest perceptual action_seq from the starting_fluent_vec to the unreachable_fluent_vec
        """

        action_seqs = []

        action_seqs = self.compute_perceptual_action_seq_bfs(starting_fluent_vec, unreachable_fluent_vec, action_seqs, max_depth)

        action_seq_str = []
        for action_seq in action_seqs:
            action_seq_str.append(action_seq.to_string(self.perceptual_model.action_labels))

        return action_seqs

    def compute_perceptual_action_seq_bfs(self, starting_vec, target_vec, action_seq, max_depth=5):
        frontier = [(starting_vec, [])]
        depth = 0

        possible_action_seqs = []
        while frontier:
            if depth >= max_depth:
                break
            parent_vec, action_seq = frontier.pop(0)
            children_vecs, transition_actions = self.compute_perceptual_transition(parent_vec)
            for i in range(children_vecs.shape[0]):
                new_action_seq = copy.copy(action_seq)
                new_action_seq.append(transition_actions[i])
                # add as a possible path
                if np.equal(children_vecs[i], target_vec).all():
                    possible_action_seqs.append(ActionSequence(new_action_seq))
                # continue searching down the tree
                else:
                    frontier.append((children_vecs[i], new_action_seq))

            depth += 1

        return possible_action_seqs

    def compute_perceptual_transition(self, fluent_vec):
        """
        computes all possible perceptual transitions from fluent_vec using perceptual model
        :param fluent_vec: initial fluent vec
        :return: new_fluent_vecs: modified fluent_vec by taking action in corresponding actions array
                 actions: action to take to cause corresponding transition in fluent_vec to new_fluent_vecs
        """

        new_fluent_vecs = []
        actions = []

        # enumerate all possible transitions
        for fluent_idx in range(fluent_vec.size):
            fluent_val = fluent_vec[fluent_idx]

            # assign fluent change type by the fluent val. 1 == cur:1, next:0, 2 == cur:0, next:1
            fluent_change_type = 1 if fluent_val == 1 else 2

            # collect perceptually causal transitions that have this fluent transition
            transition_idx = np.where(np.logical_and(self.perceptual_model.fluents == fluent_idx, self.perceptual_model.fluent_change_types == fluent_change_type))
            if len(transition_idx) == 0:
                continue
            else:
                transition_idx = transition_idx[0]
            action = self.perceptual_model.actions[transition_idx][0]

            actions.append(action)
            new_fluent_vec = copy.copy(fluent_vec)
            # switch the state in the new fluent vector
            new_fluent_vec[fluent_idx] = 0 if new_fluent_vec[fluent_idx] == 1 else 1
            new_fluent_vecs.append(new_fluent_vec)

        return np.array(new_fluent_vecs), np.array(actions)

    def compute_closest_fluents(self, unreachable_fluent_vec, known_action_seqs):
        """
        computes the closest fluents to the unreachable fluent vec
        :param unreachable_fluent_vec: the unreachable (target) fluent vec
        :param known_action_seqs: reachable fluent vecs and their action sequences
        :return:
        """

        starting_fluent_vec = []
        distances = []
        for known_action_seq in known_action_seqs.keys():
            known_fluent_vec = common.delinearize_fluent_vec(known_action_seq, self.n_fluents)
            dist = self.fluent_dist(unreachable_fluent_vec, known_fluent_vec)
            starting_fluent_vec.append(known_action_seq)
            distances.append(dist)

        # sort according to shortest distance
        distances = np.array(distances)
        starting_fluent_vec = np.array(starting_fluent_vec)
        arg_order = distances.argsort()

        distances = distances[arg_order]
        starting_fluent_vec = starting_fluent_vec[arg_order]

        return distances, starting_fluent_vec

    @staticmethod
    def fluent_dist(fluent1, fluent2):
        """
        computes the distance between two binary fluent vectors
        :param fluent1: first fluent vector
        :param fluent2: second fluent vector
        :return: the minimum number of fluent transitions necessary to convert fluent1 into fluent2
        """
        assert(fluent1.shape == fluent2.shape), "attempting to compute distance between two fluent vectors with different lengths"
        dist = 0
        for i in range(fluent1.size):
            # if they are not equal, a transition must take place to make them equal
            if fluent1[i] != fluent2[i]:
                dist += 1

        return dist

def load_trial(demonstration_file, perceptual_file):

    data, col_labels, fluent_vecs, action_vecs, fluent_labels, action_labels = load_csv(demonstration_file)
    perceptual_model = common.PerceptualModel(perceptual_file)

    #perceptual_model.pretty_print()

    causal_planner = CausalPlanner(fluent_labels, action_labels, perceptual_model)

    causal_planner.compute_action_seqs(fluent_vecs, action_vecs)

    return causal_planner

def main():
    data_dir = '../scenario_outputs/action_reversal/'
    trial_name = 'ex1_extended'
    demonstration_file = data_dir + trial_name + '.csv'
    perceptual_file = data_dir + 'output_node_' + trial_name + '.mat'

    known_action_seqs, possible_action_seqs, causal_planner = load_trial(demonstration_file, perceptual_file)

    print("All done!")


def load_csv(demonstration_file):
    """
    loads the output of the a simulation demonstration
    :param demonstration_file:
    :return: data: the full data matrix
             col_lables: labels for each column in the data matrix
             fluents: the fluent values observed in the demonstration
             actions: the actions executed in the demonstration
             fluent_labels: column labels for fluents
             action_labels: column labels for actions
    """
    data = np.loadtxt(demonstration_file, delimiter=',', skiprows=1)

    with open(demonstration_file, 'r') as f:
        col_labels = f.readline()
        col_labels = col_labels.split(',')

    col_labels = [x.strip('\r\n') for x in col_labels]

    agent_idx = col_labels.index('agent')
    fluent_labels = col_labels[1:agent_idx]
    fluents = data[:, 1:agent_idx]
    action_labels = col_labels[agent_idx + 1:]
    actions = data[:, agent_idx + 1:]

    return data, col_labels, fluents, actions, fluent_labels, action_labels

if __name__ == '__main__':
    main()
