import numpy as np
import scipy.io


class CausalNode:
    def __init__(self, fluent, action, fluent_change_type, info_gain):
        self.fluent = fluent
        self.action = action
        self.fluent_change_type = fluent_change_type
        self.info_gain = info_gain


class PerceptualModel:

    def __init__(self, mat_path):
        """
        converts the perceptual causality output into a python object
        :param mat_path: the path to the mat file
        """
        mat = scipy.io.loadmat(mat_path)

        action_labels = mat['actions'][0].tolist()
        fluent_change_labels = mat['fluent_change_labels'][0].tolist()
        fluent_labels = mat['fluents'][0].tolist()
        node_actions = mat['node_actions'][0]
        node_fluent_change_types = mat['node_fluent_change_types'][0]
        node_info_gains = mat['node_info_gains'][0]
        node_objects = mat['node_objects'][0]

        # clean cell array strings
        for i in range(len(action_labels)):
            action_labels[i] = str(action_labels[i][0])
        for i in range(len(fluent_change_labels)):
            fluent_change_labels[i] = str(fluent_change_labels[i][0])
        for i in range(len(fluent_labels)):
            fluent_labels[i] = str(fluent_labels[i][0])
        for i in range(len(node_objects)):
            node_objects[i] = str(node_objects[i][0])
            # convert to integer representation
            node_objects[i] = fluent_labels.index(node_objects[i])

        # convert matlab indexing to python indexing
        for i in range(len(node_actions)):
            node_actions[i] -= 1
        for i in range(len(node_fluent_change_types)):
            node_fluent_change_types[i] -= 1

        self.action_labels = np.array(action_labels)
        self.fluent_labels = np.array(fluent_labels)
        self.fluent_change_labels = np.array(fluent_change_labels)
        self.actions = np.array(node_actions)
        self.fluent_change_types = np.array(node_fluent_change_types)
        self.info_gains = np.array(node_info_gains)
        self.fluents = np.array(node_objects)

    def pretty_print(self):
        print("Perceptual model:")
        for i in range(self.actions.shape[0]):
            action_idx = self.actions[i]
            action_label = self.action_labels[action_idx]
            fluent_change_type = self.fluent_change_types[i]
            fluent_change_label = self.fluent_change_labels[fluent_change_type]
            fluent_label = self.fluent_labels[self.fluents[i]]

            print("\t{}:\t{}\t{}".format(action_label, fluent_label, fluent_change_label))


def delinearize_fluent_vec(index, n_fluents):
    """
    auxiliary function for recursive fluent-delinearizer
    :param index: the linear fluent value to delinearize
    :param n_fluents: the total number of fluents
    :return: a binary vector of the delinearized fluent value
    """

    fluent_vals = np.zeros((n_fluents,), dtype=int)

    return delinearize_fluent_vec_rec(index, n_fluents-1, fluent_vals)


# helper function to convert linear index to a vector
def delinearize_fluent_vec_rec(index, col, fluent_vals):
    """
    converts a linear index into a binary vector of fluent values
    :param index: the current linear index
    :param col: the current column in the binary vector
    :param fluent_vals: the binary vector
    :return: the binary vector representation of the index
    """
    if index >= pow(2, col):
        fluent_idx = len(fluent_vals) - col
        fluent_vals[fluent_idx-1] = 1
        new_index = index - pow(2, col)
    else:
        new_index = index

    if new_index > 0 and col > 0:
        fluent_vals = delinearize_fluent_vec_rec(new_index, col-1, fluent_vals)

    return fluent_vals

def linearize_fluent_vec(fluent_vals):
    """
    converts as binary vector of fluents into a linear index
    :param fluent_vals: the binary vector of fluents
    :return: the corresponding linear integer
    """
    index = 0
    for i in range(fluent_vals.size):
        column = fluent_vals.size - i - 1
        fluent_val = fluent_vals[i]
        index = index + pow(2, column) * fluent_val

    return int(index)

def tabulate(fluents):
    """
    builds a full binary truth table to enumerate all possible fluent states
    :param fluents: an array with the number of fluents as the length (i.e. the labels)
    :return: the space of all possible binary fluent states
    """

    # enumerate possible fluent states
    n_fluents = len(fluents)
    fluent_space = np.zeros((pow(2, n_fluents), n_fluents), dtype=int)
    for i in range(pow(2, n_fluents)):
        fluent_space[i] = delinearize_fluent_vec(i, n_fluents)

    return fluent_space


