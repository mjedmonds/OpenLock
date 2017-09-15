import scipy.io
import numpy as np
import common as cc


def main():
    data_dir = '../OpenLock/scenario_outputs/action_reversal/output_node_'
    trial_name = 'ex1_extended'

    perceptual_model = cc.PerceptualModel(data_dir + trial_name + '.mat')

    # tabulate full fluent space
    fluent_space = cc.tabulate(perceptual_model.fluents)

    print("All done!")

if __name__ == '__main__':
    main()
