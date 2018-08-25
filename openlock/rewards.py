from openlock.common import ENTITY_STATES
import numpy as np


class RewardStrategy(object):

    def __init__(self):
        self.REWARD_NONE = 0
        self.REWARD_CHANGE_OBS = 0.5
        self.REWARD_IMMOVABLE = -0.5
        self.REWARD_REPEATED_ACTION = -0.25
        self.REWARD_PARTIAL_SEQ = 1
        self.REWARD_UNLOCK = 10
        self.REWARD_OPEN = 50
        self.REWARD_DOOR_SEQ = 0.5
        self.SOLUTION_MULTIPLIER = 1.5
        self.counter = np.zeros(10)
        self.attempt_count = 0

    def determine_multiplier(self, env, action):
        def get_solution_index(solutions, action_seq):
            # get the index of cur_action_seq in solutions, if none, return -1
            for ind in range(len(solutions)):
                comparison = [solutions[ind][i] == action_seq[i] for i in range(len(action_seq))]
                if all(comparison):
                    return ind
            return -1

        self.SOLUTION_MULTIPLIER = 1.0

        completed_solutions = env.get_completed_solutions()
        solutions = env.get_solutions()
        cur_action_seq = env.get_current_action_seq()

        num_solutions_found = len(completed_solutions)
        unique_seq = env.determine_unique_solution() or env.determine_unique_partial_solution()
        index = get_solution_index(solutions, cur_action_seq)
        first_solution_index = -1

        if num_solutions_found != 0:
            first_solution_index = get_solution_index(solutions, completed_solutions[0])

        if unique_seq and (self.door_open(env, action) or self.door_unlocked_partial_action_seq(env)) and num_solutions_found == 0:
            # if see a unique solution,  set the counter, init
            self.SOLUTION_MULTIPLIER = 1.0 # set multiplier to 1.0 for the first time

        if index != -1 and num_solutions_found != 0 and index != first_solution_index and (self.door_open(env, action) or self.door_unlocked_partial_action_seq(env)):
            # if already seen this solution, cool the temperature, increase the counter
            self.counter[index] += 1
            cooling_percentage = self.counter[index]/(env.attempt_limit*0.3) # set threshold as 0.3 * attempt
            self.SOLUTION_MULTIPLIER = max(1.5 - 0.5*cooling_percentage, 1.0) # if smaller than 1.0, set as 1.0
        #print self.counter, self.SOLUTION_MULTIPLIER
        self.SOLUTION_MULTIPLIER = 10.

    def determine_reward(self, env, action, reward_mode):
        reward = 0
        self.attempt_count += 1
        if self.attempt_count > env.attempt_limit*3:
            self.counter = np.zeros(len(env.get_solutions()))
            self.attempt_count = 0
        self.determine_multiplier(env, action)
        door_open = self.door_open(env, action)
        if reward_mode == 'basic':
            reward = self.reward_basic(env, action)
        elif reward_mode == 'change_state':
            reward = self.reward_change_state(env, action)
        elif reward_mode == 'unique_solutions':
            reward = self.reward_unique_solution(env, action)
        elif reward_mode == 'change_state_unique_solutions':
            reward = self.reward_change_state_unique_solution(env, action)
        elif reward_mode == 'negative_immovable':
            reward = self.reward_negative_immovable(env, action)
        elif reward_mode == 'negative_immovable_unique_solutions':
            reward = self.reward_negative_immovable_unique_solutions(env, action)
        elif reward_mode == 'negative_immovable_partial_action_seq':
            reward = self.reward_negative_immovable_partial_seq(env, action)
        elif reward_mode == 'negative_immovable_negative_repeat':
            reward = self.reward_negative_immovable_negative_repeat(env, action)
        elif reward_mode == 'negative_immovable_solution_multiplier':
            reward = self.reward_negative_immovable_solution_multiplier(env, action)
        elif reward_mode == 'negative_immovable_partial_action_seq_solution_multiplier':
            reward = self.reward_negative_immovable_partial_seq_solution_multiplier(env, action)
        elif reward_mode == 'negative_immovable_partial_action_seq_solution_multiplier_door_seq':
            reward = self.reward_negative_immovable_partial_seq_solution_multiplier_door_seq(env, action)
        else:
            raise ValueError(str('Unknown reward function mode: %s'.format(reward_mode)))

        return reward, door_open

    def door_open(self, env, action):
        if self.door_unlocked(env) and action.name is 'push' and action.obj is 'door':
            return True
        else:
            return False

    def door_unlocked(self, env):
        door_lock_state = env.get_state()['OBJ_STATES']['door_lock']
        if door_lock_state == ENTITY_STATES['DOOR_UNLOCKED']:
            return True
        else:
            return False

    def door_unlocked_partial_action_seq(self, env):
        '''
        Determines if door was unlocked by this action, rather than a previous action
        Will also return true if the door is open, so checking if door_open should happen before this check
        This is the and of the door being unlocked and the current action seq being part of a full solution.
        For circumstances with more complex structure, this may return true even though the previous action did not unlock the door
        :param env: environment
        :return: True or False depending on whether or not this action
        '''
        if self.door_unlocked(env) and env.determine_partial_solution():
            return True
        else:
            return False

    def reward_basic(self, env, action):
        '''
        Give reward of REWARD_OPEN for opening the door
        Give reward of REWARD_UNLOCK for unlocking the door
        Give reward of REWARD_NONE for anything else
        '''
        # door unlocked and pushed on door
        if self.door_open(env, action):
            reward = self.REWARD_OPEN
        # door unlocked
        elif self.door_unlocked_partial_action_seq(env):
            reward = self.REWARD_UNLOCK
        # door locked
        else:
            reward = self.REWARD_NONE
        return reward

    def reward_change_state(self,env, action):
        '''
        Give reward of REWARD_OPEN for opening the door
        Give reward of REWARD_UNLOCK for unlocking the door
        Give reward of REWARD_CHANGE_OBS for changing the observation state
        Give reward of REWARD_NONE for anything else
        '''
        # door unlocked, push_door
        if self.door_open(env, action):
            reward = self.REWARD_OPEN
        # door unlocked
        elif self.door_unlocked_partial_action_seq(env):
            reward = self.REWARD_UNLOCK
        # state change
        elif env.determine_fluent_change():
            reward = self.REWARD_CHANGE_OBS
        # door locked, no state change
        else:
            reward = self.REWARD_NONE
        return reward

    def reward_unique_solution(self, env, action):
        '''
        Give reward of REWARD_OPEN for opening the door with a new action sequence
        Give reward of REWARD_UNLOCK for unlocking the door with a new action sequence
        Give reward of REWARD_NONE for anything else
        '''
        unique_seq = env.determine_unique_solution() or env.determine_unique_partial_solution()
        # door unlocked, push_door
        if self.door_open(env, action) and unique_seq:
            reward = self.REWARD_OPEN
        # door unlocked, unique solution
        elif self.door_unlocked_partial_action_seq(env) and unique_seq:
            reward = self.REWARD_UNLOCK
        # door locked, no state change
        else:
            reward = self.REWARD_NONE
        return reward

    def reward_change_state_unique_solution(self, env, action):
        '''
        Give reward of REWARD_OPEN for opening the door with a new action sequence
        Give reward of REWARD_UNLOCK for unlocking the door with a new action sequence
        Give reward of REWARD_CHANGE_OBS for changing the observation state
        Give reward of REWARD_NONE for anything else
        '''
        unique_seq = env.determine_unique_solution() or env.determine_unique_partial_solution()
        # door locked, state change
        if self.door_open(env, action) and unique_seq:
            reward = self.REWARD_OPEN
        # door unlocked
        elif self.door_unlocked_partial_action_seq(env) and unique_seq:
            reward = self.REWARD_UNLOCK
        # state change
        elif env.determine_fluent_change():
            reward = self.REWARD_CHANGE_OBS
        # door locked, no state change
        else:
            reward = self.REWARD_NONE
        return reward

    def reward_negative_immovable_unique_solutions(self, env, action):
        '''
        Give reward of REWARD_OPEN for opening the door with a new action sequence
        Give reward of REWARD_UNLOCK for unlocking the door with a new action sequence
        Give reward of REWARD_IMMOVABLE for not changing the observation state
        Give reward of REWARD_NONE for anything else
        '''
        unique_seq = env.determine_unique_solution() or env.determine_unique_partial_solution()
        # door locked, state change
        if self.door_open(env, action) and unique_seq:
            reward = self.REWARD_OPEN
        # door unlocked
        elif self.door_unlocked_partial_action_seq(env) and unique_seq:
            reward = self.REWARD_UNLOCK
        # determine if movable
        elif not env.determine_fluent_change():
            reward = self.REWARD_IMMOVABLE
        # door locked, no state change
        else:
            reward = self.REWARD_NONE
        return reward

    def reward_negative_immovable(self, env, action):
        '''
        Give reward of REWARD_OPEN for opening the door
        Give reward of REWARD_UNLOCK for unlocking the door
        Give reward of REWARD_IMMOVABLE for not changing the observation state
        Give reward of REWARD_NONE for anything else
        '''
        # door locked, state change
        if self.door_open(env, action):
            reward = self.REWARD_OPEN
        # door unlocked
        elif self.door_unlocked_partial_action_seq(env):
            reward = self.REWARD_UNLOCK
        # determine if movable
        elif not env.determine_fluent_change():
            reward = self.REWARD_IMMOVABLE
        # door locked, no state change
        else:
            reward = self.REWARD_NONE
        return reward

    def reward_negative_immovable_partial_seq(self, env, action):
        '''
        Give reward of REWARD_OPEN for opening the door with
        Give reward of REWARD_UNLOCK for unlocking the door
        Give reward of REWARD_PARTIAL_SEQ for any partial subsequence
        Give reward of REWARD_IMMOVABLE for not changing the observation state
        Give reward of REWARD_NONE for anything else
        '''
        # door locked, state change
        if self.door_open(env, action):
            reward = self.REWARD_OPEN
        # door unlocked
        elif self.door_unlocked_partial_action_seq(env):
            reward = self.REWARD_UNLOCK
        # determine if partial subsequence of a solution action seq
        elif env.determine_partial_solution():
            reward = self.REWARD_PARTIAL_SEQ
        # determine if movable
        elif not env.determine_fluent_change():
            reward = self.REWARD_IMMOVABLE
        # door locked, no state change
        else:
            reward = self.REWARD_NONE
        return reward

    def reward_negative_immovable_negative_repeat(self, env, action):
        '''
        Give reward of REWARD_OPEN for opening the door
        Give reward of REWARD_UNLOCK for unlocking the door
        Give reward of REWARD_REPEATED_ACTION for successively performing 2 same actions
        Give reward of REWARD_IMMOVABLE for not changing the observation state
        Give reward of REWARD_NONE for anything else
        '''
        # door locked, state change
        if self.door_open(env, action):
            reward = self.REWARD_OPEN
        # door unlocked
        elif self.door_unlocked_partial_action_seq(env):
            reward = self.REWARD_UNLOCK
        # determine if the last two actions are the same
        elif env.determine_repeated_action():
            reward = self.REWARD_REPEATED_ACTION
        # determine if movable
        elif not env.determine_fluent_change():
            reward = self.REWARD_IMMOVABLE
        # door locked, no state change
        else:
            reward = self.REWARD_NONE
        return reward

    def reward_negative_immovable_solution_multiplier(self, env, action):
        '''
        Give reward of REWARD_OPEN*multiplier for opening the door
        Give reward of REWARD_UNLOCK*multiplier for unlocking the door
        Give reward of REWARD_IMMOVABLE for not changing the observation state
        Give reward of REWARD_NONE for anything else

        Each new, unique solution found is multiplied by 1.5x the previous multiplier.
        The multiplier starts at 1. For instance, the reward for finding the first unique
        solution would be REWARD_OPEN, but the reward for finding the section unique solution
        would be 1.5 * REWARD_OPEN, the third 2 * 1.5 * REWARD_OPEN. This encourages the
        agent to find unique solutions without penalizing for finding repeated solutions
        '''
        num_solutions_found = len(env.get_completed_solutions())
        multiplier = self.SOLUTION_MULTIPLIER # max(1, 1 * self.SOLUTION_MULTIPLIER * num_solutions_found)
        unique_seq = env.determine_unique_solution() or env.determine_unique_partial_solution()
        # door unlocked
        if self.door_open(env, action) and unique_seq:
            reward = self.REWARD_OPEN * multiplier
        # door unlocked
        elif self.door_open(env, action):
            reward = self.REWARD_OPEN
        # door unlocked
        elif self.door_unlocked_partial_action_seq(env) and unique_seq:
            reward = self.REWARD_UNLOCK * multiplier
        # door unlocked
        elif self.door_unlocked_partial_action_seq(env):
            reward = self.REWARD_UNLOCK
        # determine if movable
        elif not env.determine_fluent_change():
            reward = self.REWARD_IMMOVABLE
        # door locked, no state change
        else:
            reward = self.REWARD_NONE
        return reward

    def reward_negative_immovable_partial_seq_solution_multiplier(self, env, action):
        '''
        Give reward of REWARD_OPEN*multiplier for opening the door
        Give reward of REWARD_UNLOCK*multiplier for unlocking the door
        Give reward of REWARD_PARTIAL_SEQ for any partial subsequence
        Give reward of REWARD_IMMOVABLE for not changing the observation state
        Give reward of REWARD_NONE for anything else

        Each new, unique solution found is multiplied by 1.5x the previous multiplier.
        The multiplier starts at 1. For instance, the reward for finding the first unique
        solution would be REWARD_OPEN, but the reward for finding the section unique solution
        would be 1.5 * REWARD_OPEN, the third 2 * 1.5 * REWARD_OPEN. This encourages the
        agent to find unique solutions without penalizing for finding repeated solutions
        '''

        num_solutions_found = len(env.get_completed_solutions())
        multiplier = self.SOLUTION_MULTIPLIER # max(1, 1 * self.SOLUTION_MULTIPLIER * num_solutions_found)
        unique_seq = env.determine_unique_solution() or env.determine_unique_partial_solution()
        # door unlocked
        if self.door_open(env, action) and unique_seq:
            reward = self.REWARD_OPEN * multiplier
        # door unlocked
        elif self.door_open(env, action):
            reward = self.REWARD_OPEN
        # door unlocked
        elif self.door_unlocked_partial_action_seq(env) and unique_seq:
            reward = self.REWARD_UNLOCK * multiplier
        # door unlocked
        elif self.door_unlocked_partial_action_seq(env):
            reward = self.REWARD_UNLOCK
        # determine if partial subsequence of a solution action seq
        elif env.determine_partial_solution():
            reward = self.REWARD_PARTIAL_SEQ
        # state change
        elif not env.determine_fluent_change():
            reward = self.REWARD_IMMOVABLE
        # door locked, no state change
        else:
            reward = self.REWARD_NONE
        return reward

    def reward_negative_immovable_partial_seq_solution_multiplier_door_seq(self, env, action):
        '''
        Give reward of REWARD_OPEN for opening the door with a new action sequence
        Give reward of REWARD_UNLOCK for unlocking the door with a new action sequence
        Give reward of REWARD_PARTIAL_SEQ for any partial subsequence
        Give reward of REWARD_IMMOVABLE for not changing the observation state
        Give reward of REWARD_DOOR_SEQ for performing an push_door action
        Give reward of REWARD_NONE for anything else

        Each new, unique solution found is multiplied by 1.5x the previous multiplier.
        The multiplier starts at 1. For instance, the reward for finding the first unique
        solution would be REWARD_OPEN, but the reward for finding the section unique solution
        would be 1.5 * REWARD_OPEN, the third 2 * 1.5 * REWARD_OPEN. This encourages the
        agent to find unique solutions without penalizing for finding repeated solutions
        '''

        num_solutions_found = len(env.get_completed_solutions())
        multiplier = max(1, 1 * self.SOLUTION_MULTIPLIER * num_solutions_found)
        unique_seq = env.determine_unique_solution() or env.determine_unique_partial_solution()
        # door unlocked
        if self.door_open(env, action) and unique_seq:
            reward = self.REWARD_OPEN * multiplier
        # door unlocked
        elif self.door_open(env, action):
            reward = self.REWARD_OPEN
        # door unlocked
        elif self.door_unlocked_partial_action_seq(env) and unique_seq:
            reward = self.REWARD_UNLOCK * multiplier
        # door unlocked
        elif self.door_unlocked_partial_action_seq(env):
            reward = self.REWARD_UNLOCK
        # determine if partial subsequence of a solution action seq
        elif env.determine_partial_solution():
            reward = self.REWARD_PARTIAL_SEQ
        # state change
        elif not env.determine_fluent_change():
            reward = self.REWARD_IMMOVABLE
        # door locked, no state change
        else:
            reward = self.REWARD_NONE
        # determine if the door_seq is right
            reward += env.determine_door_seq()*self.REWARD_DOOR_SEQ
        return reward
