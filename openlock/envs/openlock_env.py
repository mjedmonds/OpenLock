import gym
import re
import copy
import time
import numpy as np
from Box2D import b2RayCastInput, b2RayCastOutput, b2Distance

from openlock.box2d_renderer import Box2DRenderer
import openlock.common as common
from openlock.envs.world_defs.openlock_def import ArmLockDef
from openlock.kine import KinematicChain, discretize_path, InverseKinematics, generate_five_arm, TwoDKinematicTransform
from openlock.settings_render import RENDER_SETTINGS, BOX2D_SETTINGS, ENV_SETTINGS
from openlock.rewards import RewardStrategy #determine_reward, REWARD_IMMOVABLE, REWARD_OPEN
from openlock.settings_trial import CONFIG_TO_IDX, NUM_LEVERS
from gym.spaces import MultiDiscrete
from openlock.logger_env import ActionLog


from glob import glob


# TODO: add ability to move base
# TODO: more physically plausible units?

class ActionSpace:

    def __init__(self):
        pass

    @staticmethod
    def create_action_space(obj_map):
        # this must be preallocated; they are filled by position, not by symbol
        push_action_space = [None] * NUM_LEVERS
        pull_action_space = [None] * NUM_LEVERS
        door_action_space = []
        action_map = dict()
        for obj, val in list(obj_map.items()):
            if 'button' not in obj and 'door' not in obj:
                # use position to map to an integer index
                twod_config = val.config
                lever_idx = CONFIG_TO_IDX[twod_config]

                push = 'push_{}'.format(obj)
                pull = 'pull_{}'.format(obj)

                push_action_space[lever_idx] = push
                pull_action_space[lever_idx] = pull

                action_map[push] = common.Action('push', obj, 4)
                action_map[pull] = common.Action('pull', obj, 4)
            if 'button' not in obj and 'door' in obj:
                push = 'push_{}'.format(obj)

                door_action_space.append(push)

                action_map[push] = common.Action('push', obj, 4)

        action_space = push_action_space + pull_action_space + door_action_space

        return action_space, action_map


class ObservationSpace:

    def __init__(self, num_levers, append_solutions_remaining=False):
        self.append_solutions_remaining = append_solutions_remaining
        self.solutions_found = [0, 0, 0]
        self.labels = ['sln0', 'sln1', 'sln2']

        if self.append_solutions_remaining:
            self.multi_discrete = self.create_observation_space(num_levers, len(self.solutions_found))
        else:
            self.multi_discrete = self.create_observation_space(num_levers)
        self.num_levers = num_levers
        self.state = None
        self.state_labels = None


    @property
    def shape(self):
        return self.multi_discrete.shape

    @staticmethod
    def create_observation_space(num_levers, num_solutions=0):
        discrete_space = []
        num_lever_states = 2
        num_lever_colors = 2
        num_door_states = 2
        num_door_lock_states = 2
        # first num_levers represent the state of the levers
        for i in range(num_levers):
            discrete_space.append(num_lever_states)
        # second num_levers represent the colors of the levers
        for i in range(num_levers):
            discrete_space.append(num_lever_colors)
        discrete_space.append(num_door_lock_states)       # door lock
        discrete_space.append(num_door_states)       # door open
        # solutions appended
        for i in range(num_solutions):
            # solutions can only be found or not found
            discrete_space.append(2)
        discrete_space = np.array(discrete_space)
        multi_discrete = MultiDiscrete(discrete_space)
        return multi_discrete

    def create_discrete_observation_from_simulator(self, env):
        '''
        Constructs a discrete observation from the physics simulator
        :param world_def:
        :return:
        '''
        levers = env.world_def.get_levers()
        self.num_levers = len(levers)
        world_state = env.world_def.get_state()

        # need one element for state and color of each lock, need two addition for door lock status and door status
        self.state = [None] * (self.num_levers * 2 + 2)
        self.state_labels = [None] * (self.num_levers * 2 + 2)

        for lever in levers:
            # convert to index based on lever position
            lever_idx = CONFIG_TO_IDX[lever.config]

            lever_state = np.int8(world_state['OBJ_STATES'][lever.name])
            lever_active = np.int8(lever.determine_active())

            self.state_labels[lever_idx] = lever.name
            self.state[lever_idx] = lever_state
            self.state_labels[lever_idx + self.num_levers] = lever.name + '_active'
            self.state[lever_idx + self.num_levers] = lever_active

        self.state_labels[-1] = 'door'
        self.state[-1] = np.int8(world_state['OBJ_STATES']['door'])
        self.state_labels[-2] = 'door_lock'
        self.state[-2] = np.int8(world_state['OBJ_STATES']['door_lock'])

        if self.append_solutions_remaining:
            slns_found, sln_labels = self.determine_solutions_remaining(env)
            self.state.extend(slns_found)
            self.state_labels.extend(sln_labels)

        return self.state, self.state_labels

    def create_discrete_observation_from_fsm(self, env):
        '''
        constructs a discrete observation from the underlying FSM
        Used when the physics simulator is being bypassed
        :param fsmm:
        :return:
        '''
        levers = env.scenario.levers
        self.num_levers = len(levers)
        scenario_state = env.scenario.get_state()

        # need one element for state and color of each lock, need two addition for door lock status and door status
        self.state = [None] * (self.num_levers * 2 + 2)
        self.state_labels = [None] * (self.num_levers * 2 + 2)

        inactive_lock_regex = '^inactive[0-9]+$'

        # lever states
        for lever in levers:
            lever_idx = CONFIG_TO_IDX[lever.config]

            # inactive lever, state is constant
            if re.search(inactive_lock_regex, lever.name):
                lever_active = np.int8(common.ENTITY_STATES['LEVER_INACTIVE'])
            else:
                lever_active = np.int8(common.ENTITY_STATES['LEVER_ACTIVE'])

            lever_state = np.int8(scenario_state['OBJ_STATES'][lever.name])

            self.state_labels[lever_idx] = lever.name
            self.state[lever_idx] = lever_state

            self.state_labels[lever_idx + self.num_levers] = lever.name + '_active'
            self.state[lever_idx + self.num_levers] = lever_active

        # update door state
        door_lock_name = 'door_lock'
        door_lock_state = np.int8(scenario_state['OBJ_STATES'][door_lock_name])

        # todo: this is a hack to get whether or not the door is actually open; it should be part of the FSM
        door_name = 'door'
        door_state = np.int8(scenario_state['OBJ_STATES'][door_name])

        self.state_labels[-1] = door_name
        self.state[-1] = door_state
        self.state_labels[-2] = door_lock_name
        self.state[-2] = door_lock_state

        if self.append_solutions_remaining:
            slns_found, sln_labels = self.determine_solutions_remaining(env)
            self.state.extend(slns_found)
            self.state_labels.extend(sln_labels)

        return self.state, self.state_labels

    def determine_solutions_remaining(self, logger):
        # todo: this is hardcored to scenarios with a max of 3 solutions
        solutions = logger.cur_trial.solutions
        completed_solutions = logger.cur_trial.completed_solutions
        for i in range(len(completed_solutions)):
            idx = solutions.index(completed_solutions[i])
            # mark that this solution is finished
            self.solutions_found[idx] = 1

        return self.solutions_found, self.labels


class OpenLockEnv(gym.Env):
    # Set this in SOME subclasses
    metadata = {'render.modes': ['human']}  # TODO what does this do?

    def __init__(self):
        self.viewer = None

        # handle to the scenario, defined by the scenario
        self.scenario = None

        self.i = 0
        self.save_path = '../OpenLockResults/'

        self.col_label = []
        self.index_map = None
        self.results = None

        self.attempt_count = 0  # keeps track of the number of attempts
        self.action_count = 0   # keeps track of the number of actions executed
        self.action_limit = None
        self.attempt_limit = None

        self.action_executing = False    # used to disable action preemption

        self.human_agent = True
        self.reward_mode = 'basic'

        self.resetting = False          # determines if env is currently resetting (pausing to user)

        self.observation_space = None
        self.reward_strategy = RewardStrategy()
        self.reward_range = (self.reward_strategy.REWARD_IMMOVABLE, self.reward_strategy.REWARD_OPEN)

        self.use_physics = True

        self.world_def = None

        self.full_attempt_limit = False

        # action acknowledgement used by manager to log each action executed
        self.action_ack = True
        self.action_finish_ack = True

        self.solutions = []            # keeps track of solutions for this trial/scenario
        self.completed_solutions = []  # keeps track of which solutions have been completed this trial
        self.cur_action_seq = []       # keeps track of the action sequence executed this attempt

    def reset(self):
        """Resets the state of the environment and returns an initial observation.

        Returns: observation (object): the initial observation of the
                space.
        """
        if self.scenario is None:
            print('WARNING: resetting environment with no scenario')

        self.clock = 0
        self._seed()

        if self.use_physics:
            self.init_inverse_kine()

            # setup Box2D world
            self.world_def = ArmLockDef(self.invkine.kinematic_chain, 1.0 / BOX2D_SETTINGS['FPS'], 30, self.scenario)

            obj_map = self.world_def.obj_map
            levers = self.world_def.get_levers()
        else:
            # initialize obj_map for scenario
            self.scenario.init_scenario_env()

            obj_map = self.scenario.obj_map
            # todo: this is a dirty hack to get the door in
            # todo: define a global configuration that includes levers and doors
            # add door because it is not originally in the map
            obj_map['door'] = 'door'
            levers = self.scenario.levers

        self.action_space, self.action_map = ActionSpace.create_action_space(obj_map)
        self.obs_space = ObservationSpace(len(levers))

        # reset results (must be after world_def exists and action space has been created)
        self._reset_results()

        if self.use_physics:
            # setup renderer
            if not self.viewer:
                self.viewer = Box2DRenderer(self._action_grasp)

            self.viewer.reset()

            self._create_clickable_regions()

        # reset the finite state machine
        self.scenario.reset()
        self.action_count = 0

        # reset the current action sequence
        self.cur_action_seq = []

        if self.use_physics:
            if self.human_agent:
                self.render()

        self.state = self.get_state()
        # append initial observation
        # self._print_observation(state, self.action_count)
        self._append_result(self._create_state_entry(self.state, self.action_count))
        
        self.update_state_machine()

        if self.observation_space is not None:
            if self.use_physics:
                discrete_state, discrete_labels = self.observation_space.create_discrete_observation_from_simulator(self)
            else:
                discrete_state, discrete_labels = self.observation_space.create_discrete_observation_from_fsm(self)
            return np.array(discrete_state)
        else:
            return None

    def step(self, action):
        """Run one __timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.
        Accepts an action and returns a tuple (observation, reward, done, info).
        Args:
                action (Action): desired Action
        Returns:
                observation (dict): END_EFFECTOR_POS : current end effector position
                                          LOCK_STATE : true if door is locked
                reward (float) : amount of reward returned after previous action
                done (boolean): whether the episode has ended, in which case further step() calls will return undefined results
                info (dict): CONVERGED : whether algorithm succesfully coverged on action
        """
        # save a copy of the current state
        self.prev_state = self.get_state()

        if not action:
            self.world_def.step(1.0 / BOX2D_SETTINGS['FPS'],
                                BOX2D_SETTINGS['VEL_ITERS'],
                                BOX2D_SETTINGS['POS_ITERS'])

            # self._render_world_at_frame_rate()

            self.state = self.get_state()
            self.state['SUCCESS'] = False
            self.update_state_machine()
            # no action, return nothing to indicate no reward possible
            return None
        # change to simple "else:" to enable action preemption
        elif self.action_executing is False and self.resetting is False:
            self.action_executing = True
            self.i += 1
            reset = False
            trial_finished = False
            observable_action = self._create_pre_obs_entry(action)
            if observable_action:
                # ack is used by manager to determine if the action needs to be logged in the agent's logger
                self.action = ActionLog(str(action), time.time())
                self.action_ack = False

            success = False
            if self.use_physics:
                if action.name == 'goto':
                    success = self._action_go_to(action)
                elif action.name == 'goto_obj':
                    success = self._action_go_to_obj(action)
                elif action.name == 'rest':
                    success = self._action_rest()
                elif action.name == 'pull':
                    success = self._action_pull(action)
                elif action.name == 'push':
                    success = self._action_push(action)
                elif action.name == 'move':
                    success = self._action_move(action)
                elif action.name == 'move_end_frame':
                    success = self._action_move_end_frame(action)
                elif action.name == 'unlock':
                    success = self._action_unlock(action)
                elif action.name == 'reset':
                    success = self._action_reset()
                elif action.name == 'save':
                    success = self._action_save()
            else:
                success = True
                self.scenario.execute_action(action)

            self.i += 1

            # update state machine after executing a action
            self.update_state_machine(action)
            self.state = self.get_state()
            self.state['SUCCESS'] = success

            if observable_action:
                self.action_count += 1

                # self._print_observation(self.state, self.action_count)
                self._append_result(self._create_state_entry(self.state, self.action_count))
                # self.results.append(self._create_state_entry(self.state, self.action_count))
                self.action.finish(time.time())
                self.action_finish_ack = False
                self.cur_action_seq.append(self.action)

            # must update reward before potentially reset env (env may reset based on trial status)
            reward, success = self.reward_strategy.determine_reward(self, action, self.reward_mode)

            self.action_executing = False

            if self.action_count >= self.action_limit:
                reset = True
                # todo: possible to check if trial is finished from within the env?

            # update state machine in case there was a reset
            self.update_state_machine()
            if self.use_physics:
                discrete_state, discrete_labels = self.observation_space.create_discrete_observation_from_simulator(self)
            else:
                discrete_state, discrete_labels = self.observation_space.create_discrete_observation_from_fsm(self)
            # set done = trial_finished to have done signaled to agent when trials end
            # set done = reset to have done signaled to agent every time env resets
            return np.array(discrete_state), reward, reset, {'action_success': success, 'trial_finished': trial_finished, 'state_labels': discrete_labels}
        else:
            self.state = self.get_state()
            self.update_state_machine()
            return None
            # return self.state, 0, False, {}

    def render(self, mode='human', close=False):
        """Renders the environment.

        The set of supported modes varies per environment. (And some
        environments do not support rendering at all.) By convention,
        if mode is:

        - human: render to the current display or terminal and
            return nothing. Usually for human consumption.
        - rgb_array: Return an numpy.ndarray with shape (x, y, 3),
            representing RGB values for an x-by-y pixel image, suitable
            for turning into a video.
        - ansi: Return a string (str) or StringIO.StringIO containing a
            terminal-style text representation. The text can include newlines
            and ANSI escape sequences (e.g. for colors).

        Note:
                Make sure that your class's metadata 'render.modes' key includes
                    the list of supported modes. It's recommended to call super()
                    in implementations to use the functionality of this method.

        Args:
                mode (str): the mode to render with
                close (bool): close all open renderings

        Example:

        class MyEnv(Env):
                metadata = {'render.modes': ['human', 'rgb_array']}

                def render(self, mode='human'):
                        if mode == 'rgb_array':
                                return np.array(...) # return RGB frame suitable for video
                        elif mode is 'human':
                                ... # pop up a window and render
                        else:
                                super(MyEnv, self).render(mode=mode) # just raise an exception
        """

        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
                return

        if self.viewer is not None:
            self.viewer.render_multiple_worlds([self.world_def.background, self.world_def.world], mode='human')

    def _seed(self, seed=None):
        """Sets the seed for this env's random number generator(s).

            Note:
                    Some environments use multiple pseudorandom number generators.
                    We want to capture all such seeds used in order to ensure that
                    there aren't accidental correlations between multiple generators.

            Returns:
                    list<bigint>: Returns the list of seeds used in this env's random
                        number generators. The first value in the list should be the
                        "main" seed, or the value which a reproducer should pass to
                        'seed'. Often, the main seed equals the provided 'seed', but
                        this won't be true if seed=None, for example.
            """
        pass

    def update_scenario(self, scenario):
        """
        Set the environment's scenario to the specified scenario.

        :param scenario: new scenario to use
        :return: Nothing
        """
        self.scenario = scenario
        self.solutions = scenario.solutions
        self.completed_solutions = []
        self.cur_action_seq = []

    def set_action_limit(self, action_limit):
        """
        Set self.env.action_limit.

        :param action_limit: new self.env.action_limit
        :return: Nothing
        """
        self.action_limit = action_limit

    def _create_state_entry(self, state, frame):
        entry = [0] * len(self.col_label)
        entry[0] = frame
        for name, val in list(state['OBJ_STATES'].items()):
            entry[self.index_map[name]] = int(val)

        return entry

    def _create_pre_obs_entry(self, action):
        # create pre-observation entry
        entry = [0] * len(self.col_label)
        entry[0] = self.action_count
        # copy over previous state
        entry[1:self.index_map['agent']+1] = copy.copy(self.results[-1][1:self.index_map['agent']+1])

        # mark action idx
        if type(action.obj) is str:
            col = '{}_{}'.format(action.name, action.obj)
        else:
            col = action.name

        observable_action = col in self.index_map

        if observable_action:
            entry[self.index_map[col]] = 1
            # append pre-observation entry
            self._append_result(entry)

        return observable_action

    def __update_and_converge_controllers(self, new_theta):
        self.world_def.set_controllers(new_theta)
        b = 0
        theta_err = sum([e ** 2 for e in self.world_def.pos_controller.error])
        vel_err = sum([e ** 2 for e in self.world_def.vel_controller.error])
        while theta_err > ENV_SETTINGS['PID_POS_CONV_TOL'] or vel_err > ENV_SETTINGS['PID_VEL_CONV_TOL']:

            if b > ENV_SETTINGS['PID_CONV_MAX_STEPS']:
                return False

            b += 1
            self.world_def.step(1.0 / BOX2D_SETTINGS['FPS'],
                                BOX2D_SETTINGS['VEL_ITERS'],
                                BOX2D_SETTINGS['POS_ITERS'])

            # this needs to render to update the arm on the screen
            if self.human_agent:
                self._render_world_at_frame_rate()

            # update error values
            theta_err = sum([e ** 2 for e in self.world_def.pos_controller.error])
            vel_err = sum([e ** 2 for e in self.world_def.vel_controller.error])
        return True

    def _render_world_at_frame_rate(self):
        '''
        render at desired frame rate
        '''
        if self.world_def.clock % RENDER_SETTINGS['RENDER_CLK_DIV'] == 0:
            self.render()

    def update_state_machine_at_frame_rate(self):
        ''''''
        if self.world_def.clock % BOX2D_SETTINGS['STATE_MACHINE_CLK_DIV'] == 0:
            self.update_state_machine()

    def update_state_machine(self, action=None):
        self.scenario.update_state_machine(action)

    def init_inverse_kine(self):
        # initialize inverse kinematics module with chain==target
        self.theta0 = BOX2D_SETTINGS['INITIAL_THETA_VECTOR']
        self.base0 = BOX2D_SETTINGS['INITIAL_BASE_CONFIG']
        initial_config = generate_five_arm(self.theta0[0], self.theta0[1], self.theta0[2], self.theta0[3], self.theta0[4])
        self.base = TwoDKinematicTransform(x=self.base0.x, y=self.base0.y, theta=self.base0.theta)
        self.invkine = InverseKinematics(KinematicChain(self.base, initial_config),
                                         KinematicChain(self.base, initial_config))

    def _print_observation(self, state, count):
        print(str(count) + ': ' + str(state['OBJ_STATES']))
        print(str(count) + ': ' + str(state['_FSM_STATE']))

    def _append_result(self, cur_result):
        self.results.append(cur_result)
        # if len(self.results) > 2:
        #     prev_result = self.results[-1]
        #     # remove frame
        #     differences = [x != y for (x, y) in zip(prev_result[1:], cur_result[1:])]
        #     changes = differences.count(True)
        #     if changes > 2:
        #         print 'WARNING: More than 2 changes between observations'
        #     self.results.append(cur_result)
        # else:
        #     self.results.append(cur_result)

    def _reset_results(self):
        # setup .csv headers
        self.col_label = []
        self.col_label.append('frame')
        for col_name in self.get_state()['OBJ_STATES']:
            self.col_label.append(col_name)
        self.col_label.append('agent')
        for col_name in self.action_space:
            self.col_label.append(col_name)

        self.index_map = {name : idx for idx, name in enumerate(self.col_label)}

        self.results = [self.col_label]

    def _create_clickable_regions(self):
        lock_regex = '^l[0-9]+'
        inactive_lock_regex = '^inactive[0-9]+$'
        # register clickable regions
        for b2_object_name, b2_object_data in list(self.world_def.obj_map.items()):
            if re.search(lock_regex, b2_object_name) or re.search(inactive_lock_regex, b2_object_name):
                lock = b2_object_data

                lock.create_clickable(self.step, self.action_map)
                self.viewer.register_clickable_region(lock.inner_clickable)
                self.viewer.register_clickable_region(lock.outer_clickable)
                # lock inactive levers
                if re.search(inactive_lock_regex, b2_object_name):
                    self.world_def.lock_lever(lock.name)
            elif b2_object_name == 'door_right_button':
                door_button = b2_object_data
                callback_action = 'push_door'
                door_button.create_clickable(self.step, self.action_map, self.action_map[callback_action])
                self.viewer.register_clickable_region(door_button.clickable)
            elif b2_object_name == 'door_left_button':
                door_button = b2_object_data
                callback_action = 'pull_door'
                door_button.create_clickable(self.step, self.action_map, self.action_map[callback_action])
                self.viewer.register_clickable_region(door_button.clickable)
            elif b2_object_name == 'reset_button':
                reset_button = b2_object_data
                callback_action = 'reset'
                reset_button.create_clickable(self.step, self.action_map,
                                              common.Action(callback_action, (reset_button, 4)))
                self.viewer.register_clickable_region(reset_button.clickable)
            elif b2_object_name == 'save_button':
                save_button = b2_object_data
                callback_action = 'save'
                save_button.create_clickable(self.step, self.action_map,
                                             common.Action(callback_action, (save_button, 4)))
                self.viewer.register_clickable_region(save_button.clickable)

    def get_state(self):
        if self.use_physics is True and self.world_def is None:
            raise ValueError('world_def is None while trying to call get_state()')
        # get state from physics simulator
        if self.use_physics:
            state = self.get_simulator_state()
        # get state from scenario/FSM
        else:
            state = self.get_fsm_state()
        return state

    def get_fsm_state(self):
        return self.scenario.get_state()

    def get_simulator_state(self):
        return self.world_def.get_state()
        
    def determine_door_seq(self):
        # we want the last action to always be push the door, the agent will be punished if the last action is not push the door.
        cur_action_seq = self.cur_action_seq
        if len(cur_action_seq)==3:
            door_act = ActionLog("push_door",None)
            if cur_action_seq[-1] == door_act:
                return 1
            else: return -1
        return 0
    # this function also determines if the action sequence is a duplicate to unlock the door, not just open the door
    def determine_unique_solution(self):
        if len(self.cur_action_seq) != len(self.solutions[0]):
            return False
        # if this is a complete action sequence and it is not a solution, return false
        # full action sequence
        # solution is unique if it is in the list of solutions and not in the solutions found
        if self.cur_action_seq in self.solutions and self.cur_action_seq not in self.completed_solutions:
            return True
        else:
            return False

    def determine_partial_solution(self):
        # order matters, so we need to compare element by element
        cur_action_seq = self.cur_action_seq
        solutions = self.solutions
        for solution in solutions:
            assert len(cur_action_seq) <= len(solution), 'Action sequence is somehow longer than solution'
            comparison = [solution[i] == cur_action_seq[i] for i in range(len(cur_action_seq))]
            if all(comparison):
                return True
        return False

    def determine_unique_partial_solution(self):
        for completed_solution in self.completed_solutions:
            if self.cur_action_seq == completed_solution[:len(self.cur_action_seq)]:
                return False
        # if the partial sequence is not in the completed solutions, just check if the partial sequence is
        # part of the solutions at all
        return self.determine_partial_solution()

    def determine_trial_success(self):
        return len(self.completed_solutions) > 0 and\
               len(self.solutions) > 0 and\
               len(self.completed_solutions) == len(self.solutions)

    def determine_fluent_change(self):
        prev_fluent_state = self.prev_state['OBJ_STATES']
        cur_fluent = self.state['OBJ_STATES']
        return prev_fluent_state != cur_fluent

    def determine_moveable_action(self, action):
        '''
        determines if the action is movable. Treats all active levers as movable, regardless of FSM
        If you need to detect if the action will cause an effect, negative the determine_fluent_change function
        :param action:
        :return:
        '''
        if self.use_physics:
            state, labels = self.obs_space.create_discrete_observation_from_simulator(self)
        else:
            state, labels = self.obs_space.create_discrete_observation_from_fsm(self)
        obj_name = action.obj
        if obj_name == 'door':
            # door being movable depends on door lock
            if state[labels.index('door_lock')] == 1:
                return False
            else:
                return True
        active = state[labels.index(obj_name + '_active')]
        if active:
            return True
        else:
            return False



    def determine_repeated_action(self):
        cur_action_seq = self.cur_action_seq
        if len(cur_action_seq) >= 2 and cur_action_seq[-2] == cur_action_seq[-1]:
            return True
        return False

    def _export_results(self):
        save_count = len(glob(self.save_path + 'results[0-9]*.csv'))
        np.savetxt(self.save_path + 'results{}.csv'.format(save_count), self.results, delimiter=',', fmt='%s')

    def _action_go_to(self, twod_config):
        # get configuatin of end effector
        targ_x, targ_y, targ_theta = twod_config

        # draw arrow to show target location
        args = (targ_x, targ_y, targ_theta, 0.5, 1, common.Color(0.8, 0.8, 0.8))
        self.viewer.markers['targ_arrow'] = ('arrow', args)

        # update current config
        self.invkine.kinematic_chain.update_chain(self.world_def.get_rel_config())

        # generate discretized waypoints
        waypoints = discretize_path(self.invkine.kinematic_chain.get_total_delta_config(),
                                    common.TwoDConfig(targ_x, targ_y, targ_theta),
                                    ENV_SETTINGS['PATH_INTERP_STEP_DELTA'])

        if len(waypoints) == 1:
            # already at the target config
            return True

        for i in range(1, len(waypoints)):  # waypoint 0 is current config

            # update kinematics model to reflect current world config
            self.invkine.kinematic_chain.update_chain(self.world_def.get_rel_config())

            # update inverse kinematics
            self.invkine.set_current_config(self.invkine.kinematic_chain)
            self.invkine.target = waypoints[i]

            # find inverse kinematics solution
            a = 0
            err = self.invkine.get_error()
            new_config = None
            while err > ENV_SETTINGS['INVK_CONV_TOL']:

                if a > ENV_SETTINGS['INVK_CONV_MAX_STEPS']:
                    return False
                a = a + 1

                # get delta theta
                d_theta = self.invkine.get_delta_theta_dls(lam=ENV_SETTINGS['INVK_DLS_LAMBDA'])

                # current theta along convergence path
                cur_config = self.invkine.kinematic_chain.get_rel_config()  # ignore virtual base link

                # new theta along convergence path

                # TODO: this is messy
                new_config = [cur_config[0]] + [common.TwoDConfig(cur.x, cur.y, cur.theta + delta) for cur, delta in
                                                zip(cur_config[1:], d_theta)]

                # update inverse kinematics model to reflect step along convergence path
                self.invkine.kinematic_chain.update_chain(new_config)

                err = self.invkine.get_error()

            # theta found, update controllers and wait until controllers converge and stop
            if new_config:
                if not self.__update_and_converge_controllers([c.theta for c in new_config[1:]]):
                    # could not converge
                    return False

        # succesfully reached target config

        # delete target arrow
        if 'targ_arrow' in list(self.viewer.markers.keys()):
            del self.viewer.markers['targ_arrow']

        return True

    def _action_go_to_obj(self, obj_name):
        """

        Args:
            object: reference to Box2D fixture that you want to go to

        Returns:

        """
        obj = self.world_def.obj_map[obj_name].fixture

        # find face facing us by raycasting from end effector to center of fixture
        end_eff = self.world_def.end_effector_fixture
        end_eff_shape = end_eff.shape
        end_eff_mass_data = end_eff.massData
        obj_mass_data = obj.massData

        obj_center = obj.body.GetWorldPoint(obj_mass_data.center)
        end_effector_center = end_eff.body.GetWorldPoint(end_eff_mass_data.center)

        input = b2RayCastInput(p1=end_effector_center, p2=obj_center, maxFraction=200)
        output = b2RayCastOutput()

        hit = obj.RayCast(output, input, 0)
        if hit:
            hit_point = input.p1 + output.fraction * (input.p2 - input.p1)
            normal = output.normal

            angle = np.arctan2(-normal[1], -normal[0])


            end_effector_offset = end_eff_shape.radius * normal # TODO: is this the right offset?

            desired_config = common.TwoDConfig(hit_point[0] + end_effector_offset[0],
                                        hit_point[1] + end_effector_offset[1],
                                        common.wrapToMinusPiToPi(angle))

            self._action_go_to(desired_config)

            # we way have gotten close to obj, but lets move forward until we graze
            # TODO: selective tolerance of INVK/PID controllers for rough/fine movement
            i = 0
            while len(self.world_def.arm_bodies[-1].contacts) == 0 and i < 5:
                i += 1
                self._action_move_end_frame(common.TwoDConfig(0.5, 0, 0))
            return True if len(self.world_def.arm_bodies[-1].contacts) > 0 else False
        else:
            # path is blocked
            return False

    def _action_rest(self):
        # discretize path
        cur_theta = [cur.theta for cur in self.world_def.get_rel_config()[1:]]

        # calculate number of discretized steps
        delta = [common.wrapToMinusPiToPi(t - c) for t, c in zip(self.theta0, cur_theta)]

        num_steps = max([int(abs(d / ENV_SETTINGS['PATH_INTERP_STEP_DELTA'])) for d in delta])

        if num_steps == 0:
            # we're already within step_delta of our desired config in all dimensions
            return True

        #TODO: refactor

        # generate discretized path
        waypoints = []
        for i in range(0, num_steps + 1):
            waypoints.append([common.wrapToMinusPiToPi(cur + i * d / num_steps) \
                              for cur, d in zip(cur_theta, delta)])

        # sanity check: we actually reach the target config

        # TODO: arbitrary double comparison
        assert all([abs(common.wrapToMinusPiToPi(waypoints[-1][i] - self.theta0[i]))  < 0.01 for i in range(0, len(self.theta0))])

        for waypoint in waypoints:
            if not self.__update_and_converge_controllers(waypoint):
                return False
        
        return True

    def _action_pull(self, action):
        name = action.obj
        distance = action.params

        if not self._action_go_to_obj(name):
            return False

        if not self._action_grasp():
            return False
        
        cur_x, cur_y, cur_theta = self.world_def.get_abs_config()[-1]
        neg_normal = (-np.cos(cur_theta), -np.sin(cur_theta))
        new_config = common.TwoDConfig(cur_x + neg_normal[0] * distance,
                                cur_y + neg_normal[1] * distance,
                                cur_theta)

        if not self._action_go_to(new_config):
            self._action_grasp() # remove connection
            return False

        if not self._action_grasp():
            return False

        return True

    def _action_push(self, action):
        name = action.obj
        distance = action.params

        if not self._action_go_to_obj(name):
            return False

        if not self._action_move_end_frame(common.TwoDConfig(distance, 0, 0)):
            return False

        return True

    def _action_grasp(self, targ_fixture=None):
        # TODO: you can do better than this lol
        for i in range(0, 100):
            if self._action_grasp_attempt(targ_fixture):
                return True
        return False

    def _action_grasp_attempt(self, targ_fixture=None):
        # NOTE: It's a little tricky to grab objects when you're EXACTLY
        # touching, instead, we compute the shortest distance between the two
        # shapes once the bounding boxes start to overlap. This let's us grab
        # objects which are close. See: http://www.iforce2d.net/b2dtut/collision-anatomy

        if len(self.world_def.grasped_list) > 0:
            # we are already holding something
            for connection in self.world_def.grasped_list:
                if targ_fixture and not (connection.bodyA == targ_fixture.body or \
                                        connection.bodyB == targ_fixture.body):
                    continue
                else:
                    self.world_def.world.DestroyJoint(connection)
            self.world_def.grasped_list = []
            return True
        else:
            if len(self.world_def.arm_bodies[-1].contacts) > 0:
                # grab all the things!
                for contact_edge in self.world_def.arm_bodies[-1].contacts:
                    fix_A = contact_edge.contact.fixtureA
                    fix_B = contact_edge.contact.fixtureB

                    if targ_fixture and not (fix_A == targ_fixture or fix_B == targ_fixture):
                        continue
                    else:
                        # indiscriminate grab or found target

                        # find shortest distance between two shapes
                        dist_result = b2Distance(shapeA=fix_A.shape,
                                                 shapeB=fix_B.shape,
                                                 transformA=fix_A.body.transform,
                                                 transformB=fix_B.body.transform)

                        point_A = fix_A.body.GetLocalPoint(dist_result.pointA)
                        point_B = fix_B.body.GetLocalPoint(dist_result.pointB)

                        # TODO experiment with other joints
                        self.world_def.grasped_list.append(self.world_def.world.CreateDistanceJoint(bodyA=fix_A.body,
                                                                                bodyB=fix_B.body,
                                                                                localAnchorA=point_A,
                                                                                localAnchorB=point_B,
                                                                                frequencyHz=1,
                                                                                dampingRatio=1,
                                                                                collideConnected=True
                                                                                ))
                return True
            else:
                return False

    def _action_move(self, action):
        delta_x, delta_y, delta_theta = action.params

        cur_x, cur_y, cur_theta = self.world_def.get_abs_config()[-1]

        return self._action_go_to(common.TwoDConfig(cur_x + delta_x,
                                                    cur_y + delta_y,
                                                    cur_theta + delta_theta))

    def _action_move_end_frame(self, params):
        delta_x, delta_y, delta_theta = params

        cur_x, cur_y, cur_theta = self.world_def.get_abs_config()[-1]

        x_axis = (np.cos(cur_theta), np.sin(cur_theta))
        y_axis = (-x_axis[1], x_axis[0])

        new_config = common.TwoDConfig(cur_x + x_axis[0] * delta_x + y_axis[0] * delta_y,
                                       cur_y + x_axis[1] * delta_x + y_axis[1] * delta_y,
                                       cur_theta + delta_theta)

        return self._action_go_to(new_config)

    # def _action_unlock(self, params):
    #     name = params
    #
    #     lock, joint, _ = self.world_def.obj_map[name]
    #     self._action_push_perp((lock, abs(joint.lowerLimit)))
    #
    # def _action_lock(self, params):
    #     name = params
    #
    #     lock, joint, _ = self.world_def.obj_map[name]
    #     self._action_pull_perp((lock, abs(joint.lowerLimit)))

    def _action_reset(self):
        self.reset()
        return True

    def _action_save(self):
        self._export_results()
        self.reset()
        return True

    def _action_nothing(self):
        return True

    def get_avail_actions(self):
        return self.world_def.scenario.actions




def main():
    env = OpenLockEnv()


if __name__ == '__main__':
    main()
