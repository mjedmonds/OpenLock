import gym
import re
import copy
import time
import numpy as np
from Box2D import b2RayCastInput, b2RayCastOutput, b2Distance
from gym.spaces import MultiDiscrete

from openlock.box2d_renderer import Box2DRenderer
import openlock.common as common
from openlock.envs.world_defs.openlock_def import ArmLockDef
from openlock.kine import (
    KinematicChain,
    discretize_path,
    InverseKinematics,
    generate_five_arm,
    TwoDKinematicTransform,
)
from openlock.rewards import (
    RewardStrategy,
)  # determine_reward, REWARD_IMMOVABLE, REWARD_OPEN
from openlock.settings_trial import select_trial, get_trial
from openlock.settings_scenario import select_scenario
from openlock.settings_render import RENDER_SETTINGS, BOX2D_SETTINGS, ENV_SETTINGS
from openlock.logger_env import ActionLog, TrialLog


from glob import glob


# TODO: add ability to move base
# TODO: more physically plausible units?


class ActionSpace:
    def __init__(self):
        pass

    @staticmethod
    def create_action_space(env, obj_map):
        # this must be preallocated; they are filled by position, not by symbol
        # todo: minus 2 is for door and door_lock; magic number, programmatically remove this
        num_levers = len(obj_map.keys()) - 2
        push_action_space = [None] * num_levers
        pull_action_space = [None] * num_levers
        door_action_space = []
        action_map = dict()
        action_map_external_role = dict()
        action_map_role_external = dict()
        for obj, val in list(obj_map.items()):
            if "button" not in obj and "door" not in obj:
                if env.lever_index_mode == "position":
                    name = val.position.name
                else:
                    name = obj
                role = obj
                # use position to map to an integer index
                twod_config = val.position.config
                lever_idx = env.config_to_idx[twod_config]

                # todo: refactor this, three mappings is complicated
                name_push = "push_{}".format(name)
                name_pull = "pull_{}".format(name)
                role_push = "push_{}".format(role)
                role_pull = "pull_{}".format(role)

                push_action_space[lever_idx] = name_push
                pull_action_space[lever_idx] = name_pull

                role_push_action = common.Action("push", role, 4)
                role_pull_action = common.Action("pull", role, 4)

                name_push_action = common.Action("push", name, 4)
                name_pull_action = common.Action("pull", name, 4)

                action_map[name_push] = name_push_action
                action_map[name_pull] = name_pull_action

                # role based mapping from external names to internal
                action_map_external_role[name_push] = role_push_action
                action_map_external_role[name_pull] = role_pull_action

                action_map_role_external[role_push] = name_push_action
                action_map_role_external[role_pull] = name_pull_action

            if "button" not in obj and "door" in obj and "door_lock" != obj:
                name_push = "push_{}".format(obj)
                name_action = common.Action("push", obj, 4)

                door_action_space.append(name_push)

                action_map[name_push] = name_action
                action_map_external_role[name_push] = name_action
                action_map_role_external[name_push] = name_action

        action_space = push_action_space + pull_action_space + door_action_space

        return (
            action_space,
            action_map,
            action_map_external_role,
            action_map_role_external,
        )


class ObservationSpace:
    def __init__(self, num_levers, append_solutions_remaining=False):
        self.append_solutions_remaining = append_solutions_remaining
        self.solutions_found = [0, 0, 0]
        self.labels = ["sln0", "sln1", "sln2"]

        if self.append_solutions_remaining:
            self.multi_discrete = self.create_observation_space(
                num_levers, len(self.solutions_found)
            )
        else:
            self.multi_discrete = self.create_observation_space(num_levers)
        self.num_levers = num_levers
        self.state = None
        self.state_labels = None
        self.external_to_role_mapping = None
        self.role_to_external_mapping = None

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
        discrete_space.append(num_door_lock_states)  # door lock
        discrete_space.append(num_door_states)  # door open
        # solutions appended
        for i in range(num_solutions):
            # solutions can only be found or not found
            discrete_space.append(2)
        discrete_space = np.array(discrete_space)
        multi_discrete = MultiDiscrete(discrete_space)
        return multi_discrete

    def create_internal_state_external_state_mappings(self, env):
        # todo: refactor this into a more coherent state/action conversion
        external_to_internal_action_map = env.action_map_external_role
        internal_state_external_state_mapping = dict()
        external_state_internal_state_mapping = dict()
        for (
            external_action_name,
            internal_action,
        ) in external_to_internal_action_map.items():
            external_state_name = external_action_name.split("_", 1)[1]
            internal_state_name = internal_action.obj
            internal_state_external_state_mapping[
                internal_state_name
            ] = external_state_name
            external_state_internal_state_mapping[
                external_state_name
            ] = internal_state_name
        return (
            internal_state_external_state_mapping,
            external_state_internal_state_mapping,
        )

    def create_discrete_observation(self, env):
        # create mapping from internal simulator state to external state
        if self.role_to_external_mapping or self.external_to_role_mapping is None:
            self.role_to_external_mapping, self.external_to_role_mapping = self.create_internal_state_external_state_mappings(
                env
            )
        if env.use_physics:
            discrete_state, discrete_labels = self.create_discrete_observation_from_simulator(
                env
            )
        else:
            discrete_state, discrete_labels = self.create_discrete_observation_from_fsm(
                env
            )
        # convert internal state labels to external labels
        # todo: refactor this, this is a very brittle way of doing this mapping
        for i in range(len(discrete_labels)):
            if discrete_labels[i] in self.role_to_external_mapping.keys():
                discrete_labels[i] = self.role_to_external_mapping[discrete_labels[i]]
            if discrete_labels[i].endswith("_active"):
                base_label = discrete_labels[i].split("_", 1)[0]
                if base_label in self.role_to_external_mapping.keys():
                    base_label = self.role_to_external_mapping[base_label]
                discrete_labels[i] = base_label + "_active"
        return discrete_state, discrete_labels

    def create_discrete_observation_from_simulator(self, env):
        """
        Constructs a discrete observation from the physics simulator
        :param world_def:
        :return:
        """
        levers = env.world_def.get_levers()
        self.num_levers = len(levers)
        world_state = env.world_def.get_state()

        # need one element for state and color of each lock, need two addition for door lock status and door status
        state = [None] * (self.num_levers * 2 + 2)
        state_labels = [None] * (self.num_levers * 2 + 2)

        for lever in levers:
            # convert to index based on lever position
            lever_idx = env.config_to_idx[lever.position.config]

            lever_state = np.int8(world_state["OBJ_STATES"][lever.name])
            lever_active = np.int8(lever.determine_active())

            state_labels[lever_idx] = lever.name
            state[lever_idx] = lever_state
            state_labels[lever_idx + self.num_levers] = lever.name + "_active"
            state[lever_idx + self.num_levers] = lever_active

        state_labels[-1] = "door"
        state[-1] = np.int8(world_state["OBJ_STATES"]["door"])
        state_labels[-2] = "door_lock"
        state[-2] = np.int8(world_state["OBJ_STATES"]["door_lock"])

        if self.append_solutions_remaining:
            slns_found, sln_labels = self.determine_solutions_remaining(env)
            state.extend(slns_found)
            state_labels.extend(sln_labels)

        return state, state_labels

    def create_discrete_observation_from_fsm(self, env):
        """
        constructs a discrete observation from the underlying FSM
        Used when the physics simulator is being bypassed
        :param fsmm:
        :return:
        """
        levers = env.scenario.levers
        self.num_levers = len(levers)
        scenario_state = env.scenario.get_state()

        # need one element for state and color of each lock, need two addition for door lock status and door status
        state = [None] * (self.num_levers * 2 + 2)
        state_labels = [None] * (self.num_levers * 2 + 2)

        # lever states
        for lever in levers:
            lever_idx = env.config_to_idx[lever.position.config]

            # inactive lever, state is constant
            if re.search(common.INACTIVE_LOCK_REGEX_STR, lever.name):
                lever_active = np.int8(common.ENTITY_STATES["LEVER_INACTIVE"])
            else:
                lever_active = np.int8(common.ENTITY_STATES["LEVER_ACTIVE"])

            lever_state = np.int8(scenario_state["OBJ_STATES"][lever.name])

            state_labels[lever_idx] = lever.name
            state[lever_idx] = lever_state

            state_labels[lever_idx + self.num_levers] = lever.name + "_active"
            state[lever_idx + self.num_levers] = lever_active

        # update door state
        door_lock_name = "door_lock"
        door_lock_state = np.int8(scenario_state["OBJ_STATES"][door_lock_name])

        # todo: this is a hack to get whether or not the door is actually open; it should be part of the FSM
        door_name = "door"
        door_state = np.int8(scenario_state["OBJ_STATES"][door_name])

        state_labels[-1] = door_name
        state[-1] = door_state
        state_labels[-2] = door_lock_name
        state[-2] = door_lock_state

        if self.append_solutions_remaining:
            slns_found, sln_labels = self.determine_solutions_remaining(env)
            state.extend(slns_found)
            state_labels.extend(sln_labels)

        return state, state_labels

    def determine_solutions_remaining(self, cur_trial):
        # todo: this does not work currently
        raise RuntimeError("determine_solutions_remaining() is currently broken")
        solutions = cur_trial.solutions
        completed_solutions = cur_trial.completed_solutions
        for i in range(len(completed_solutions)):
            idx = solutions.index(completed_solutions[i])
            # mark that this solution is finished
            self.solutions_found[idx] = 1

        return self.solutions_found, self.labels


class OpenLockEnv(gym.Env):
    # Set this in SOME subclasses
    metadata = {"render.modes": ["human"]}  # TODO what does this do?

    def __init__(self):
        self.viewer = None

        # handle to the scenario, defined by the scenario
        self.scenario = None

        self.i = 0
        self.clock = 0
        self.save_path = "../OpenLockResults/"

        self.col_label = []
        self.index_map = None
        self.results = None

        self.attempt_count = 0  # keeps track of the number of attempts
        self.action_count = 0  # keeps track of the number of actions executed
        self.action_limit = None
        self.attempt_limit = None

        self.full_attempt_limit = False

        self.action_executing = False  # used to disable action preemption

        self.human_agent = True
        self.reward_mode = "basic"

        self.lever_index_mode = (
            "role"
        )  # controls whether or not to build action_map based on lever role or position
        self.observation_space = None
        self.action_space = None
        self.action_map = None
        # internal action map to go from external to internal latent action
        self.action_map_external_role = None
        # external action map to go from internal to external action
        self.action_map_role_external = None

        self.reward_strategy = RewardStrategy()
        self.reward_range = (
            self.reward_strategy.REWARD_IMMOVABLE,
            self.reward_strategy.REWARD_OPEN,
        )

        self.use_physics = True

        self.world_def = None

        self.states = []
        self.config_to_idx = dict()
        self.position_to_idx = dict()
        self.idx_to_position = dict()
        self.attribute_order = []
        self.attribute_labels = dict()
        self.attribute_function_map = {
            "position": self.get_obj_position_name,
            "color": self.get_obj_color,
        }
        # current trial to keep track of progress through this trial
        self.cur_trial = None
        # keeps track of current state. todo: can this safely be removed
        self.cur_state = None
        self.prev_state = None
        # keeps track of which trials have been completed this execution
        self.completed_trials = []

    def initialize_for_scenario(self, scenario_name):
        self._set_scenario(scenario_name)
        trial_scenario_name = scenario_name

        _, lever_configs = get_trial(scenario_name)

        self._set_lever_configs(lever_configs)
        self.config_to_idx = {
            lever_configs[i].LeverPosition.config: i for i in range(len(lever_configs))
        }
        self.position_to_idx = {
            lever_configs[i].LeverPosition.name: i for i in range(len(lever_configs))
        }
        self.idx_to_position = {
            i: lever_configs[i].LeverPosition.name for i in range(len(lever_configs))
        }
        # todo: elegantly include door; at this stage of initialization we don't have access to obj_map
        door_idx = len(self.config_to_idx.keys())
        self.config_to_idx[common.ObjectPositionEnum.DOOR.config] = door_idx
        self.position_to_idx["door"] = door_idx
        self.idx_to_position[door_idx] = "door"

        self.states = list(self.position_to_idx.keys())
        self.attribute_order = ["position", "color"]
        self.attribute_labels = {
            "color": common.COLOR_LABELS,
            "position": list(self.position_to_idx.keys()),
        }

    def reset(self):
        """Resets the state of the environment and returns an initial observation.

        Returns: observation (object): the initial observation of the
                space.
        """
        if self.scenario is None:
            print("WARNING: resetting environment with no scenario")

        self.clock = 0
        self._seed()

        if self.use_physics:
            self.init_inverse_kine()

            # setup Box2D world
            self.world_def = ArmLockDef(
                self.invkine.kinematic_chain,
                1.0 / BOX2D_SETTINGS["FPS"],
                30,
                self.scenario,
            )

            obj_map = self.world_def.obj_map
            levers = self.world_def.get_levers()
        else:
            # initialize obj_map for scenario
            self.scenario.init_scenario_env()

            obj_map = self.scenario.obj_map

            levers = self.scenario.levers

        self.action_space, self.action_map, self.action_map_external_role, self.action_map_role_external = ActionSpace.create_action_space(
            self, obj_map
        )
        self.observation_space = ObservationSpace(len(levers))

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
        self.cur_trial.add_attempt()

        if self.use_physics:
            if self.human_agent:
                self.render()

        self.cur_state = self.get_state()
        # append initial observation
        # self._print_observation(state, self.action_count)
        self._append_result(self._create_state_entry())

        self.update_state_machine()

        if self.observation_space is not None:
            discrete_state, discrete_labels = self.observation_space.create_discrete_observation(
                self
            )
            return np.array(discrete_state)
        else:
            raise ValueError(
                "Attempting to reset environment with no observation space. Cannot return state."
            )

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
                info (dict): CONVERGED : whether algorithm succesfully coverged on action
        """
        # save a copy of the current state
        self.prev_state = self.get_state()

        # rendering step
        if not action:
            self.world_def.step(
                1.0 / BOX2D_SETTINGS["FPS"],
                BOX2D_SETTINGS["VEL_ITERS"],
                BOX2D_SETTINGS["POS_ITERS"],
            )

            # self._render_world_at_frame_rate()

            self.cur_state = self.get_state()
            self.cur_state["SUCCESS"] = False
            self.update_state_machine()
            # no action, return nothing to indicate no reward possible
            return None
        # change to simple "else:" to enable action preemption
        elif self.action_executing is False:
            self.action_executing = True
            self.i += 1
            reset = False
            action_success = False
            attempt_success = False
            trial_success = False
            reward = None
            done = False
            observable_action = self._create_pre_obs_entry(action)
            if observable_action:
                # ack is used by manager to determine if the action needs to be logged in the agent's logger
                self.cur_trial.cur_attempt.add_action(str(action))

            # convert external action to internal action
            if str(action) in self.action_map_external_role.keys():
                action_role = self.action_map_external_role[str(action)]
            else:
                action_role = action
            # execute action
            if self.use_physics:
                action_success = self._execute_physics_action(action_role)
            else:
                action_success = True
                self.scenario.execute_fsm_action(action_role)

            self.i += 1

            self.cur_state = self.get_state()
            self.cur_state["SUCCESS"] = action_success

            if observable_action:
                reward = self.finish_action(action)

                # if 10 < self.cur_trial.cur_attempt.reward < 50:
                #     print('reward is 20...')
                #     reward, _ = self.reward_strategy.determine_reward(self, action, self.reward_mode)

            if self.determine_attempt_finished():
                done = True
                attempt_success = self.determine_unique_solution()

            discrete_state, discrete_labels = self._create_discrete_state()

            self.action_executing = False

            return (
                np.array(discrete_state),
                reward,
                done,
                {
                    "action_success": action_success,
                    "attempt_success": attempt_success,
                    "results": self.results,
                    "state_labels": discrete_labels,
                },
            )
        else:
            self.cur_state = self.get_state()
            self.update_state_machine()
            return None
            # return self.state, 0, False, {}

    def render(self, mode="human", close=False):
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
            self.viewer.render_multiple_worlds(
                [self.world_def.background, self.world_def.world], mode="human"
            )

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

    # code to run before human and computer trials
    def setup_trial(
        self,
        scenario_name,
        action_limit,
        attempt_limit,
        specified_trial=None,
        multiproc=False,
    ):
        """
        Set the env class variables and select a trial (specified if provided, otherwise a random trial from the scenario name).

        This method should be called before running human and computer trials.
        Returns the trial selected (string).

        :param scenario_name: name of scenario (e.g. those defined in settings_trial.PARAMS)
        :param action_limit: number of actions permitted
        :param attempt_limit: number of attempts permitted
        :param specified_trial: optional specified trial. If none, get_trial is used to select trial
        :param multiproc: disables printing if running in multiple threads
        :return state: state of env after reset
        :return trial_selected: the selected_trial as returned by get_trial or select_trial
        """
        self._set_scenario(scenario_name)
        # set limits
        self.attempt_count = 0
        self.attempt_limit = attempt_limit
        self.action_limit = action_limit
        # select trial
        if specified_trial is None:
            trial_selected, lever_configs = get_trial(
                scenario_name, self.completed_trials
            )
            if trial_selected is None:
                if not multiproc:
                    print(
                        "WARNING: no more trials available. Resetting completed_trials."
                    )
                    print(self.completed_trials)
                self.completed_trials = []
                trial_selected, lever_configs = get_trial(
                    scenario_name, self.completed_trials
                )
        else:
            trial_selected, lever_configs = select_trial(specified_trial)

        self._set_lever_configs(lever_configs)
        self.observation_space = ObservationSpace(len(self.scenario.levers))

        self.scenario.init_scenario_env()
        obj_map = self.scenario.obj_map
        action_space, action_map, action_map_external_role, action_map_role_external = ActionSpace.create_action_space(
            self, obj_map
        )

        external_solutions = [
            [
                action_map_role_external[str(solution_action)]
                for solution_action in solution
            ]
            for solution in self.scenario.solutions
        ]

        self.cur_trial = TrialLog(
            trial_selected, scenario_name, external_solutions, time.time()
        )

        if not multiproc:
            print(
                "INFO: New trial {}. There are {} unique solutions remaining.".format(
                    trial_selected, len(self.scenario.solutions)
                )
            )

        return trial_selected

    def finish_trial(self, trial_selected):
        self.completed_trials.append(trial_selected)

    def finish_attempt(self):
        self.attempt_count += 1

        # stores whether or not this attempt executed a unique solution
        action_seq = self.get_current_action_seq(convert_to_action=True)
        attempt_success = self.cur_trial.finish_attempt(
            self.results, action_seq
        )

        pause = self.update_user(attempt_success)

        # pauses if the human user unlocked the door but didn't push on the door
        if self.use_physics and self.human_agent and pause:
            # pause for 4 sec to allow user to view lock
            t_end = time.time() + 4
            while time.time() < t_end:
                self.render()
            self.update_state_machine()

        self.cur_trial.add_attempt()

    def finish_action(self, action):
        self.action_count += 1

        # self._print_observation(self.state, self.action_count)
        self._append_result(self._create_state_entry())
        # self.results.append(self._create_state_entry(self.state, self.action_count))

        # must finish action before computing reward
        self.cur_trial.cur_attempt.finish_action(self.results)

        reward, _ = self.reward_strategy.determine_reward(
            self, action, self.reward_mode
        )

        # add reward to current attempt
        self.cur_trial.cur_attempt.add_reward(reward)

        return reward

    def _set_scenario(self, scenario_name):
        # update scenario if needed
        if self.scenario is None or scenario_name is not self.scenario.name:
            scenario = select_scenario(scenario_name, use_physics=self.use_physics)
            self.scenario = scenario

    def _set_lever_configs(self, lever_configs):
        self.scenario.set_lever_configs(lever_configs)

    def _reset_results(self):
        # setup .csv headers
        self.col_label = []
        self.col_label.append("frame")
        discrete_states, discrete_labels = self._create_discrete_state()
        for col_name in discrete_labels:
            self.col_label.append(col_name)
        self.col_label.append("agent")
        for col_name in self.action_space:
            self.col_label.append(col_name)

        self.index_map = {name: idx for idx, name in enumerate(self.col_label)}

        self.results = [self.col_label]

    def _create_discrete_state(self):
        return self.observation_space.create_discrete_observation(self)

    def _create_state_entry(self):
        frame = self.action_count
        discrete_state, discrete_labels = self._create_discrete_state()
        entry = [0] * len(self.col_label)
        entry[0] = frame
        for name, val in zip(discrete_labels, discrete_state):
            entry[self.index_map[name]] = int(val)

        return entry

    def _create_pre_obs_entry(self, action):
        # create pre-observation entry
        entry = [0] * len(self.col_label)
        entry[0] = self.action_count
        # copy over previous state
        entry[1 : self.index_map["agent"] + 1] = copy.copy(
            self.results[-1][1 : self.index_map["agent"] + 1]
        )

        # mark action idx
        if type(action.obj) is str:
            col = "{}_{}".format(action.name, action.obj)
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
        while (
            theta_err > ENV_SETTINGS["PID_POS_CONV_TOL"]
            or vel_err > ENV_SETTINGS["PID_VEL_CONV_TOL"]
        ):

            if b > ENV_SETTINGS["PID_CONV_MAX_STEPS"]:
                return False

            b += 1
            self.world_def.step(
                1.0 / BOX2D_SETTINGS["FPS"],
                BOX2D_SETTINGS["VEL_ITERS"],
                BOX2D_SETTINGS["POS_ITERS"],
            )

            # this needs to render to update the arm on the screen
            if self.human_agent:
                self._render_world_at_frame_rate()

            # update error values
            theta_err = sum([e ** 2 for e in self.world_def.pos_controller.error])
            vel_err = sum([e ** 2 for e in self.world_def.vel_controller.error])
        return True

    def _render_world_at_frame_rate(self):
        """
        render at desired frame rate
        """
        if self.world_def.clock % RENDER_SETTINGS["RENDER_CLK_DIV"] == 0:
            self.render()

    def update_state_machine_at_frame_rate(self):
        """"""
        if self.world_def.clock % BOX2D_SETTINGS["STATE_MACHINE_CLK_DIV"] == 0:
            self.update_state_machine()

    def update_state_machine(self, action=None):
        self.scenario.update_state_machine(action)

    def init_inverse_kine(self):
        # initialize inverse kinematics module with chain==target
        self.theta0 = BOX2D_SETTINGS["INITIAL_THETA_VECTOR"]
        self.base0 = BOX2D_SETTINGS["INITIAL_BASE_CONFIG"]
        initial_config = generate_five_arm(
            self.theta0[0],
            self.theta0[1],
            self.theta0[2],
            self.theta0[3],
            self.theta0[4],
        )
        self.base = TwoDKinematicTransform(
            x=self.base0.x, y=self.base0.y, theta=self.base0.theta
        )
        self.invkine = InverseKinematics(
            KinematicChain(self.base, initial_config),
            KinematicChain(self.base, initial_config),
        )

    def _print_observation(self, state, count):
        print(str(count) + ": " + str(state["OBJ_STATES"]))
        print(str(count) + ": " + str(state["_FSM_STATE"]))

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

    def _create_clickable_regions(self):
        # register clickable regions
        for b2_object_name, b2_object_data in list(self.world_def.obj_map.items()):
            if re.search(common.LOCK_REGEX_STR, b2_object_name) or re.search(
                common.INACTIVE_LOCK_REGEX_STR, b2_object_name
            ):
                lock = b2_object_data

                lock.create_clickable(self.step)
                self.viewer.register_clickable_region(lock.inner_clickable)
                self.viewer.register_clickable_region(lock.outer_clickable)
                # lock inactive levers
                if re.search(common.INACTIVE_LOCK_REGEX_STR, b2_object_name):
                    self.world_def.lock_lever(lock.name)
            elif b2_object_name == "door_right_button":
                door_button = b2_object_data
                door_button.create_clickable(
                    self.step, callback_action=common.Action("push", "door", 4)
                )
                self.viewer.register_clickable_region(door_button.clickable)
            elif b2_object_name == "door_left_button":
                door_button = b2_object_data
                door_button.create_clickable(
                    self.step, callback_action=common.Action("pull", "door", 4)
                )
                self.viewer.register_clickable_region(door_button.clickable)
            elif b2_object_name == "reset_button":
                reset_button = b2_object_data
                callback_action = "reset"
                reset_button.create_clickable(
                    self.step,
                    callback_action=common.Action(callback_action, reset_button, 4),
                )
                self.viewer.register_clickable_region(reset_button.clickable)
            elif b2_object_name == "save_button":
                save_button = b2_object_data
                callback_action = "save"
                save_button.create_clickable(
                    self.step,
                    callback_action=common.Action(callback_action, save_button, 4),
                )
                self.viewer.register_clickable_region(save_button.clickable)

    def update_user(self, attempt_success, multithreaded=False):
        """
        Print update to the user.
        Either all solutions have been found, there are solutions remaining, or the user has
        reached the attempt limit and the trial is over without finding all solutions.

        :param attempt_success:
        :param multithreaded:
        :return: two booleans, the first representing whether the all solutions have been found (trial is finished), the second representing whether the simulator should pause (for when the user opened the door).
        """
        pause = False
        completed_solutions = self.get_completed_solutions()
        num_solutions_remaining = self.get_num_solutions_remaining()
        # continue or end trial
        if self.get_trial_success():
            if not multithreaded:
                print("INFO: You found all of the solutions. ")
            pause = True  # pause if they open the door
        elif self.attempt_count < self.attempt_limit:
            # alert user to the number of solutions remaining
            if attempt_success is True:
                if not multithreaded:
                    print(
                        "INFO: You found a solution. There are {} unique solutions remaining.".format(
                            num_solutions_remaining
                        )
                    )
                pause = True  # pause if they open the door
            else:
                if not multithreaded and self.human_agent:
                    print(
                        "INFO: Ending attempt. Action limit reached. There are {} unique solutions remaining. You have {} attempts remaining.".format(
                            num_solutions_remaining,
                            self.attempt_limit - self.attempt_count,
                        )
                    )
                # pause if the door lock is missing and the agent is a human
                if (
                    self.human_agent
                    and self.get_state()["OBJ_STATES"]["door_lock"]
                    == common.ENTITY_STATES["DOOR_UNLOCKED"]
                ):
                    pause = True
        else:
            if not multithreaded:
                print(
                    "INFO: Ending trial. Attempt limit reached. You found {} unique solutions".format(
                        len(completed_solutions)
                    )
                )

        return pause

    def get_state(self):
        if self.use_physics is True and self.world_def is None:
            raise ValueError("world_def is None while trying to call get_state()")
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

    def get_num_solutions_remaining(self):
        return len(self.get_solutions()) - len(self.get_completed_solutions())

    def get_internal_variable_name(self, obj_name):
        # need to convert to internal object name
        if obj_name in self.observation_space.external_to_role_mapping.keys():
            obj_name = self.observation_space.external_to_role_mapping[obj_name]
        return obj_name

    def get_internal_action_name(self, action_str):
        action_name, obj_name = action_str.split("_", 1)
        obj_name = self.get_internal_variable_name(obj_name)
        return action_name + "_" + obj_name

    def get_obj_color(self, obj_name):
        obj_name = self.get_internal_variable_name(obj_name)
        # todo: this is hacky, refactor, but doors and door_locks have no color attribute
        if obj_name == "door_lock" or obj_name == "door":
            return "GREY"
        if self.use_physics:
            obj = self.world_def.obj_map[obj_name]
        else:
            obj = self.scenario.obj_map[obj_name]
        color = common.COLOR_TO_COLOR_NAME[obj.color]
        return color

    def get_obj_position_name(self, obj_name):
        obj_name = self.get_internal_variable_name(obj_name)
        # todo: refactor so there is a single obj_map in env that is set depending upon use_physics
        if self.use_physics:
            obj = self.world_def.obj_map[obj_name]
        else:
            obj = self.scenario.obj_map[obj_name]
        return obj.position.name

    def get_obj_attributes(self, obj_name):
        """
        returns dict of attribute values for obj_name in the simulator.
        :param obj_name:
        :return:
        """
        obj_name = self.get_internal_variable_name(obj_name)
        obj_attributes = dict()
        for attribute_name in self.attribute_order:
            obj_attributes[attribute_name] = self.attribute_function_map[
                attribute_name
            ](obj_name)
        return obj_attributes

    def get_trial_success(self):
        return self.cur_trial.success

    def get_current_action_seq(self, convert_to_str=False, get_internal_action_seq=False, convert_to_action=False):
        cur_action_sequence = self.cur_trial.cur_attempt.action_seq
        if get_internal_action_seq and self.lever_index_mode != "role":
            cur_action_sequence = [
                ActionLog(self.get_internal_action_name(x.name), x.start_time)
                for x in cur_action_sequence
            ]
        if convert_to_str:
            cur_action_sequence = [str(x) for x in cur_action_sequence]
        if convert_to_action:
            cur_action_sequence = [common.Action(a.name.split("_")[0], a.name.split("_")[1], None) for a in cur_action_sequence]
        return cur_action_sequence

    def get_completed_solutions(self, convert_to_str=False):
        completed_solutions = self.cur_trial.completed_solutions
        if convert_to_str:
            completed_solutions = [str(x) for x in completed_solutions]
        return completed_solutions

    def get_solutions(self, convert_to_str=False):
        solutions =self.cur_trial.solutions
        if convert_to_str:
            solutions = [str(x) for x in solutions]
        return solutions

    def get_num_solutions(self):
        return len(self.cur_trial.solutions)

    def determine_attempt_finished(self):
        if self.action_count >= self.action_limit:
            return True
        else:
            return False

    def determine_door_seq(self):
        # we want the last action to always be push the door, the agent will be punished if the last action is not push the door.
        cur_action_seq = self.get_current_action_seq(convert_to_str=True)
        if len(cur_action_seq) == 3:
            door_act = ActionLog("push_door", None)
            if cur_action_seq[-1] == door_act:
                return 1
            else:
                return -1
        return 0

    # this function also determines if the action sequence is a duplicate to unlock the door, not just open the door
    def determine_unique_solution(self):
        cur_action_seq = self.get_current_action_seq(convert_to_str=True)
        solutions = self.get_solutions(convert_to_str=True)
        # todo: need more robust way - assumes solutions are all the same length
        if len(cur_action_seq) != len(solutions[0]):
            return False

        completed_solutions = self.get_completed_solutions(convert_to_str=True)
        # if this is a complete action sequence and it is not a solution, return false
        # full action sequence
        # solution is unique if it is in the list of solutions and not in the solutions found
        if cur_action_seq in solutions and cur_action_seq not in completed_solutions:
            return True
        else:
            return False

    def determine_partial_solution(self):
        """
        Determines if the current action sequence is part of a solution
        :return: True if the current action sequence is part of a solution, False otherwise
        """
        cur_action_seq = self.get_current_action_seq(convert_to_str=True)
        if cur_action_seq in [x[: len(cur_action_seq)] for x in self.get_solutions(convert_to_str=True)]:
            return True
        else:
            return False

    def determine_unique_partial_solution(self):
        cur_action_seq = self.get_current_action_seq(convert_to_str=True)
        completed_solutions = self.get_completed_solutions(convert_to_str=True)
        for completed_solution in completed_solutions:
            if cur_action_seq == completed_solution[: len(cur_action_seq)]:
                return False
        # if the partial sequence is not in the completed solutions, just check if the partial sequence is
        # part of the solutions at all
        return self.determine_partial_solution()

    def determine_fluent_change(self):
        prev_fluent_state = self.prev_state["OBJ_STATES"]
        cur_fluent = self.cur_state["OBJ_STATES"]
        return prev_fluent_state != cur_fluent

    def determine_repeated_action(self):
        cur_action_seq = self.get_current_action_seq(convert_to_str=True)
        if len(cur_action_seq) >= 2 and cur_action_seq[-2] == cur_action_seq[-1]:
            return True
        return False

    def determine_moveable_action(self, action):
        """
        determines if the action is movable. Treats all active levers as movable, regardless of FSM
        If you need to detect if the action will cause an effect, negative the determine_fluent_change function
        :param action:
        :return:
        """
        if self.use_physics:
            state, labels = self.observation_space.create_discrete_observation_from_simulator(
                self
            )
        else:
            state, labels = self.observation_space.create_discrete_observation_from_fsm(
                self
            )
        obj_name = action.obj
        if obj_name == "door":
            # door being movable depends on door lock
            if state[labels.index("door_lock")] == 1:
                return False
            else:
                return True
        active = state[labels.index(obj_name + "_active")]
        if active:
            return True
        else:
            return False

    def _export_results(self):
        save_count = len(glob(self.save_path + "results[0-9]*.csv"))
        np.savetxt(
            self.save_path + "results{}.csv".format(save_count),
            self.results,
            delimiter=",",
            fmt="%s",
        )

    def _execute_physics_action(self, action):
        """
        executes an action using the physics simulator
        :param action: action to execute
        :return: action_success: whether or not the action executed successfully
        """
        action_success = False
        if action.name == "goto":
            action_success = self._action_go_to(action)
        elif action.name == "goto_obj":
            action_success = self._action_go_to_obj(action)
        elif action.name == "rest":
            action_success = self._action_rest()
        elif action.name == "pull":
            action_success = self._action_pull(action)
        elif action.name == "push":
            action_success = self._action_push(action)
        elif action.name == "move":
            action_success = self._action_move(action)
        elif action.name == "move_end_frame":
            action_success = self._action_move_end_frame(action)
        elif action.name == "unlock":
            action_success = self._action_unlock(action)
        elif action.name == "reset":
            action_success = self._action_reset()
        elif action.name == "save":
            action_success = self._action_save()

        # update state machine after executing a action
        self.update_state_machine(action)

        return action_success

    def _action_go_to(self, twod_config):
        # get configuatin of end effector
        targ_x, targ_y, targ_theta = twod_config

        # draw arrow to show target location
        args = (targ_x, targ_y, targ_theta, 0.5, 1, common.Color(0.8, 0.8, 0.8))
        self.viewer.markers["targ_arrow"] = ("arrow", args)

        # update current config
        self.invkine.kinematic_chain.update_chain(self.world_def.get_rel_config())

        # generate discretized waypoints
        waypoints = discretize_path(
            self.invkine.kinematic_chain.get_total_delta_config(),
            common.TwoDConfig(targ_x, targ_y, targ_theta),
            ENV_SETTINGS["PATH_INTERP_STEP_DELTA"],
        )

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
            while err > ENV_SETTINGS["INVK_CONV_TOL"]:

                if a > ENV_SETTINGS["INVK_CONV_MAX_STEPS"]:
                    return False
                a = a + 1

                # get delta theta
                d_theta = self.invkine.get_delta_theta_dls(
                    lam=ENV_SETTINGS["INVK_DLS_LAMBDA"]
                )

                # current theta along convergence path
                cur_config = (
                    self.invkine.kinematic_chain.get_rel_config()
                )  # ignore virtual base link

                # new theta along convergence path

                # TODO: this is messy
                new_config = [cur_config[0]] + [
                    common.TwoDConfig(cur.x, cur.y, cur.theta + delta)
                    for cur, delta in zip(cur_config[1:], d_theta)
                ]

                # update inverse kinematics model to reflect step along convergence path
                self.invkine.kinematic_chain.update_chain(new_config)

                err = self.invkine.get_error()

            # theta found, update controllers and wait until controllers converge and stop
            if new_config:
                if not self.__update_and_converge_controllers(
                    [c.theta for c in new_config[1:]]
                ):
                    # could not converge
                    return False

        # succesfully reached target config

        # delete target arrow
        if "targ_arrow" in list(self.viewer.markers.keys()):
            del self.viewer.markers["targ_arrow"]

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

            end_effector_offset = (
                end_eff_shape.radius * normal
            )  # TODO: is this the right offset?

            desired_config = common.TwoDConfig(
                hit_point[0] + end_effector_offset[0],
                hit_point[1] + end_effector_offset[1],
                common.wrapToMinusPiToPi(angle),
            )

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
        delta = [
            common.wrapToMinusPiToPi(t - c) for t, c in zip(self.theta0, cur_theta)
        ]

        num_steps = max(
            [int(abs(d / ENV_SETTINGS["PATH_INTERP_STEP_DELTA"])) for d in delta]
        )

        if num_steps == 0:
            # we're already within step_delta of our desired config in all dimensions
            return True

        # TODO: refactor

        # generate discretized path
        waypoints = []
        for i in range(0, num_steps + 1):
            waypoints.append(
                [
                    common.wrapToMinusPiToPi(cur + i * d / num_steps)
                    for cur, d in zip(cur_theta, delta)
                ]
            )

        # sanity check: we actually reach the target config

        # TODO: arbitrary double comparison
        assert all(
            [
                abs(common.wrapToMinusPiToPi(waypoints[-1][i] - self.theta0[i])) < 0.01
                for i in range(0, len(self.theta0))
            ]
        )

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
        new_config = common.TwoDConfig(
            cur_x + neg_normal[0] * distance,
            cur_y + neg_normal[1] * distance,
            cur_theta,
        )

        if not self._action_go_to(new_config):
            self._action_grasp()  # remove connection
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
                if targ_fixture and not (
                    connection.bodyA == targ_fixture.body
                    or connection.bodyB == targ_fixture.body
                ):
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

                    if targ_fixture and not (
                        fix_A == targ_fixture or fix_B == targ_fixture
                    ):
                        continue
                    else:
                        # indiscriminate grab or found target

                        # find shortest distance between two shapes
                        dist_result = b2Distance(
                            shapeA=fix_A.shape,
                            shapeB=fix_B.shape,
                            transformA=fix_A.body.transform,
                            transformB=fix_B.body.transform,
                        )

                        point_A = fix_A.body.GetLocalPoint(dist_result.pointA)
                        point_B = fix_B.body.GetLocalPoint(dist_result.pointB)

                        # TODO experiment with other joints
                        self.world_def.grasped_list.append(
                            self.world_def.world.CreateDistanceJoint(
                                bodyA=fix_A.body,
                                bodyB=fix_B.body,
                                localAnchorA=point_A,
                                localAnchorB=point_B,
                                frequencyHz=1,
                                dampingRatio=1,
                                collideConnected=True,
                            )
                        )
                return True
            else:
                return False

    def _action_move(self, action):
        delta_x, delta_y, delta_theta = action.params

        cur_x, cur_y, cur_theta = self.world_def.get_abs_config()[-1]

        return self._action_go_to(
            common.TwoDConfig(cur_x + delta_x, cur_y + delta_y, cur_theta + delta_theta)
        )

    def _action_move_end_frame(self, params):
        delta_x, delta_y, delta_theta = params

        cur_x, cur_y, cur_theta = self.world_def.get_abs_config()[-1]

        x_axis = (np.cos(cur_theta), np.sin(cur_theta))
        y_axis = (-x_axis[1], x_axis[0])

        new_config = common.TwoDConfig(
            cur_x + x_axis[0] * delta_x + y_axis[0] * delta_y,
            cur_y + x_axis[1] * delta_x + y_axis[1] * delta_y,
            cur_theta + delta_theta,
        )

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


if __name__ == "__main__":
    main()
