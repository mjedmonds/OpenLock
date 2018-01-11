import gym
import re
import time
import numpy as np
from shapely.geometry import Polygon, Point
from Box2D import b2Color, b2_kinematicBody, b2_staticBody, b2RayCastInput, b2RayCastOutput, b2Transform, b2Shape, b2Distance
from gym import spaces
from gym.utils import seeding

from gym_lock.box2d_renderer import Box2DRenderer
import gym_lock.common as common
from gym_lock.envs.world_defs.arm_lock_def import ArmLockDef
from gym_lock.kine import KinematicChain, discretize_path, InverseKinematics, generate_five_arm, TwoDKinematicTransform
from gym_lock.settings_render import RENDER_SETTINGS, BOX2D_SETTINGS, ENV_SETTINGS
from gym_lock.settings_trial import REWARD_NONE, REWARD_OPEN, REWARD_UNLOCK
from gym_lock.space_manager import ActionSpace, ObservationSpace

from glob import glob


# TODO: add ability to move base
# TODO: more physically plausible units?


class ArmLockEnv(gym.Env):
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
        self.logger = None      # logs participant data
        self.action_limit = None
        self.attempt_limit = None

        self.action_executing = False    # used to disable action preemption

        self.human_agent = True

    def _reset(self):
        """Resets the state of the environment and returns an initial observation.

        Returns: observation (object): the initial observation of the
                space.
        """
        if self.scenario is None:
            print('WARNING: resetting environment with no scenario')

        # TODO: properly define these
        self.observation_space = None
        self.reward_range = (0, 5)
        self.clock = 0
        self._seed()

        self.init_inverse_kine()

        # setup Box2D world
        self.world_def = ArmLockDef(self.invkine.kinematic_chain, 1.0 / BOX2D_SETTINGS['FPS'], 30, self.scenario)

        self.action_space, self.action_map = ActionSpace.create_action_space(self.world_def.obj_map)
        self.obs_space = ObservationSpace(len(self.world_def.get_locks()))

        # reset results (must be after world_def exists and action space has been created)
        self._reset_results()

        # setup renderer
        if not self.viewer:
            self.viewer = Box2DRenderer(self._action_grasp)

        self.viewer.reset()

        self._create_clickable_regions()

        # reset the finite state machine
        self.scenario.reset()
        self.action_count = 0

        self._render()
        state = self.get_state()
        # append initial observation
        # self._print_observation(state, self.action_count)
        self.results.append(self._create_state_entry(state, self.action_count))
        
        self._update_state_machine()

        return state

    def _step(self, action):
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

        if not action:
            self.world_def.step(1.0 / BOX2D_SETTINGS['FPS'],
                                BOX2D_SETTINGS['VEL_ITERS'],
                                BOX2D_SETTINGS['POS_ITERS'])

            self._render_world_at_frame_rate()

            state = self.get_state()
            state['SUCCESS'] = False
            self._update_state_machine()
            # no action, return nothing to indicate no reward possible
            return None
        # change to simple "else:" to enable action preemption
        elif self.action_executing is False:
            self.action_executing = True
            self.i += 1
            observable_action = self._create_pre_obs_entry(action)
            if observable_action:
                self.logger.cur_trial.cur_attempt.add_action(action.name + '_' + action.params[0])

            success = False
            if action.name == 'goto':
                success = self._action_go_to(action.params)
            elif action.name == 'goto_obj':
                success = self._action_go_to_obj(action.params)
            elif action.name == 'rest':
                success = self._action_rest()
            elif action.name == 'pull':
                success = self._action_pull(action.params)
            elif action.name == 'push':
                success = self._action_push(action.params)
            elif action.name == 'move':
                success = self._action_move(action.params)
            elif action.name == 'move_end_frame':
                success = self._action_move_end_frame(action.params)
            elif action.name == 'unlock':
                success = self._action_unlock(action.params)
            elif action.name == 'reset':
                success = self._action_reset()
            elif action.name == 'save':
                success = self._action_save()

            self.i += 1

            # update state machine after executing a action
            self._update_state_machine()
            state = self.get_state()
            state['SUCCESS'] = success

            if observable_action:
                self.action_count += 1
                # self._print_observation(state, self.action_count)
                self.results.append(self._create_state_entry(state, self.action_count))
                self.logger.cur_trial.cur_attempt.finish_action()

            # must update reward before potentially reset env (env may reset based on trial status)
            reward, success = self._determine_reward(action)

            # above the allowed number of actions, need to increment the attempt count and reset the simulator
            if self.action_limit is not None and self.action_count >= self.action_limit:

                self.attempt_count += 1
                attempt_success = self.logger.cur_trial.finish_attempt(results=self.results)

                # update the user about their progress
                trial_finished, pause = self._update_user(attempt_success)

                # pauses if the user unlocked the door but didn't push on the door
                if pause:
                    # pause for 4 sec to allow user to view lock
                    t_end = time.time() + 4
                    while time.time() < t_end:
                        self._render()
                        self._update_state_machine()

                if not trial_finished:
                    self._reset()  # reset if we are not done with this trial
                    self.logger.cur_trial.add_attempt()

            self.action_executing = False
            state = self.get_state()

            # update state machine in case there was a reset
            self._update_state_machine()
            return state, reward, success, {}
        else:
            state = self.get_state()
            self._update_state_machine()
            return None
            return state, 0, False, {}

    def _render(self, mode='human', close=False):
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

    def _create_state_entry(self, state, frame):
        entry = [0] * len(self.col_label)
        entry[0] = frame
        for name, val in state['OBJ_STATES'].items():
            entry[self.index_map[name]] = int(val)

        return entry

    def _create_pre_obs_entry(self, action):
        # create pre-observation entry
        entry = [0] * len(self.col_label)
        entry[0] = self.action_count
        # copy over previous state
        entry[1:self.index_map['agent']+1] = self.results[-1][1:self.index_map['agent']+1]

        # mark action idx
        if type(action.params[0]) is str:
            col = '{}_{}'.format(action.name, action.params[0])
        else:
            col = action.name

        observable_action = col in self.index_map

        if observable_action:
            entry[self.index_map[col]] = 1
            # append pre-observation entry
            self.results.append(entry)

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
            self._render()

    def _update_state_machine_at_frame_rate(self):
        ''''''
        if self.world_def.clock % BOX2D_SETTINGS['STATE_MACHINE_CLK_DIV'] == 0:
            self._update_state_machine()

    def _update_state_machine(self):
        self.scenario.update_state_machine()
   
    def _update_user(self, attempt_success):
        pause = False
        # continue or end trial
        if self.logger.cur_trial.success is True:
            print "INFO: You found all of the solutions. Ending trial."
            trial_finished = True
            pause = True            # pause if they open the door
        elif self.attempt_count < self.attempt_limit:
            # alert user to the number of solutions remaining
            if attempt_success is True:
                print "INFO: You found a solution. There are {} unique solutions remaining.".format(self.logger.cur_trial.num_solutions_remaining)
                pause = True            # pause if they open the door
            else:
                print "INFO: Ending attempt. Action limit reached. There are {} unique solutions remaining. You have {} attempts remaining.".format(self.logger.cur_trial.num_solutions_remaining, self.attempt_limit - self.attempt_count)
                # pause if the door lock is missing and the agent is a human
                if self.human_agent and self.get_state()['OBJ_STATES']['door_lock'] is False:
                    pause = True
            trial_finished = False
        else:
            print "INFO: Ending trial. Attempt limit reached. You found {} unique solutions".format(
                len(self.logger.cur_trial.solutions) - self.logger.cur_trial.num_solutions_remaining)
            trial_finished = True

        return trial_finished, pause

    def _determine_reward(self, action, attempt_success=False):
        # todo: this reward does not consider whether or not the action sequence has been finished before
        # todo: success also has the same limitation
        success = False
        if self.world_def.door_lock is not None:
            reward = REWARD_NONE
        elif self.world_def.door_lock is None and action.name is 'push' and action.params[0] is 'door':
            reward = REWARD_OPEN
            success = True
        else:
            reward = REWARD_UNLOCK

        return reward, success

    def init_inverse_kine(self):
        # initialize inverse kinematics module with chain==target
        self.theta0 = BOX2D_SETTINGS['INITIAL_THETA_VECTOR']
        self.base0 = BOX2D_SETTINGS['INITIAL_BASE_CONFIG']
        initial_config = generate_five_arm(self.theta0[0], self.theta0[1], self.theta0[2], self.theta0[3], self.theta0[4])
        self.base = TwoDKinematicTransform(x=self.base0.x, y=self.base0.y, theta=self.base0.theta)
        self.invkine = InverseKinematics(KinematicChain(self.base, initial_config),
                                         KinematicChain(self.base, initial_config))

    
    def _print_observation(self, state, count):
        print str(count) + ': ' + str(state['OBJ_STATES'])
        print str(count) + ': ' + str(state['_FSM_STATE'])

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
        for b2_object_name, b2_object_data in self.world_def.obj_map.items():
            if re.search(lock_regex, b2_object_name) or re.search(inactive_lock_regex, b2_object_name):
                lock = b2_object_data

                lock.create_clickable(self._step, self.action_map)
                self.viewer.register_clickable_region(lock.inner_clickable)
                self.viewer.register_clickable_region(lock.outer_clickable)
                # lock inactive levers
                if re.search(inactive_lock_regex, b2_object_name):
                    self.world_def.lock_lever(lock.name)
            elif b2_object_name == 'door_right_button':
                door_button = b2_object_data
                callback_action = 'push_door'
                door_button.create_clickable(self._step, self.action_map, self.action_map[callback_action])
                self.viewer.register_clickable_region(door_button.clickable)
            elif b2_object_name == 'door_left_button':
                door_button = b2_object_data
                callback_action = 'pull_door'
                door_button.create_clickable(self._step, self.action_map, self.action_map[callback_action])
                self.viewer.register_clickable_region(door_button.clickable)
            elif b2_object_name == 'reset_button':
                reset_button = b2_object_data
                callback_action = 'reset'
                reset_button.create_clickable(self._step, self.action_map,
                                              common.Action(callback_action, (reset_button, 4)))
                self.viewer.register_clickable_region(reset_button.clickable)
            elif b2_object_name == 'save_button':
                save_button = b2_object_data
                callback_action = 'save'
                save_button.create_clickable(self._step, self.action_map,
                                             common.Action(callback_action, (save_button, 4)))
                self.viewer.register_clickable_region(save_button.clickable)

    def get_state(self):
        if self.world_def is None:
            raise ValueError('world_def is None while trying to call get_state()')
        return self.world_def.get_state()

    def _export_results(self):
        save_count = len(glob(self.save_path + 'results[0-9]*.csv'))
        np.savetxt(self.save_path + 'results{}.csv'.format(save_count), self.results, delimiter=',', fmt='%s')

    def _action_go_to(self, config):
        # get configuatin of end effector
        targ_x, targ_y, targ_theta = config

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
        if 'targ_arrow' in self.viewer.markers.keys():
            del self.viewer.markers['targ_arrow']

        return True

    def _action_go_to_obj(self, params):
        """

        Args:
            object: reference to Box2D fixture that you want to go to

        Returns:

        """
        obj = self.world_def.obj_map[params].fixture

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

    def _action_pull(self, params):
        name, distance = params

        if not self._action_go_to_obj(name):
            return False

        if not self._action_grasp():
            print 'no'
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

    def _action_push(self, params):
        name, distance = params

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

    def _action_move(self, params):
        delta_x, delta_y, delta_theta = params

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
        self._reset()
        return True

    def _action_save(self):
        self._export_results()
        self._reset()
        return True

    def _action_nothing(self):
        return True

    def get_avail_actions(self):
        return self.world_def.scenario.actions


def main():
    env = ArmLockEnv()


if __name__ == '__main__':
    main()
