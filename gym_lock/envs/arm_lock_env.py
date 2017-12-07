import gym
import re
from shapely.geometry import Polygon, Point
from Box2D import b2Color, b2_kinematicBody, b2_staticBody, b2RayCastInput, b2Transform, b2Shape, b2Distance
from gym import spaces
from gym.utils import seeding

from gym_lock.box2d_renderer import Box2DRenderer
from gym_lock.common import *
from gym_lock.envs.world_defs.arm_lock_def import ArmLockDef, b2RayCastOutput
from gym_lock.kine import KinematicChain, discretize_path, InverseKinematics, generate_five_arm, \
    TwoDKinematicTransform
from gym_lock.settings_render import RENDER_SETTINGS, BOX2D_SETTINGS, ENV_SETTINGS, CURRENT_SCENARIO
from glob import glob


# TODO: add ability to move base
# TODO: more physically plausible units?


class ArmLockEnv(gym.Env):
    # Set this in SOME subclasses
    metadata = {'render.modes': ['human']}  # TODO what does this do?

    def __init__(self):
        self.viewer = None

        self.scenario = CURRENT_SCENARIO # handle to the scenario, defined by the scenario

        init_obs = self._reset()

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
        
        self.i = 0
        self.save_path = '../../OpenLockResults/'

        # append initial observation
        self.results.append(self.create_state_entry(init_obs, self.i))

    def create_state_entry(self, state, i):
        entry = [0] * len(self.col_label)
        entry[0] = i
        for name, val in state['OBJ_STATES'].items():
            entry[self.index_map[name]] = int(val)

        return entry

    def __update_and_converge_controllers(self, new_theta):
        self.world_def.set_controllers(new_theta)
        b = 0
        theta_err = sum([e ** 2 for e in self.world_def.pos_controller.error])
        vel_err = sum([e ** 2 for e in self.world_def.vel_controller.error])
        while (theta_err > ENV_SETTINGS['PID_POS_CONV_TOL'] \
                       or vel_err > ENV_SETTINGS['PID_VEL_CONV_TOL']):

            if b > ENV_SETTINGS['PID_CONV_MAX_STEPS']:
                return False

            b += 1
            self.world_def.step(1.0 / BOX2D_SETTINGS['FPS'],
                                BOX2D_SETTINGS['VEL_ITERS'],
                                BOX2D_SETTINGS['POS_ITERS'])

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
            return state, 0, False, {}
        else:
            self.i += 1
            # create pre-observation entry
            entry = [0] * len(self.col_label)
            entry[0] = self.i
            # copy over previous state
            entry[1:self.index_map['agent']] = self.results[-1][1:self.index_map['agent']]

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
                success = self._action_reset(action.params)
            elif action.name == 'save':
                success = self._action_save(action.params)

            state = self.get_state()
            state['SUCCESS'] = success
            self.i += 1

            if observable_action:
                print state['OBJ_STATES']
                print state['_FSM_STATE']
                self.results.append(self.create_state_entry(state, self.i))

            return state, 0, False, {}

    def _reset(self):
        """Resets the state of the environment and returns an initial observation.

        Returns: observation (object): the initial observation of the
                space.
        """
        # TODO: properly define these
        self.observation_space = spaces.Box(-np.inf, np.inf, [4])  # [x, y, vx, vy]
        self.reward_range = (-np.inf, np.inf)
        self.clock = 0
        self._seed()

        # initialize inverse kinematics module with chain==target
        self.theta0 = BOX2D_SETTINGS['INITIAL_THETA_VECTOR']
        self.base0 = BOX2D_SETTINGS['INITIAL_BASE_CONFIG']
        initial_config = generate_five_arm(self.theta0[0], self.theta0[1], self.theta0[2], self.theta0[3], self.theta0[4])
        self.base = TwoDKinematicTransform(x=self.base0.x, y=self.base0.y, theta=self.base0.theta)
        self.invkine = InverseKinematics(KinematicChain(self.base, initial_config),
                                         KinematicChain(self.base, initial_config))

        # setup Box2D world
        self.world_def = ArmLockDef(self.invkine.kinematic_chain, 1.0 / BOX2D_SETTINGS['FPS'], 30, self.scenario)

        self.action_space = []
        self.action_map = dict()
        for obj, val in self.world_def.obj_map.items():
            if 'button' not in obj:
                push = 'push_{}'.format(obj)
                pull = 'pull_{}'.format(obj)

                self.action_space.append(pull)
                self.action_space.append(push)

                self.action_map[push] = Action('push', (obj, 4))
                self.action_map[pull] = Action('pull', (obj, 4))

        # setup renderer
        if not self.viewer:
            self.viewer = Box2DRenderer(self._action_grasp)

            lock_regex = '^l[0-9]+$'
            # register clickable regions
            print 'register?'
            for b2_object_name, b2_object_data in self.world_def.obj_map.items():
                if re.search(lock_regex, b2_object_name):
                    lock = b2_object_data

                    lock.create_clickable(self._step, self.action_map)
                    self.viewer.register_clickable_region(lock.inner_clickable)
                    self.viewer.register_clickable_region(lock.outer_clickable)

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
                    reset_button.create_clickable(self._step, self.action_map, Action(callback_action, (reset_button, 4)))
                    self.viewer.register_clickable_region(reset_button.clickable)
                elif b2_object_name == 'save_button':
                    save_button = b2_object_data
                    callback_action = 'save'
                    save_button.create_clickable(self._step, self.action_map, Action(callback_action, (save_button, 4)))
                    self.viewer.register_clickable_region(save_button.clickable)


        self.viewer.reset()
        self._render()

        return self.get_state()

    def get_state(self):
        return {
            'END_EFFECTOR_POS': self.world_def.get_abs_config()[-1],
            'END_EFFECTOR_FORCE': TwoDForce(self.world_def.contact_listener.norm_force, self.world_def.contact_listener.tan_force),
            # 'DOOR_ANGLE' : self.obj_map['door'][1].angle,
            # 'LOCK_TRANSLATIONS' : {name : val[1].translation for name, val in self.obj_map.items() if name != 'door'},
            'OBJ_STATES': {name: val.ext_test(val.joint) for name, val in self.world_def.obj_map.items() if 'button' not in name},  # ext state
            # 'OBJ_STATES': {name: val[3](val[1]) for name, val in self.obj_map.items() if 'button' not in name},
        # ext state
            'LOCK_STATE': self.world_def.obj_map['door'].int_test(self.world_def.obj_map['door'].joint),
            # 'LOCK_STATE': self.obj_map['door'][2](self.obj_map['door'][1]),
            '_FSM_STATE': self.scenario.fsmm.observable_fsm.state + self.scenario.fsmm.latent_fsm.state if self.scenario is not None else ValueError('No Scenario set'),
        }

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

        def printer(x):
            print x
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
                return

#        if self.viewer is None:
#            print 'legglo'
#            self.viewer = Box2DRenderer(self._action_grasp)
            
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

    # TODO por the rest of this over
    # def manual_draw(self):
    #     """
    #     This implements code normally present in the C++ version, which calls
    #     the callbacks that you see in this class (DrawSegment, DrawSolidCircle,
    #     etc.).
    #
    #     This is implemented in Python as an example of how to do it, and also a
    #     test.
    #     """
    #     colors = RENDER_SETTINGS['COLORS']
    #
    #     world = self.world_def.world
    #
    #     # if self.test.selected_shapebody:
    #     #     sel_shape, sel_body = self.test.selected_shapebody
    #     # else:
    #     #     sel_shape = None
    #
    #     if RENDER_SETTINGS['DRAW_SHAPES']:
    #         for body in world.bodies:
    #             transform = body.transform
    #             for fixture in body.fixtures:
    #                 shape = fixture.shape
    #
    #                 if not body.active:
    #                     color = colors['active']
    #                 elif body.type == b2_staticBody:
    #                     color = colors['static']
    #                 elif body.type == b2_kinematicBody:
    #                     color = colors['kinematic']
    #                 elif not body.awake:
    #                     color = colors['asleep']
    #                 else:
    #                     color = colors['default']
    #
    #                 # self.DrawShape(fixture, transform,
    #                 #                color)
    #
    #                 # if settings.drawJoints:
    #                 #     for joint in world.joints:
    #                 #         self.DrawJoint(joint)
    #                 #
    #                 # # if settings.drawPairs
    #                 # #   pass
    #                 #
    #                 # if settings.drawAABBs:
    #                 #     color = b2Color(0.9, 0.3, 0.9)
    #                 #     # cm = world.contactManager
    #                 #     for body in world.bodies:
    #                 #         if not body.active:
    #                 #             continue
    #                 #         transform = body.transform
    #                 #         for fixture in body.fixtures:
    #                 #             shape = fixture.shape
    #                 #             for childIndex in range(shape.childCount):
    #                 #                 self.DrawAABB(shape.getAABB(
    #                 #                     transform, childIndex), color)

    # TODO: return states

    def _action_go_to(self, config):
        # get configuatin of end effector
        targ_x, targ_y, targ_theta = config

        # draw arrow to show target location
        args = (targ_x, targ_y, targ_theta, 0.5, 1, Color(0.8, 0.8, 0.8))
        self.viewer.markers['targ_arrow'] = ('arrow', args)

        # update current config
        self.invkine.kinematic_chain.update_chain(self.world_def.get_rel_config())

        # generate discretized waypoints
        waypoints = discretize_path(self.invkine.kinematic_chain.get_total_delta_config(),
                                    TwoDConfig(targ_x, targ_y, targ_theta),
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
            while (err > ENV_SETTINGS['INVK_CONV_TOL']):

                if a > ENV_SETTINGS['INVK_CONV_MAX_STEPS']:
                    return False
                a = a + 1

                # get delta theta
                d_theta = self.invkine.get_delta_theta_dls(lam=ENV_SETTINGS['INVK_DLS_LAMBDA'])

                # current theta along convergence path
                cur_config = self.invkine.kinematic_chain.get_rel_config()  # ignore virtual base link

                # new theta along convergence path

                # TODO: this is messy
                new_config = [cur_config[0]] + [TwoDConfig(cur.x, cur.y, cur.theta + delta) for cur, delta in
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
        object = self.world_def.obj_map[params].fixture

        # find face facing us by raycasting from end effector to center of fixture
        end_eff = self.world_def.end_effector_fixture
        end_eff_shape = end_eff.shape
        end_eff_mass_data = end_eff.massData
        obj_mass_data = object.massData

        object_center = object.body.GetWorldPoint(obj_mass_data.center)
        end_effector_center = end_eff.body.GetWorldPoint(end_eff_mass_data.center)

        input = b2RayCastInput(p1=end_effector_center, p2=object_center, maxFraction=200)
        output = b2RayCastOutput()

        hit = object.RayCast(output, input, 0)
        if hit:
            hit_point = input.p1 + output.fraction * (input.p2 - input.p1)
            normal = output.normal

            angle = np.arctan2(-normal[1], -normal[0])


            end_effector_offset = end_eff_shape.radius * normal # TODO: is this the right offset?

            desired_config = TwoDConfig(hit_point[0] + end_effector_offset[0],
                                        hit_point[1] + end_effector_offset[1],
                                        wrapToMinusPiToPi(angle))

            self._action_go_to(desired_config)

            # we way have gotten close to object, but lets move forward until we graze
            # TODO: selective tolerance of INVK/PID controllers for rough/fine movement
            i = 0
            while (len(self.world_def.arm_bodies[-1].contacts) == 0 and i < 5):
                i += 1
                self._action_move_end_frame(TwoDConfig(0.5, 0, 0))
            return True if len(self.world_def.arm_bodies[-1].contacts) > 0 else False
        else:
            # path is blocked
            return False

    def _action_rest(self):
        # discretize path
        cur_theta = [cur.theta for cur in self.world_def.get_rel_config()[1:]]

        # calculate number of discretized steps
        delta = [wrapToMinusPiToPi(t - c) for t, c in zip(self.theta0, cur_theta)]

        num_steps = max([int(abs(d / ENV_SETTINGS['PATH_INTERP_STEP_DELTA'])) for d in delta])

        if num_steps == 0:
            # we're already within step_delta of our desired config in all dimensions
            return True

        #TODO: refactor

        # generate discretized path
        waypoints = []
        for i in range(0, num_steps + 1):
            waypoints.append([wrapToMinusPiToPi(cur + i * d / num_steps) \
                              for cur, d in zip(cur_theta, delta)])

        # sanity check: we actually reach the target config

        # TODO: arbitrary double comparison
        assert all([abs(wrapToMinusPiToPi(waypoints[-1][i] - self.theta0[i]))  < 0.01 for i in range(0, len(self.theta0))])

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
        new_config = TwoDConfig(cur_x + neg_normal[0] * distance,
                                cur_y + neg_normal[1] * distance,
                                cur_theta)

        if not self._action_go_to(new_config):
            return False

        if not self._action_grasp():
            return False

        return True

    def _action_push(self, params):
        name, distance = params

        if not self._action_go_to_obj(name):
            return False

        if not self._action_move_end_frame(TwoDConfig(distance, 0, 0)):
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

        return self._action_go_to(TwoDConfig(cur_x + delta_x,
                                             cur_y + delta_y,
                                             cur_theta + delta_theta))

    def _action_move_end_frame(self, params):
        delta_x, delta_y, delta_theta = params

        cur_x, cur_y, cur_theta = self.world_def.get_abs_config()[-1]

        x_axis = (np.cos(cur_theta), np.sin(cur_theta))
        y_axis = (-x_axis[1], x_axis[0])

        new_config = TwoDConfig(cur_x + x_axis[0] * delta_x + y_axis[0] * delta_y,
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

    def _action_reset(self, params):
        self._reset()
        return True

    def _action_save(self, params):
        save_count = len(glob(self.save_path + 'results[0-9]*.csv'))
        np.savetxt(self.save_path + 'results{}.csv'.format(save_count), self.results, delimiter=',', fmt='%s')
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
