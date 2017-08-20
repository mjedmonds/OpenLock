import gym
from Box2D import b2Color, b2_kinematicBody, b2_staticBody
from gym import spaces
from gym.utils import seeding

from gym_lock.box2d_renderer import Box2DRenderer
from gym_lock.common import *
from gym_lock.envs.world_defs.arm_lock_def import ArmLockDef
from gym_lock.kine import KinematicChain, discretize_path, InverseKinematics, generate_five_arm, \
    TwoDKinematicTransform
from gym_lock.settings import RENDER_SETTINGS, BOX2D_SETTINGS, ENV_SETTINGS


# TODO: add ability to move base
# TODO: more physically plausible units?

class ArmLockEnv(gym.Env):
    # Set this in SOME subclasses
    metadata = {'render.modes': ['human']}  # TODO what does this do?

    def __init__(self):

        # TODO: properly define these
        self.action_space = spaces.Discrete(5)  # up, down, left, right
        self.observation_space = spaces.Box(-np.inf, np.inf, [4])  # [x, y, vx, vy]
        self.reward_range = (-np.inf, np.inf)
        self.clock = 0
        self._seed()

        # initialize inverse kinematics module with chain==target
        theta0 = BOX2D_SETTINGS['INITIAL_THETA_VECTOR']
        base0 = BOX2D_SETTINGS['INITIAL_BASE_CONFIG']
        initial_config = generate_five_arm(theta0[0], theta0[1], theta0[2], theta0[3], theta0[4])
        self.base = TwoDKinematicTransform(x=base0.x, y=base0.y, theta=base0.theta)
        self.invkine = InverseKinematics(KinematicChain(self.base, initial_config),
                                         KinematicChain(self.base, initial_config))

        # setup Box2D world
        self.world_def = ArmLockDef(self.invkine.kinematic_chain, 1.0 / BOX2D_SETTINGS['FPS'], 30)

        # setup renderer
        self.viewer = Box2DRenderer(self.world_def.end_effector_grasp_callback)

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

            # render at desired frame rate
            if self.world_def.clock % RENDER_SETTINGS['RENDER_CLK_DIV'] == 0:
                self._render()

            # update error values
            theta_err = sum([e ** 2 for e in self.world_def.pos_controller.error])
            vel_err = sum([e ** 2 for e in self.world_def.vel_controller.error])
        return True

    def _step(self, action):
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.
        Accepts an action and returns a tuple (observation, reward, done, info).
        Args:
                action (TwoDConfig): desired end effector configuration
        Returns:
                observation (dict): END_EFFECTOR_POS : current end effector position
                                          LOCK_STATE : true if door is locked
                reward (float) : amount of reward returned after previous action
                done (boolean): whether the episode has ended, in which case further step() calls will return undefined results
                info (dict): CONVERGED : whether algorithm succesfully coverged on action
        """

        if action:

            # get configuatin of end effector
            targ_x, targ_y, targ_theta = action

            # draw arrow to show target location
            self.world_def.draw_target_arrow(targ_x, targ_y, targ_theta)

            # update current config
            self.invkine.kinematic_chain.update_chain(self.world_def.get_rel_config())

            # generate discretized waypoints
            waypoints = discretize_path(self.invkine.kinematic_chain.get_total_delta_config(),
                                        action,
                                        ENV_SETTINGS['PATH_INTERP_STEP_DELTA'])

            # we're already at the config
            if waypoints is None:
                # TODO: return something meaningful
                return self.world_def.get_state(), 0, False, {'CONVERGED': True}

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
                        return self.world_def.get_state(), 0, False, {'CONVERGED': False}
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
                        return self.world_def.get_state(), 0, False, {'CONVERGED': False}

            return self.world_def.get_state(), 0, False, {'CONVERGED': True}

        else:
            self.world_def.step(1.0 / BOX2D_SETTINGS['FPS'],
                                BOX2D_SETTINGS['VEL_ITERS'],
                                BOX2D_SETTINGS['POS_ITERS'])
            if self.world_def.clock % RENDER_SETTINGS['RENDER_CLK_DIV'] == 0:
                self._render()
            return self.world_def.get_state(), 0, False, {}

    def _reset(self):
        """Resets the state of the environment and returns an initial observation.

        Returns: observation (object): the initial observation of the
                space.
        """
        # initialize inverse kinematics module with chain==target
        # theta0 = BOX2D_SETTINGS['INITIAL_THETA_VECTOR']
        # base0 = BOX2D_SETTINGS['INITIAL_BASE_CONFIG']
        # initial_config = generate_five_arm(theta0[0], theta0[1], theta0[2], theta0[3], theta0[4])
        # self.base = TwoDKinematicTransform(x=base0.x, y=base0.y, theta=base0.theta)
        # self.invkine = InverseKinematics(KinematicChain(self.base, initial_config),
        #                                  KinematicChain(self.base, initial_config))
        #
        # # setup Box2D world
        # self.world_def = ArmLockDef(self.invkine.kinematic_chain, 1.0 / BOX2D_SETTINGS['FPS'], 30)
        # return self.world_def.get_state()

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

            # if self.viewer is None:
            self.viewer = Box2DRenderer(self.world_def.end_effector_grasp_callback)

        self.viewer.render_world(self.world_def.world, mode)

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
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # ADDITIONS



    def ManualDraw(self):
        """
        This implements code normally present in the C++ version, which calls
        the callbacks that you see in this class (DrawSegment, DrawSolidCircle,
        etc.).

        This is implemented in Python as an example of how to do it, and also a
        test.
        """
        colors = {
            'active': b2Color(0.5, 0.5, 0.3),
            'static': b2Color(0.5, 0.9, 0.5),
            'kinematic': b2Color(0.5, 0.5, 0.9),
            'asleep': b2Color(0.6, 0.6, 0.6),
            'default': b2Color(0.9, 0.7, 0.7),
        }

        settings = fwSettings
        world = self.world_def.world

        # if self.test.selected_shapebody:
        #     sel_shape, sel_body = self.test.selected_shapebody
        # else:
        #     sel_shape = None

        if settings.drawShapes:
            for body in world.bodies:
                transform = body.transform
                for fixture in body.fixtures:
                    shape = fixture.shape

                    if not body.active:
                        color = colors['active']
                    elif body.type == b2_staticBody:
                        color = colors['static']
                    elif body.type == b2_kinematicBody:
                        color = colors['kinematic']
                    elif not body.awake:
                        color = colors['asleep']
                    else:
                        color = colors['default']

                    self.DrawShape(fixture, transform,
                                   color)

                    # if settings.drawJoints:
                    #     for joint in world.joints:
                    #         self.DrawJoint(joint)
                    #
                    # # if settings.drawPairs
                    # #   pass
                    #
                    # if settings.drawAABBs:
                    #     color = b2Color(0.9, 0.3, 0.9)
                    #     # cm = world.contactManager
                    #     for body in world.bodies:
                    #         if not body.active:
                    #             continue
                    #         transform = body.transform
                    #         for fixture in body.fixtures:
                    #             shape = fixture.shape
                    #             for childIndex in range(shape.childCount):
                    #                 self.DrawAABB(shape.getAABB(
                    #                     transform, childIndex), color)


def main():
    env = ArmLockEnv()


if __name__ == '__main__':
    main()
