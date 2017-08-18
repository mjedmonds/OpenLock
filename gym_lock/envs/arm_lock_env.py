import Box2D as b2
from Box2D import b2Color, b2_kinematicBody, b2_dynamicBody, b2_pi, b2CircleShape, b2Mul, b2PolygonShape, b2EdgeShape, \
    b2_staticBody
import gym
import numpy as np
from gym import spaces
from gym_lock.box2d_renderer import Box2DRenderer
from gym_lock.envs.pyglet_framework import PygletFramework
from gym_lock.envs.settings import fwSettings
from gym.utils import seeding
from matplotlib import pyplot as plt
from gym_lock.common import FPS, Color
from gym_lock.envs.world_defs.arm_lock_def import ArmLockDef
from gym_lock.kine import KinematicChain, discretize_path, InverseKinematics, generate_four_arm, TwoDKinematicTransform

VIEWPORT_W = 800
VIEWPORT_H = 800
SCALE = 15.0  # affects how fast-paced the game is, forces should be adjusted as well
RENDER_DIV = 100

class ArmLockEnv(gym.Env):
    # Set this in SOME subclasses
    metadata = {'render.modes': ['human']}  # TODO what does this do?

    ## Override in SOME subclasses
    # def _close(self):
    #        pass

    # Set these in ALL subclasses

    def __init__(self):

        self.action_space = spaces.Discrete(5)  # up, down, left, right
        self.observation_space = spaces.Box(-np.inf, np.inf, [4])  # [x, y, vx, vy]
        self.reward_range = (-np.inf, np.inf)
        self._seed()
        self.clock = 0


        # inverse kinematics params
        self.alpha = 0.01  # for invk transpose alg
        self.lam = 1  # for invk dls alg
        self.epsilon = 0.01  # for convergence on path waypoint
        self.step_delta = 0.1  # for path discretization

        # initialize inverse kinematics module with chain==target
        initial_config = generate_four_arm(np.pi, np.pi/2, np.pi/2, 0, np.pi/2)
        self.base = TwoDKinematicTransform()
        self.chain = KinematicChain(self.base, initial_config)
        self.target = KinematicChain(self.base, initial_config)
        self.invkine = InverseKinematics(self.chain, self.target)

        # setup Box2D world
        self.world_def = ArmLockDef(self.chain, 30)

        # setup rendering
        # self.viewer = None
        self.viewer = Box2DRenderer(self.world_def.end_effector_grasp)

    def update_current_config(self):
        cur_theta = [c.theta for c in self.world_def.get_rel_config()[1:]]
        new_conf = generate_four_arm(cur_theta[0], cur_theta[1], cur_theta[2], cur_theta[3], cur_theta[4])
        self.chain = KinematicChain(self.base, new_conf)

    def _step(self, action):
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.
        Accepts an action and returns a tuple (observation, reward, done, info).
        Args:
                action (object): an action provided by the environment
        Returns:
                observation (object): agent's observation of the current environment
                reward (float) : amount of reward returned after previous action
                done (boolean): whether the episode has ended, in which case further step() calls will return undefined results
                info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """

        if action:

            targ_x, targ_y, targ_theta = action.get_abs_config()[-1]
            self.world_def.draw_target_arrow(targ_x, targ_y, targ_theta)

            # update current config
            self.update_current_config()

            # generate discretized waypoints
            waypoints = discretize_path(self.chain, action, self.step_delta)
            # print action
            # print len(waypoints)

            # we're already at the config
            if waypoints is None:
                return np.zeros(4), 0, True, dict()

            for i in range(1, len(waypoints)):  # waypoint 0 is current config

                # update kinematics model to reflect current world config
                self.update_current_config()

                # update inverse kinematics
                self.invkine.set_current_config(self.chain)
                self.invkine.set_target(waypoints[i])


                # find inverse kinematics solution
                print 'converging'
                a = 0
                err = self.invkine.get_error()
                new_theta = None
                while (err > 0.01):
                    a = a + 1

                    if a > 5000:
                        return np.zeros(4), 0, True, dict()

                    # get delta theta
                    d_theta = self.invkine.get_delta_theta_dls(lam=0.75)

                    # current theta along convergence path
                    cur_theta = [c.theta for c in self.invkine.kinematic_chain.get_rel_config()[1:]]  # ignore virtual base link

                    # new theta along convergence path
                    new_theta = [cur + delta for cur, delta in zip(cur_theta, d_theta)]

                    # update inverse kinematics model to reflect step along convergence path
                    self.invkine.set_current_config(KinematicChain(self.base, generate_four_arm(new_theta[0],
                                                                                                new_theta[1],
                                                                                                new_theta[2],
                                                                                                new_theta[3],
                                                                                                new_theta[4])))

                    err = self.invkine.get_error()
                print 'waypoint converged in {} iterations'.format(a)
                self.chain = self.invkine.kinematic_chain

                # theta found, update controllers and wait until controllers converge and stop
                if new_theta:
                    self.world_def.set_controllers(new_theta)
                    print 'converging on theta'
                    b = 0
                    # self.world_def.step(1.0 / FPS, 10, 10)
                    theta_err = sum([e ** 2 for e in self.world_def.pos_controller.error])
                    vel_err = sum([e ** 2 for e in self.world_def.vel_controller.error])
                    while (theta_err > 0.001 or vel_err > 0.0001):
                        if b > 2000:
                            # print self.world_def.lock_joint.translation
                            return np.zeros(4), 0, False, dict()

                        b += 1
                        self.world_def.step(1.0 / FPS, 10, 10)
                        if self.world_def.clock % 10 == 0:
                            self._render()
                        theta_err = sum([e ** 2 for e in self.world_def.pos_controller.error])
                        vel_err = sum([e ** 2 for e in self.world_def.vel_controller.error])
                        # joint_speed = sum([joint.speed ** 2 for joint in self.world_def.arm_joints])
                        # if b % 10 == 0:
                        #     self._render()
                    print 'converged on theta in {} iterations'.format(b)


                # converged on that waypoint
            # if len(all_dtheta) > 0:
            #     print d_theta
            #     for i in range(0, len(all_dtheta[0])):
            #         plt.plot(all_dtheta[:][i])
            # plt.show()

            return np.zeros(4), 0, True, dict()
        else:
            self.world_def.step(1.0 / FPS, 10, 10)
            if self.world_def.clock % RENDER_DIV == 0:
                self._render()
            return np.zeros(4), 0, False, dict()

    def _reset(self):
        """Resets the state of the environment and returns an initial observation.

        Returns: observation (object): the initial observation of the
                space.
        """
        pass

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
            self.viewer = Box2DRenderer(self.world_def.end_effector_grasp)

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
