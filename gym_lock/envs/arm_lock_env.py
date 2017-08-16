import Box2D as b2
import gym
import numpy as np
from gym import spaces
from gym.envs.classic_control import rendering
from gym.utils import seeding
from matplotlib import pyplot as plt
from gym_lock.common import FPS
from gym_lock.envs.world_defs.arm_lock_def import ArmLockDef
from gym_lock.kine import KinematicChain, discretize_path, InverseKinematics, generate_four_arm, TwoDKinematicTransform

VIEWPORT_W = 1200
VIEWPORT_H = 800
SCALE = 30.0  # affects how fast-paced the game is, forces should be adjusted as well


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
        self.viewer = None
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
        self.world_def = ArmLockDef(self.chain, 35)

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

            all_dtheta = []

            # update current config
            self.update_current_config()

            # generate discretized waypoints
            waypoints = discretize_path(self.chain, action, self.step_delta)
            print action
            print len(waypoints)

            # we're already at the config
            if waypoints is None:
                return np.zeros(4), 0, True, dict()

            for i in range(1, len(waypoints)):  # waypoint 0 is current config

                # update current configuration
                self.update_current_config()

                # update invk
                self.invkine.set_current_config(self.chain)
                self.invkine.set_target(waypoints[i])

                print 'converging'
                a = 0
                err = self.invkine.get_error()
                while (err > 1):
                    a = a + 1

                    # get delta theta
                    d_theta = self.invkine.get_delta_theta_dls(lam=0.5)
                    all_dtheta.append(d_theta)

                    # update controllers
                    self.world_def.set_controllers(d_theta)

                    # for i in range(0, 500):
                    #     self.world_def.step(1.0 / FPS, 10, 10)
                        # print self.world_def.pos_controller.error

                    # wait for PID to converge and stop
                    print 'converging on theta'
                    b = 0
                    # self.world_def.step(1.0 / FPS, 10, 10)
                    theta_err = sum([e ** 2 for e in self.world_def.pos_controller.error])
                    vel_err = sum([e ** 2 for e in self.world_def.vel_controller.error])
                    while (theta_err > 0.001 or vel_err > 0.001):
                        if b % 10 == 0 and b > 250:
                            self._render()
                        if b > 2000:
                            # print self.world_def.lock_joint.translation
                            return np.zeros(4), 0, False, dict()

                        b += 1
                        self.world_def.step(1.0 / FPS, 10, 10)
                        theta_err = sum([e ** 2 for e in self.world_def.pos_controller.error])
                        vel_err = sum([e ** 2 for e in self.world_def.vel_controller.error])
                        # joint_speed = sum([joint.speed ** 2 for joint in self.world_def.arm_joints])
                        # if b % 10 == 0:
                        #     self._render()
                    print 'converged on theta in {} iterations'.format(b)

                    # print 'PID converged in {} iterations'.format(b)

                    # update current config
                    self.update_current_config()
                    # update inverse kine
                    self.invkine.set_current_config(self.chain)
                    err = self.invkine.get_error()
                    if a > 50:
                        print d_theta
                        self._render()

                print 'waypoint converged in {} iterations'.format(a)
                self._render()

                # converged on that waypoint
            # if len(all_dtheta) > 0:
            #     print d_theta
            #     for i in range(0, len(all_dtheta[0])):
            #         plt.plot(all_dtheta[:][i])
            # plt.show()

            return np.zeros(4), 0, True, dict()
        else:
            self.world_def.step(1.0 / FPS, 10, 10)
            return np.zeros(4), 0, False, dict()

    # def _step(self, action):
    #     """Run one timestep of the environment's dynamics. When end of
    #     episode is reached, you are responsible for calling `reset()`
    #     to reset this environment's state.
    #
    #     Accepts an action and returns a tuple (observation, reward, done, info).
    #
    #     Args:
    #             action (object): an action provided by the environment
    #
    #     Returns:
    #             observation (object): agent's observation of the current environment
    #             reward (float) : amount of reward returned after previous action
    #             done (boolean): whether the episode has ended, in which case further step() calls will return undefined results
    #             info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
    #     """
    #     # action = virtual KinematicLink
    #
    #     if action:
    #
    #         # set target
    #         self.target = action
    #
    #         # update current configuration
    #         cur_theta = [c.theta for c in self.world_def.get_rel_config()[1:]] # ignore virtual base link
    #         new_conf = generate_four_arm(cur_theta[0], cur_theta[1], cur_theta[2], cur_theta[3])
    #         self.chain = KinematicChain(new_conf)
    #
    #         # update invk model
    #         self.invkine.set_current_config(self.chain)
    #         self.invkine.set_target(self.target)
    #
    #         # print 'converging'
    #         a = 0
    #         err = self.invkine.get_error()
    #         while (err > self.epsilon):
    #             a = a + 1
    #
    #             print 'a: {} err: {}'.format(a, err)
    #
    #             # update current configuration
    #             cur_theta = [c.theta for c in self.world_def.get_rel_config()[1:]]  # ignore virtual base link
    #             new_conf = generate_four_arm(cur_theta[0], cur_theta[1], cur_theta[2], cur_theta[3])
    #             self.chain = KinematicChain(new_conf)
    #
    #             # update inverse kine
    #             self.invkine.set_current_config(self.chain)
    #
    #             # get delta theta
    #             d_theta = self.invkine.get_delta_theta_dls(lam=1, clamp_theta=0.05, clamp_err=False)
    #             print d_theta
    #             # d_theta = self.invkine.get_delta_theta_trans()
    #
    #             # update controllers
    #             self.world_def.set_controllers(d_theta)
    #
    #             # step
    #             for i in range(0, 5):
    #                 self.world_def.step(1.0 / FPS, 10, 10)
    #
    #             # update error
    #             err = self.invkine.get_error()
    #             super(ArmLockEnv, self).render()
    #
    #
    #             # print 'converged in {} iterations'.format(a)
    #         # converged on that waypoint
    #
    #     self.world_def.step(1.0 / FPS, 10, 10)
    #     return np.zeros(4), 0, False, dict()

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

        if self.viewer is None:
            self.viewer = rendering.Viewer(VIEWPORT_W, VIEWPORT_H)
            self.viewer.set_bounds(-VIEWPORT_W / SCALE, VIEWPORT_W / SCALE, -VIEWPORT_H / SCALE, VIEWPORT_H / SCALE)

        for body in self.world_def.world:
            for fixture in body.fixtures:
                t = body.transform
                if isinstance(fixture.shape, b2.b2EdgeShape):
                    self.viewer.draw_line(fixture.shape.vertices[0], fixture.shape.vertices[1])
                elif isinstance(fixture.shape, b2.b2CircleShape):
                    # print fixture.body.transform
                    trans = rendering.Transform(translation=t * fixture.shape.pos)
                    self.viewer.draw_circle(fixture.shape.radius).add_attr(trans)
                elif isinstance(fixture.shape, b2.b2PolygonShape):
                    vertices = [fixture.body.transform * v for v in fixture.shape.vertices]
                    self.viewer.draw_polygon(vertices, filled=False)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

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


def main():
    env = ArmLockEnv()


if __name__ == '__main__':
    main()
