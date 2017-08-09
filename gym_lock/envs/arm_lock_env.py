import Box2D as b2
import gym
import numpy as np
from gym import spaces
from gym.envs.classic_control import rendering
from gym.utils import seeding
from Queue import Queue

from gym_lock.envs.world_defs.arm_lock_def import ArmLockDef
from gym_lock.kine import KinematicChain, KinematicLink, InverseKinematics, generate_valid_config
from gym_lock.common import transform_to_theta, wrapToMinusPiToPi

VIEWPORT_W = 1200
VIEWPORT_H = 800
SCALE = 30.0  # affects how fast-paced the game is, forces should be adjusted as well
FPS = 30


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

        # inverse kinematics params
        self.alpha = 0.01 # for invk transpose alg
        self.lam = 1 # for invk dls alg
        self.epsilon = 0.1 # for convergence on path waypoint
        self.step_delta = 0.05 # for path discretization

        # initialize inverse kinematics module with chain==target
        initial_config = generate_valid_config(0, 0, 0)
        self.chain = KinematicChain(initial_config)
        self.target = KinematicChain(generate_valid_config(np.pi, 0, 0))
        self.invkine = InverseKinematics(self.chain, self.target)

        # setup Box2D world
        self.world_def = ArmLockDef(self.chain.get_abs_config(), 25)

    def _discretize_path(self, action):

        # calculate number of discretized steps
        cur = self.chain.get_total_delta_config()
        targ = action.get_total_delta_config()

        delta = [t - c for t, c in zip(targ, cur)]

        num_steps = max([int(abs(d / self.step_delta)) for d in delta])

        if num_steps == 0:
            return None

        # generate discretized path
        waypoints = []
        for i in range(1, num_steps + 1):
            waypoints.append(KinematicLink(x=cur.x + i * delta[0] / num_steps,
                                         y=cur.y + i * delta[1] / num_steps,
                                         theta=wrapToMinusPiToPi(cur.theta + i * delta[2] / num_steps)))

        # sanity check: we actually reach the target config

        assert np.allclose(waypoints[-1].get_transform(), action.get_transform())
        return waypoints

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
        # action = virtual KinematicLink

        if action:

           # update current configuration
           cur_theta = [c.theta for c in self.world_def.get_rel_config()[1:]] # ignore virtual base link
           new_conf = generate_valid_config(cur_theta[0], cur_theta[1], cur_theta[2])
           self.chain = KinematicChain(new_conf)

           # update invk model
           self.invkine.set_current_config(self.chain)

           # generate discretized waypoints
           waypoints = self._discretize_path(action)

           if waypoints is None:
               return np.zeros(4), 0, True, dict()

           for i in range(0, len(waypoints)):

                # set next waypoint
                self.invkine.set_target(waypoints[i])

                # while err > eta, converge
                err = self.invkine.get_error()  # prime the loop

                # print 'converging'
                a = 0
                err = self.invkine.get_error()
                while (err > self.epsilon):
                    a = a + 1

                    # get delta theta
                    d_theta = self.invkine.get_delta_theta_dls(lam=self.lam)
                    # d_theta = self.invkine.get_delta_theta_trans()

                    # update controllers
                    self.world_def.set_controllers(d_theta)

                    # step
                    self.world_def.step(1.0 / FPS, 10, 10)

                    # update current configuration
                    cur_theta = [c.theta for c in self.world_def.get_rel_config()[1:]]  # ignore virtual base link
                    new_conf = generate_valid_config(cur_theta[0], cur_theta[1], cur_theta[2])
                    self.chain = KinematicChain(new_conf)

                    # update inverse kine
                    self.invkine.set_current_config(self.chain)

                    # update error
                    err = self.invkine.get_error()

                # print 'converged in {} iterations'.format(a)
                super(ArmLockEnv, self).render()
                # converged on that waypoint

        # if action:
        #     print 'take action'
        #     # update arm kinematic model
        #     c = self.world_def.get_rel_config()
        #     # exit()
        #
        #     joint_config = [{'name': '0-0'},
        #                     {'name': '0+1-', 'theta': c[1].theta, 'screw': [0, 0, 0, 0, 0, 1]},
        #                     {'name': '1-1+', 'x': 5},
        #                     {'name': '1+2-', 'theta': c[2].theta, 'screw': [0, 0, 0, 0, 0, 1]},
        #                     {'name': '2-2+', 'x': 5},
        #                     {'name': '2+3-', 'theta': c[3].theta, 'screw': [0, 0, 0, 0, 0, 1]},
        #                     {'name': '3-3+', 'x': 5}]
        #
        #     new_chain = KinematicChain(joint_config)
        #
        #     # update target kinematic model
        #     target_config = action
        #     self.target = KinematicChain(target_config)
        #     for link in self.target.chain:
        #         print link.get_transform()
        #     print 'total'
        #     print self.target.get_transform()
        #
        #     # update inverse kinematics model
        #     self.invkine.set_current_config(new_chain)
        #     self.invkine.set_target(self.target)
        #
        #     # update PID controllers
        #     delta_theta = self.invkine.get_delta_theta()
        #     self.world_def.set_controllers(delta_theta)
        #     print 'action taken'

        self.world_def.step(1.0 / FPS, 10, 10)
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

