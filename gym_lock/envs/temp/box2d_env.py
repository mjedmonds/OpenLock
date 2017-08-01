import gym

import numpy as np

import Box2D as b2

from gym import error, spaces
from gym.utils import closer, seeding
from gym.envs.classic_control import rendering

FPS = 30


class LockEnv(gym.Env):
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

        # setup Box2D world
        self.world = b2.b2World(gravity=(0, -10), doSleep=True)
        self.world.gravity = (0.0, 0.0)
        self.initial_position = (0, 5)
        self.initial_angle = b2.b2_pi
        self.initial_linear_velocity = (0., 0.)  # (x0[2], x0[3])
        self.initial_target_obj_position = (5., 20.)  # (x0[4], x0[5])
        self.initial_target_obj_linear_velocity = (0., 0.)  # (x0[6], x0[7]),
        self.initial_angular_velocity = 0
        self.target_initial = (0, 0)

        ground = self.world.CreateBody(position=(0, 0))
        ground.CreateEdgeChain(
            [(-20, -20),
             (-20, 20),
             (20, 20),
             (20, -20),
             (-20, -20)]
        )

        xf1 = b2.b2Transform()
        xf1.angle = 0.3524 * b2.b2_pi
        xf1.position = b2.b2Mul(xf1.R, (1.0, 0.0))

        xf2 = b2.b2Transform()
        xf2.angle = -0.3524 * b2.b2_pi
        xf2.position = b2.b2Mul(xf2.R, (-1.0, 0.0))
        self.body = self.world.CreateDynamicBody(
            position=self.initial_position,
            angle=self.initial_angle,
            linearVelocity=self.initial_linear_velocity,
            angularVelocity=self.initial_angular_velocity,
            angularDamping=5,
            linearDamping=0.1,
            shapes=[b2.b2CircleShape(radius=1)],
            shapeFixture=b2.b2FixtureDef(density=1.0),
        )

        # Create piston

        # create piston head
        self.target_obj = self.world.CreateDynamicBody(
            position=(0, 0),
            angle=0,
            linearVelocity=self.initial_linear_velocity,
            angularVelocity=0,
            angularDamping=5,
            linearDamping=0.3,
            shapes=[b2.b2PolygonShape(box=(2, 1))],
            shapeFixture=b2.b2FixtureDef(density=1.0),
        )

        # piston head desired location
        self.target = self.world.CreateStaticBody(
            position=(0, -5),
            angle=0,
            shapes=[b2.b2PolygonShape(box=(2, 1))],
        )
        self.target.active = False
        # ghost body which IS active for piston
        self.target_ghost = self.world.CreateStaticBody(
            position=(0, -5),
        )

        # piston constraining walls
        eta = 0.1
        self.piston_walls = self.world.CreateBody(position=(0, 0))
        ground.CreateEdgeChain(
            [(-3, 1), (-3, -10), (3, -10), (3, 1), (2 + eta, 1), (2 + eta, -9), (-2 - eta, -9), (-2 - eta, 1), (-3, 1)])

        # constrain piston movement
        self.world.CreatePrismaticJoint(
            bodyA=self.target_obj,
            bodyB=self.target_ghost,
            anchor=(self.target.position + self.target_obj.position) / 2,
            axis=(0, 1),
            lowerTranslation=0.0,
            upperTranslation=5.0,
            enableLimit=True,
            # motorForce=1.0,
            motorSpeed=0.0,
            enableMotor=True,
        )
        # attach spring to piston head
        self.world.CreateDistanceJoint(
            bodyA=self.target_obj,
            bodyB=self.target_ghost,
            anchorA=self.target_obj.position,
            anchorB=self.target_ghost.position,
            frequencyHz=0.5,
            dampingRatio=0.8,
            collideConnected=True)

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
        self.body.ApplyForce(force=action, point=self.body.position, wake=True)
        self.world.Step(1.0 / FPS, 10, 10)

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
        for body in self.world:
            for fixture in body.fixtures:
                if isinstance(fixture, b2.b2EdgeShape)
                    rendering.Viewer.draw_line}

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



