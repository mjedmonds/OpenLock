import gym

import numpy as np

import Box2D as b2

from gym import error, spaces
from gym.utils import closer, seeding
from gym.envs.classic_control import rendering

from gym_lock.envs.arm_world_def import ArmWorldDef

VIEWPORT_W = 600
VIEWPORT_H = 400
SCALE  = 30.0   # affects how fast-paced the game is, forces should be adjusted as well
FPS = 30


class ArmEnv(gym.Env):

    # Set this in SOME subclasses
    metadata = {'render.modes': ['human']} #TODO what does this do?


    ## Override in SOME subclasses
    #def _close(self):
    #        pass

    # Set these in ALL subclasses

    def __init__(self):


        self.action_space = spaces.Discrete(5) # up, down, left, right 
        self.observation_space = spaces.Box(-np.inf, np.inf, [4]) # [x, y, vx, vy]
        self.reward_range = (-np.inf, np.inf)
        self._seed()
        self.viewer = None
        self.world_def = ArmWorldDef()



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
        self.world_def.step(1.0/FPS, 10, 10)
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
                    trans = rendering.Transform(translation=t*fixture.shape.pos)
                    self.viewer.draw_circle(fixture.shape.radius).add_attr(trans)
                elif isinstance(fixture.shape, b2.b2PolygonShape):
                    vertices = [fixture.body.transform * v for v in fixture.shape.vertices]
                    self.viewer.draw_polygon(vertices, filled=False)



        return self.viewer.render(return_rgb_array = mode=='rgb_array')


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



