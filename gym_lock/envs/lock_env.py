import gym
logger = logging.getLogger(__name__)

import numpy as np

from gym import error
from gym.utils import closer

class LockEnv(gym.Env):

  def __new__(cls, *args, **kwargs):
      # We use __new__ since we want the env author to be able to
      # override __init__ without remembering to call super.
      env = super(Env, cls).__new__(cls)
      env._env_closer_id = env_closer.register(env)
      env._closed = False
      env._spec = None

      # Will be automatically set when creating an environment via 'make'
      return env

  # Set this in SOME subclasses
  metadata = {'render.modes': []}
  reward_range = (-np.inf, np.inf)

  # Override in SOME subclasses
  def _close(self):
      pass

  # Set these in ALL subclasses
  action_space = None
  observation_space = None

  # Override in ALL subclasses
  def _step(self, action): raise NotImplementedError
  def _reset(self): raise NotImplementedError
  def _render(self, mode='human', close=False): return
  def _seed(self, seed=None): return []

  # Do not override
  _owns_render = True

  def step(self, action):
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
      return self._step(action)

  def reset(self):
      """Resets the state of the environment and returns an initial observation.

      Returns: observation (object): the initial observation of the
          space.
      """
      return self._reset()

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
      if not close: # then we have to check rendering mode
          modes = self.metadata.get('render.modes', [])
          if len(modes) == 0:
              raise error.UnsupportedMode('{} does not support rendering (requested mode: {})'.format(self, mode))
          elif mode not in modes:
              raise error.UnsupportedMode('Unsupported rendering mode: {}. (Supported modes for {}: {})'.format(mode, self, modes))
      return self._render(mode=mode, close=close)

  def close(self):
      """Override _close in your subclass to perform any necessary cleanup.

      Environments will automatically close() themselves when
      garbage collected or when the program exits.
      """
      # _closed will be missing if this instance is still
      # initializing.
      if not hasattr(self, '_closed') or self._closed:
          return

      if self._owns_render:
          self.render(close=True)

      self._close()
      env_closer.unregister(self._env_closer_id)
      # If an error occurs before this line, it's possible to
      # end up with double close.
      self._closed = True

  def seed(self, seed=None):
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
      return self._seed(seed)



