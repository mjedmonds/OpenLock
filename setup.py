from setuptools import setup

setup(name='gym_lock',
      version='0.0.1',
      # Box2D must be installed from source, and is not listed here
      install_requires=['gym', 'numpy', 'pyglet', 'pygraphviz', 'pymdptoolbox', 'transitions', 'shapely', 'jsonpickle', 'keras', 'h5py', 'matplotlib']
      )
