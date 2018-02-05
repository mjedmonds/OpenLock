from setuptools import setup

setup(name='gym_lock',
      version='0.0.1',
      # Box2D must be installed from source
      install_requires=['gym', 'numpy', 'pyglet', 'Box2D', 'pygraphviz', 'pymdptoolbox', 'transitions', 'shapely', 'jsonpickle', 'keras', 'tensorflow', 'h5py', 'matplotlib']
      )
