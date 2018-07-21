#!/usr/bin/env python

from distutils.core import setup

setup(name='OpenLock',
      version='1.0',
      description='OpenLock OpenAI Gym Environment',
      author='Mark Edmonds',
      author_email='mark@mjedmonds.com',
      url='https://github.com/mjedmonds/OpenLock',
      packages=['openlock', 'openlock.scenarios', 'openlock.envs', 'openlock.envs.world_defs'],
     )