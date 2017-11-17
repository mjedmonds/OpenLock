from collections import namedtuple

import numpy as np

from shapely.geometry import Polygon, Point

TwoDConfig = namedtuple('Config', 'x y theta')
TwoDForce = namedtuple('Force', 'norm tan')
Action = namedtuple('action', 'name params') # params should be list-like or a single value

Color = namedtuple('Color', 'r g b')

class Clickable(object):

    def __init__(self, test, callback, callback_args=[], test_args=[]):
        self.test = test
        self.callback = callback
        self.callback_args = callback_args
        self.test_args = test_args

    def test_region(self, world_xy):
        return self.test(world_xy, *self.test_args)

    def call(self):
        return self.callback(*self.callback_args)

class Object():
    def __init__(self, fixture, joint, int_test, ext_test, name):
        self.fixture = fixture
        self.joint = joint

        self.int_test = int_test
        self.ext_test = ext_test

        self.name = name

class Lock(Object):
    # step is env step function
    def __init__(self, lock_fixture, joint, int_test, ext_test, outer_track, inner_track, name):
        Object.__init__(self, lock_fixture, joint, int_test, ext_test, name)

        self.outer_track = outer_track
        self.inner_track = inner_track

        self.inner_vertices = [self.inner_track.GetWorldPoint(vertex) for vertex in self.inner_track.fixtures[0].shape.vertices]
        self.outer_vertices = [self.outer_track.GetWorldPoint(vertex) for vertex in self.outer_track.fixtures[0].shape.vertices]
        self.inner_poly = Polygon(self.inner_vertices)
        self.outer_poly = Polygon(self.outer_vertices)

        self.inner_clickable = None
        self.outer_clickable = None



    def create_clickable(self, step, action_map):
        push = 'push_perp_{}'.format(self.name)
        pull = 'pull_perp_{}'.format(self.name)

        self.inner_clickable = Clickable(lambda xy, poly: poly.contains(Point(xy)), step,
                                         callback_args=[action_map[pull]], test_args=[self.inner_poly])
        self.outer_clickable = Clickable(lambda xy, poly: poly.contains(Point(xy)), step,
                                         callback_args=[action_map[push]], test_args=[self.outer_poly])


class Door(Object):
    def __init__(self, door_fixture, door_joint, int_test, ext_test, name):
        Object.__init__(self, door_fixture, door_joint, int_test, ext_test, name)


def wrapToMinusPiToPi(original):
    return (original + np.pi) % (2 * np.pi) - np.pi


def transform_to_theta(transform):
    return np.arccos(transform[0, 0]) * np.sign(np.arcsin(transform[1, 0]))


def clamp_mag(array_like, clamp_mag):
    for i in range(0, len(array_like)):
        if abs(array_like[i]) > clamp_mag:
            array_like[i] = clamp_mag * np.sign(array_like[i])
    return array_like
