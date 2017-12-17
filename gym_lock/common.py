from collections import namedtuple

import numpy as np

from shapely.geometry import Polygon, Point
from Box2D import *


class LeverRole:
    inactive = 'inactive'
    l0 = 'l0'
    l1 = 'l1'
    l2 = 'l2'
    l3 = 'l3'
    l4 = 'l4'
    l5 = 'l5'
    l6 = 'l6'


TwoDConfig = namedtuple('Config', 'x y theta')
TwoDForce = namedtuple('Force', 'norm tan')
Action = namedtuple('action', 'name params') # params should be list-like or a single value
LeverConfig = namedtuple('lever_config', 'TwoDConfig LeverRole, opt_params')    # role should be an enum indicating which lever this

Color = namedtuple('Color', 'r g b')

COLORS = {
    'active': Color(0.6, 0.6, 0.6),
    'inactive': Color(0.9, 0.9, 0.9),
    'static': Color(0.5, 0.9, 0.5),
    'kinematic': Color(0.5, 0.5, 0.9),
    'asleep': Color(0.6, 0.6, 0.6),
    'reset_button': Color(0.8, 0.1, 0.23),
    'save_button': Color(0.5, 0.9, 0.5),
    'default': Color(0.9, 0.7, 0.7),
}


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
    def __init__(self, name, fixture=None, joint=None, color=None, int_test=None, ext_test=None):
        self.fixture = fixture
        self.joint = joint

        self.int_test = int_test
        self.ext_test = ext_test

        self.name = name
        self.color = color


class Lock(Object):
    def __init__(self, world_def, name, config, color, opt_params=None):
        Object.__init__(self, name)

        if opt_params:
            self.fixture, self.joint, self.outer_track, self.inner_track = self._create_lock(world_def, config, **opt_params)
        else:
            self.fixture, self.joint, self.outer_track, self.inner_track = self._create_lock(world_def, config)

        self.inner_vertices = [self.inner_track.GetWorldPoint(vertex) for vertex in self.inner_track.fixtures[0].shape.vertices]
        self.outer_vertices = [self.outer_track.GetWorldPoint(vertex) for vertex in self.outer_track.fixtures[0].shape.vertices]
        self.inner_poly = Polygon(self.inner_vertices)
        self.outer_poly = Polygon(self.outer_vertices)

        self.int_test = lambda joint: joint.translation < (joint.upperLimit + joint.lowerLimit) / 2.0
        self.ext_test = lambda joint: joint.translation > (joint.upperLimit + joint.lowerLimit) / 2.0

        self.inner_clickable = None
        self.outer_clickable = None

        self.color = color

    def _create_lock(self, world_def, config, width=0.5, length=5, lower_lim=-2, upper_lim=0):
        x, y, theta = config

        fixture_def = b2FixtureDef(
            shape=b2PolygonShape(vertices=[(-length, -width),
                                           (-length, width),
                                           (length, width),
                                           (length, -width)]),
            density=1,
            friction=1.0,
            categoryBits=0x0010,
            maskBits=0x1101
        )

        # passing userData sets the color of the lock to the be same as the object
        # used to set the color in box2drenderer
        lock_body = world_def.world.CreateDynamicBody(
            position=(x, y),
            angle=theta,
            angularDamping=0.8,
            linearDamping=0.8,
            userData=self
        )

        lock_fixture = lock_body.CreateFixture(fixture_def)

        joint_axis = (-np.sin(theta), np.cos(theta))
        lock_joint = world_def.world.CreatePrismaticJoint(
            bodyA=lock_fixture.body,
            bodyB=world_def.ground,
            # anchor=(0, 0),
            anchor=lock_fixture.body.position,
            # localAnchorA=lock.body.position,
            # localAnchorB=self.ground.position,
            axis=joint_axis,
            lowerTranslation=lower_lim,
            upperTranslation=upper_lim,
            enableLimit=True,
            motorSpeed=0,
            maxMotorForce=abs(b2Dot(lock_body.massData.mass * world_def.world.gravity, b2Vec2(joint_axis))),
            enableMotor=True,
            userData={'plot_padding': width,
                      'joint_axis': joint_axis,
                      'obj_type': 'lock_joint'},
        )

        # create lock track in background
        xf1, xf2 = lock_fixture.body.transform, world_def.ground.transform
        x1, x2 = xf1.position, xf2.position
        p1, p2 = lock_joint.anchorA, lock_joint.anchorB
        padding = width
        width = 0.5

        # plot the bounds in which body A's anchor point can move relative to B
        local_axis = lock_body.GetLocalVector(joint_axis)
        world_axis = lock_body.GetWorldVector(local_axis)
        lower_lim, upper_lim = lock_joint.limits
        middle_lim = (upper_lim + lower_lim) / 2.0
        end1 = -world_axis * (upper_lim + padding)
        middle = -world_axis * middle_lim
        end2 = -world_axis * (lower_lim - padding)
        norm = b2Vec2(-world_axis[1], world_axis[0])

        inner_vertices = [end1 + norm * width, end1 - norm * width, middle - norm * width, middle + norm * width]
        outer_vertices = [middle - norm * width, middle + norm * width, end2 - norm * width, end2 + norm * width]

        # passing userData makes the color of the track the same as the lever
        inner_lock_track_body = world_def.background.CreateStaticBody(position=p2,
                                                                      active=False,
                                                                      shapes=b2PolygonShape(vertices=inner_vertices),
                                                                      userData=self)

        # passing userData makes the color of the track the same as the lever
        outer_lock_track_body = world_def.background.CreateStaticBody(position=p2,
                                                                      active=False,
                                                                      shapes=b2PolygonShape(vertices=outer_vertices),
                                                                      userData=self)
        trans = b2Transform()
        trans.SetIdentity()

        return lock_fixture, lock_joint, outer_lock_track_body, inner_lock_track_body

    # step is world_def step function
    def create_clickable(self, step, action_map):
        push = 'push_{}'.format(self.name)
        pull = 'pull_{}'.format(self.name)

        self.inner_clickable = Clickable(lambda xy, poly: poly.contains(Point(xy)), step,
                                         callback_args=[action_map[pull]], test_args=[self.inner_poly])
        self.outer_clickable = Clickable(lambda xy, poly: poly.contains(Point(xy)), step,
                                         callback_args=[action_map[push]], test_args=[self.outer_poly])


class Door(Object):
    # def __init__(self, door_fixture, door_joint, int_test, ext_test, name):
    def __init__(self, world_def, name, config, color):
        # Object.__init__(self, name, door_fixture, joint=door_joint, int_test=int_test, ext_test=ext_test)
        Object.__init__(self, name)
        self.fixture, self.joint, self.lock = self._create_door(world_def, config)

        # Register door components with ENV (TODO: can this be removed?)
        world_def.door = self.fixture
        world_def.door_hinge = self.joint
        world_def.door_lock = self.lock

        open_test = lambda door_hinge: abs(door_hinge.angle) > np.pi / 16
        self.int_test = open_test
        self.ext_test = open_test

        self.color = color

    def _create_door(self, world_def, config, width=0.5, length=10, locked=True):
        # TODO: add relocking ability
        # create door
        x, y, theta = config

        fixture_def = b2FixtureDef(
            shape=b2PolygonShape(vertices=[(0, -width),
                                           (0, width),
                                           (length, width),
                                           (length, -width)]),
            density=1,
            friction=1.0,
            categoryBits=0x0010,
            maskBits=0x1101)

        door_body = world_def.world.CreateDynamicBody(
            position=(x, y),
            angle=theta,
            angularDamping=0.8,
            linearDamping=0.8,
            userData=self
        )

        door_fixture = door_body.CreateFixture(fixture_def)

        door_hinge = world_def.world.CreateRevoluteJoint(
            bodyA=door_fixture.body,  # end of link A
            bodyB=world_def.ground,  # beginning of link B
            localAnchorA=(0, 0),
            localAnchorB=(x, y),
            enableMotor=True,
            motorSpeed=0,
            enableLimit=False,
            maxMotorTorque=500
        )

        door_lock = None
        if locked:
            delta_x = np.cos(theta) * length
            delta_y = np.sin(theta) * length
            door_lock = world_def.world.CreateWeldJoint(
                bodyA=door_fixture.body,  # end of link A
                bodyB=world_def.ground,  # beginning of link B
                localAnchorB=(x + delta_x, y + delta_y)
            )

        return door_fixture, door_hinge, door_lock


class Button(Object):
    def __init__(self, world_def, config, color, name, height, width, x_offset=0, y_offset=0, clickable=None):
        Object.__init__(self, name)
        x, y, theta = config
        button = world_def.world.CreateStaticBody(
            position=(x + x_offset, y + y_offset),
            angle=theta,
            shapes=b2PolygonShape(box=(height, width)),
            userData=self
        )
        self.fixture = button.fixtures[0]
        self.color = color
        self.clickable = None

    def create_clickable(self, step, action_map, callback_args):
        vertices = [self.fixture.body.GetWorldPoint(vertex) for vertex in self.fixture.shape.vertices]
        poly = Polygon(vertices)
        self.clickable = Clickable(lambda xy, poly: poly.contains(Point(xy)), step, callback_args=[callback_args], test_args=[poly])


def wrapToMinusPiToPi(original):
    return (original + np.pi) % (2 * np.pi) - np.pi


def transform_to_theta(transform):
    return np.arccos(transform[0, 0]) * np.sign(np.arcsin(transform[1, 0]))


def clamp_mag(array_like, clamp_mag):
    for i in range(0, len(array_like)):
        if abs(array_like[i]) > clamp_mag:
            array_like[i] = clamp_mag * np.sign(array_like[i])
    return array_like


def print_instructions():
    print 'Hello! Welcome to the game!'

    # time.sleep(1)

    # time.sleep(1)
    print '''See that door on your right? It is the vertical vertical on your right, with the
             red circle (the door hinge) and black circle (it's lock). That is your only escape.'''
    # time.sleep(1)
    print    '''To open it, you must manipulate the three locks (the rectangles above, below, and
             to your left). Their behavior is unknown! You'll know that you unlocked the door
             when the black circle goes away'''
    # time.sleep(1)
    print 'ready...'
    # time.sleep(1)
    print 'set...'
    # time.sleep(1)
    print 'go!'
