from collections import namedtuple

import numpy as np

from shapely.geometry import Polygon, Point
from matplotlib import pyplot as plt
from Box2D import *

ENTITY_STATES = {
    'LEVER_PUSHED': 0,
    'LEVER_PULLED': 1,
    'DOOR_UNLOCKED': 0,
    'DOOR_LOCKED': 1,
    'DOOR_CLOSED': 0,
    'DOOR_OPENED': 1,
    'LEVER_ACTIVE': 1,
    'LEVER_INACTIVE': 0
}


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
LeverConfig = namedtuple('lever_config', 'TwoDConfig LeverRole opt_params')    # role should be an enum indicating which lever this

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


class Action:
    def __init__(self, name, obj, params):
        self.name = name
        self.obj = obj
        self.params = params

    def __str__(self):
        return self.name + '_' + self.obj


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


class Object:
    def __init__(self, name, fixture=None, joint=None, color=None, int_test=None, ext_test=None):
        self.fixture = fixture
        self.joint = joint

        self.int_test = int_test
        self.ext_test = ext_test

        self.name = name
        self.color = color


class Lever(Object):
    def __init__(self, name, config, color, opt_params=None):
        Object.__init__(self, name)

        # # new
        # self.int_test = lambda joint: ENTITY_STATES['LEVER_PULLED'] if joint.translation < (joint.upperLimit + joint.lowerLimit) / 2.0 else ENTITY_STATES['LEVER_PUSHED']
        # self.ext_test = lambda joint: ENTITY_STATES['LEVER_PULLED'] if joint.translation > (joint.upperLimit + joint.lowerLimit) / 2.0 else ENTITY_STATES['LEVER_PUSHED']
        # # old
        # self.int_test = lambda joint: joint.translation < (joint.upperLimit + joint.lowerLimit) / 2.0
        # self.ext_test = lambda joint: joint.translation > (joint.upperLimit + joint.lowerLimit) / 2.0
        self.int_test = self.int_test_wrapper
        self.ext_test = self.ext_test_wrapper

        self.inner_clickable = None
        self.outer_clickable = None

        self.color = color
        self.config = config
        self.opt_params = opt_params

    def int_test_old(self, joint):
        return joint.translation < (joint.upperLimit + joint.lowerLimit) / 2.0

    def ext_test_old(self, joint):
        return joint.translation > (joint.upperLimit + joint.lowerLimit) / 2.0

    def int_test_new(self, joint):
        if joint.translation < (joint.upperLimit + joint.lowerLimit) / 2.0:
            return ENTITY_STATES['LEVER_PULLED']
        else:
            return ENTITY_STATES['LEVER_PUSHED']

    def ext_test_new(self, joint):
        if joint.translation > (joint.upperLimit + joint.lowerLimit) / 2.0:
            return ENTITY_STATES['LEVER_PULLED']
        else:
            return ENTITY_STATES['LEVER_PUSHED']

    def ext_test_wrapper(self, joint):
        assert(self.ext_test_old(joint) == self.ext_test_new(joint))
        return self.ext_test_new(joint)

    def int_test_wrapper(self, joint):
        assert(self.int_test_old(joint) == self.int_test_new(joint))
        return self.int_test_new(joint)

    def __str__(self):
        return self.name

    def __repr__(self):
        return str(self)

    def create_lever(self, world_def, config, width=0.5, length=5, lower_lim=-2, upper_lim=0):
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

        # passing userData sets the color of the lever to the be same as the object
        # used to set the color in box2drenderer
        lever_body = world_def.world.CreateDynamicBody(
            position=(x, y),
            angle=theta,
            angularDamping=0.8,
            linearDamping=0.8,
            userData=self
        )

        lever_body.gravityScale = 0

        lever_fixture = lever_body.CreateFixture(fixture_def)

        joint_axis = (-np.sin(theta), np.cos(theta))
        lever_joint = world_def.world.CreatePrismaticJoint(
            bodyA=lever_fixture.body,
            bodyB=world_def.ground,
            # anchor=(0, 0),
            anchor=lever_fixture.body.position,
            # localAnchorA=lever.body.position,
            # localAnchorB=self.ground.position,
            axis=joint_axis,
            lowerTranslation=lower_lim,
            upperTranslation=upper_lim,
            enableLimit=True,
            motorSpeed=0,
            maxMotorForce=abs(b2Dot(lever_body.massData.mass * world_def.world.gravity, b2Vec2(joint_axis))),
            enableMotor=True,
            userData={'plot_padding': width,
                      'joint_axis': joint_axis,
                      'obj_type': 'lever_joint'},
        )

        # create lever track in background
        xf1, xf2 = lever_fixture.body.transform, world_def.ground.transform
        x1, x2 = xf1.position, xf2.position
        p1, p2 = lever_joint.anchorA, lever_joint.anchorB
        padding = width
        width = 0.5

        # plot the bounds in which body A's anchor point can move relative to B
        local_axis = lever_body.GetLocalVector(joint_axis)
        world_axis = lever_body.GetWorldVector(local_axis)
        lower_lim, upper_lim = lever_joint.limits
        middle_lim = (upper_lim + lower_lim) / 2.0
        end1 = -world_axis * (upper_lim + padding)
        middle = -world_axis * middle_lim
        end2 = -world_axis * (lower_lim - padding)
        norm = b2Vec2(-world_axis[1], world_axis[0])

        inner_vertices = [end1 + norm * width, end1 - norm * width, middle - norm * width, middle + norm * width]
        outer_vertices = [middle - norm * width, middle + norm * width, end2 - norm * width, end2 + norm * width]

        # passing userData makes the color of the track the same as the lever
        inner_lever_track_body = world_def.background.CreateStaticBody(position=p2,
                                                                      active=False,
                                                                      shapes=b2PolygonShape(vertices=inner_vertices),
                                                                      userData=self)

        # passing userData makes the color of the track the same as the lever
        outer_lever_track_body = world_def.background.CreateStaticBody(position=p2,
                                                                      active=False,
                                                                      shapes=b2PolygonShape(vertices=outer_vertices),
                                                                      userData=self)
        trans = b2Transform()
        trans.SetIdentity()

        self.inner_vertices = [inner_lever_track_body.GetWorldPoint(vertex) for vertex in
                               inner_lever_track_body.fixtures[0].shape.vertices]
        self.outer_vertices = [outer_lever_track_body.GetWorldPoint(vertex) for vertex in
                               outer_lever_track_body.fixtures[0].shape.vertices]
        self.inner_poly = Polygon(self.inner_vertices)
        self.outer_poly = Polygon(self.outer_vertices)

        self.fixture = lever_fixture
        self.joint = lever_joint

        return lever_fixture, lever_joint, outer_lever_track_body, inner_lever_track_body

    # step is world_def step function
    def create_clickable(self, step, action_map):
        push = 'push_{}'.format(self.name)
        pull = 'pull_{}'.format(self.name)

        self.inner_clickable = Clickable(lambda xy, poly: poly.contains(Point(xy)), step,
                                         callback_args=[action_map[pull]], test_args=[self.inner_poly])
        self.outer_clickable = Clickable(lambda xy, poly: poly.contains(Point(xy)), step,
                                         callback_args=[action_map[push]], test_args=[self.outer_poly])

    def determine_active(self):
        if self.color == COLORS['active']:
            return ENTITY_STATES['LEVER_ACTIVE']
        elif self.color == COLORS['inactive']:
            return ENTITY_STATES['LEVER_INACTIVE']
        else:
            raise ValueError('Expected lever to be active or inactive, different color set')

class Door(Object):
    # def __init__(self, door_fixture, door_joint, int_test, ext_test, name):
    def __init__(self, world_def, name, config, color):
        # Object.__init__(self, name, door_fixture, joint=door_joint, int_test=int_test, ext_test=ext_test)
        Object.__init__(self, name)
        self.fixture, self.joint, self.lock = self._create_door(world_def, config)

        # old
        # open_test = lambda door_hinge: ENTITY_STATES['DOOR_MOVED'] if abs(door_hinge.angle) > np.pi / 16 else ENTITY_STATES['DOOR_MOVED']
        # old
        # open_test = lambda door_hinge: abs(door_hinge.angle) > np.pi / 16

        self.int_test = self.open_test
        self.ext_test = self.open_test

        self.color = color

    def open_test_old(self, door_hinge):
        return abs(door_hinge.angle) > np.pi / 16

    def open_test_new(self, door_hinge):
        if abs(door_hinge.angle) > np.pi / 16:
            return ENTITY_STATES['DOOR_OPENED']
        else:
            return ENTITY_STATES['DOOR_CLOSED']

    def open_test(self, door_hinge):
        assert(self.open_test_old(door_hinge) == self.open_test_new(door_hinge))
        return self.open_test_new(door_hinge)

    def lock_present(self):
        if self.lock is None:
            return ENTITY_STATES['DOOR_UNLOCKED']
        else:
            return ENTITY_STATES['DOOR_LOCKED']

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

        door_body.gravityScale = 0

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


def plot_rewards(rewards, epsilons, filename, width=12, height=6):
    plt.clf()
    assert len(epsilons) == len(rewards)
    moving_avg = compute_moving_average(rewards, 100)
    fig = plt.gcf()
    ax = plt.gca()
    fig.set_size_inches(width, height)
    plt.xlim((0, len(rewards)))
    r, = plt.plot(rewards, color='red', linestyle='-', linewidth=0.5, label='reward', alpha=0.5)
    ave_r, = plt.plot(moving_avg, color='blue', linestyle='-', linewidth=0.8, label='avg_reward')
    # e, = plt.plot(epsilons, color='blue', linestyle='--', alpha=0.5, label='epsilon')
    plt.legend([r, ave_r], ['reward', 'average reward'])
    plt.ylabel('Reward')
    plt.xlabel('Episode #')
    plt.savefig(filename)


def plot_rewards_trial_switch_points(rewards, epsilons, trial_switch_points, filename, plot_xticks=False):
    plt.clf()
    assert len(epsilons) == len(rewards)
    moving_avg = compute_moving_average(rewards, 100)
    fig = plt.gcf()
    ax = plt.gca()
    fig.set_size_inches(12, 6)
    plt.xlim((0, len(rewards)))
    # mark where the trials changed
    for trial_switch_point in trial_switch_points:
        plt.axvline(trial_switch_point, color='black', linewidth=0.5, linestyle='--', alpha=0.3)
    r, = plt.plot(rewards, color='red', linestyle='-', linewidth=0.5, label='reward', alpha=0.5)
    ave_r, = plt.plot(moving_avg, color='blue', linestyle='-', linewidth=0.8, label='avg_reward')
    # e, = plt.plot(epsilons, color='blue', linestyle='--', alpha=0.5, label='epsilon')
    plt.legend([r, ave_r], ['reward', 'average reward'])
    if plot_xticks:
        xtick_points, xtick_labels = create_xtick_labels(trial_switch_points)
        plt.xticks(xtick_points, xtick_labels)
        # vertical alignment of xtick labels
        va = [0 if x % 2 == 0 else -0.03 for x in range(len(xtick_points))]
        for t, y in zip(ax.get_xticklabels(), va):
            t.set_y(y)
    plt.ylabel('Reward')
    plt.xlabel('Episode # and trial #')
    plt.savefig(filename)


def compute_moving_average(rewards, window):
    cur_window_size = 1
    moving_average = []
    for i in range(len(rewards)-1):
        lower_idx = max(0, i-cur_window_size)
        average = sum(rewards[lower_idx:i+1]) / cur_window_size
        moving_average.append(average)
        cur_window_size += 1
        if cur_window_size > window:
            cur_window_size = window
    return moving_average


def create_xtick_labels(trial_switch_points):
    xtick_points = [0]
    xtick_labels = ['0']
    prev_switch_point = 0
    trial_count = 1
    for trial_switch_point in trial_switch_points:
        xtick_point = ((trial_switch_point - prev_switch_point) / 2) + prev_switch_point
        xtick_points.append(xtick_point)
        xtick_labels.append('trial ' + str(trial_count))
        xtick_points.append(trial_switch_point)
        xtick_labels.append(str(trial_switch_point))
        trial_count += 1
        prev_switch_point = trial_switch_point
    return xtick_points, xtick_labels


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
