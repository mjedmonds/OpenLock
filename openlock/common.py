from collections import namedtuple
import copy

import numpy as np

from shapely.geometry import Polygon, Point
from Box2D import *


ENTITY_STATES = {
    "LEVER_PUSHED": 0,
    "LEVER_PULLED": 1,
    "DOOR_UNLOCKED": 0,
    "DOOR_LOCKED": 1,
    "DOOR_CLOSED": 0,
    "DOOR_OPENED": 1,
    "LEVER_ACTIVE": 1,
    "LEVER_INACTIVE": 0,
}

MAX_FORCE = 100000

COLOR_LABELS = ["GREY", "WHITE"]

DOOR_WIDTH = 0.5
DOOR_LENGTH = 10

TwoDConfig = namedtuple("Config", "x y theta")
TwoDForce = namedtuple("Force", "norm tan")

LOCK_REGEX_STR = "^l[0-9]+"
INACTIVE_LOCK_REGEX_STR = "^inactive[0-9]+$"


class LeverRoleEnum:
    inactive = "inactive"
    l0 = "l0"
    l1 = "l1"
    l2 = "l2"
    l3 = "l3"
    l4 = "l4"
    l5 = "l5"
    l6 = "l6"


class ObjectPosition:
    def __init__(self, twodconfig, name):
        self.config = twodconfig
        self.name = name

    def __str__(self):
        return self.name + ": " + str(self.config)

    def __repr__(self):
        return str(self)


class ObjectPositionEnum:
    UPPER = ObjectPosition(TwoDConfig(0, 15, 0), "UPPER")
    LEFT = ObjectPosition(TwoDConfig(-15, 0, np.pi / 2), "LEFT")
    LOWER = ObjectPosition(TwoDConfig(0, -15, -np.pi), "LOWER")
    UPPERLEFT = ObjectPosition(TwoDConfig(-11, 11, np.pi / 4), "UPPERLEFT")
    UPPERRIGHT = ObjectPosition(TwoDConfig(11, 11, -np.pi / 4), "UPPERRIGHT")
    LOWERLEFT = ObjectPosition(TwoDConfig(-11, -11, 3 * np.pi / 4), "LOWERLEFT")
    LOWERRIGHT = ObjectPosition(TwoDConfig(11, -11, 5 * np.pi / 4), "LOWERRIGHT")
    RIGHT = ObjectPosition(TwoDConfig(15, 0, -np.pi / 2), "RIGHT")
    DOOR = ObjectPosition(TwoDConfig(15, 0, -np.pi / 2), "door")
    DOOR_LOCK = ObjectPosition(
        TwoDConfig(DOOR.config[0], DOOR.config[1] + DOOR_LENGTH / 2, DOOR.config[2]),
        "door_lock",
    )


LeverConfig = namedtuple(
    "lever_config", "LeverPosition LeverRoleEnum opt_params"
)  # role should be an enum indicating which lever this

Color = namedtuple("Color", "r g b")


GREY = Color(0.6, 0.6, 0.6)
GREEN = Color(0.5, 0.9, 0.5)
PURPLE = Color(0.5, 0.5, 0.9)
RED = Color(1.0, 0, 0)
DARK_GREY = Color(0.35, 0.35, 0.35)
BLUE = Color(0.0, 0.0, 1.0)
BLACK = Color(0, 0, 0)
WHITE = Color(0.9, 0.9, 0.9)
PINK = Color(0.8, 0.1, 0.23)
LIGHT_PINK = Color(0.9, 0.7, 0.7)

COLOR_NAME_TO_COLOR = {
    "active": GREY,
    "inactive": WHITE,
    "static": GREEN,
    "kinematic": PURPLE,
    "asleep": GREY,
    "default": GREY,
    "rev_joint": RED,
    "pris_joint": DARK_GREY,
    "dist_joint": BLUE,
    "weld_joint": BLACK,
    "reset_button": PINK,
    "save_button": GREEN,
}

COLORS = COLOR_NAME_TO_COLOR

COLOR_TO_COLOR_NAME = {
    Color(0.6, 0.6, 0.6): "GREY",
    Color(0.5, 0.9, 0.5): "GREEN",
    Color(0.5, 0.5, 0.9): "PURPLE",
    Color(1.0, 0, 0): "RED",
    Color(0.35, 0.35, 0.35): "DARK_GREY",
    Color(0.0, 0.0, 1.0): "BLUE",
    Color(0, 0, 0): "BLACK",
    Color(0.9, 0.9, 0.9): "WHITE",
}

def generate_effect_probabilities(l0=1.0, l1=1.0, l2=1.0, l3=1.0, door=1.0, others=0.0):
    return {
        "l0": l0,
        "l1": l1,
        "l2": l2,
        "l3": l3,
        "door": door,
        "others": others
    }


def assign_effect_probabilities(obj_name, effect_probabilities):
    if obj_name in effect_probabilities.keys():
        effect_probability = effect_probabilities[obj_name]
    else:
        effect_probability = effect_probabilities["others"]
    return effect_probability

class Action:
    def __init__(self, name, obj, params):
        self.name = name
        self.obj = obj
        self.params = params

    def __str__(self):
        return self.name + "_" + self.obj

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return self.name == other.name and self.obj == other.obj

    def eq_str(self, other_str):
        return str(self) == other_str

    def eq_action_log(self, other_action_log):
        return str(self) == other_action_log.name

    def __hash__(self):
        return hash(str(self) + str(self.params))

    @staticmethod
    def make_action_from_str(action_str):
        name, obj = action_str.split("_")
        return Action(name, obj, None)


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
    def __init__(
        self,
        name,
        position=None,
        fixture=None,
        joint=None,
        color=None,
        int_test=None,
        ext_test=None,
        effect_probability=1.0
    ):
        self.fixture = fixture
        self.joint = joint

        self.int_test = int_test
        self.ext_test = ext_test

        self.name = name
        self.position = position
        self.color = color
        self.effect_probability = effect_probability


class Lever(Object):
    def __init__(self, name, position, color, opt_params=None, effect_probability=1.0):
        Object.__init__(self, name, effect_probability=effect_probability)

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
        self.position = position
        self.opt_params = opt_params
        self.locked = False
        self.gravity = None

    @property
    def in_physics_simulator(self):
        return self.joint is not None

    def int_test_old(self, joint):
        return joint.translation < (joint.upperLimit + joint.lowerLimit) / 2.0

    def ext_test_old(self, joint):
        return joint.translation > (joint.upperLimit + joint.lowerLimit) / 2.0

    def int_test_new(self, joint):
        if joint.translation < (joint.upperLimit + joint.lowerLimit) / 2.0:
            return ENTITY_STATES["LEVER_PULLED"]
        else:
            return ENTITY_STATES["LEVER_PUSHED"]

    def ext_test_new(self, joint):
        if joint.translation > (joint.upperLimit + joint.lowerLimit) / 2.0:
            return ENTITY_STATES["LEVER_PULLED"]
        else:
            return ENTITY_STATES["LEVER_PUSHED"]

    def ext_test_wrapper(self, joint):
        assert self.ext_test_old(joint) == self.ext_test_new(joint)
        return self.ext_test_new(joint)

    def int_test_wrapper(self, joint):
        assert self.int_test_old(joint) == self.int_test_new(joint)
        return self.int_test_new(joint)

    def lock(self):
        if self.in_physics_simulator:
            self.joint.maxMotorForce = MAX_FORCE
        self.locked = True

    def unlock(self):
        if self.in_physics_simulator:
            lock = self.fixture
            joint_axis = (-np.sin(lock.body.angle), np.cos(lock.body.angle))
            self.joint.maxMotorForce = abs(
                b2Dot(lock.body.massData.mass * self.gravity, b2Vec2(joint_axis))
            )
        self.locked = False

    def __str__(self):
        return self.name

    def __repr__(self):
        return str(self)

    def create_lever(
        self, world_def, position, width=0.5, length=5, lower_lim=-2, upper_lim=0
    ):
        x, y, theta = position.config
        self.gravity = world_def.world.gravity

        fixture_def = b2FixtureDef(
            shape=b2PolygonShape(
                vertices=[
                    (-length, -width),
                    (-length, width),
                    (length, width),
                    (length, -width),
                ]
            ),
            density=1,
            friction=1.0,
            categoryBits=0x0010,
            maskBits=0x1101,
        )

        # passing userData sets the color of the lever to the be same as the object
        # used to set the color in box2drenderer
        lever_body = world_def.world.CreateDynamicBody(
            position=(x, y),
            angle=theta,
            angularDamping=0.8,
            linearDamping=0.8,
            userData=self,
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
            maxMotorForce=abs(
                b2Dot(
                    lever_body.massData.mass * world_def.world.gravity,
                    b2Vec2(joint_axis),
                )
            ),
            enableMotor=True,
            userData={
                "plot_padding": width,
                "joint_axis": joint_axis,
                "obj_type": "lever_joint",
            },
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

        inner_vertices = [
            end1 + norm * width,
            end1 - norm * width,
            middle - norm * width,
            middle + norm * width,
        ]
        outer_vertices = [
            middle - norm * width,
            middle + norm * width,
            end2 - norm * width,
            end2 + norm * width,
        ]

        # passing userData makes the color of the track the same as the lever
        inner_lever_track_body = world_def.background.CreateStaticBody(
            position=p2,
            active=False,
            shapes=b2PolygonShape(vertices=inner_vertices),
            userData=self,
        )

        # passing userData makes the color of the track the same as the lever
        outer_lever_track_body = world_def.background.CreateStaticBody(
            position=p2,
            active=False,
            shapes=b2PolygonShape(vertices=outer_vertices),
            userData=self,
        )
        trans = b2Transform()
        trans.SetIdentity()

        self.inner_vertices = [
            inner_lever_track_body.GetWorldPoint(vertex)
            for vertex in inner_lever_track_body.fixtures[0].shape.vertices
        ]
        self.outer_vertices = [
            outer_lever_track_body.GetWorldPoint(vertex)
            for vertex in outer_lever_track_body.fixtures[0].shape.vertices
        ]
        self.inner_poly = Polygon(self.inner_vertices)
        self.outer_poly = Polygon(self.outer_vertices)

        self.fixture = lever_fixture
        self.joint = lever_joint

        return (
            lever_fixture,
            lever_joint,
            outer_lever_track_body,
            inner_lever_track_body,
        )

    # step is world_def step function
    def create_clickable(self, step):
        push = Action("push", self.name, 4)
        pull = Action("pull", self.name, 4)

        self.inner_clickable = Clickable(
            lambda xy, poly: poly.contains(Point(xy)),
            step,
            callback_args=[pull],
            test_args=[self.inner_poly],
        )
        self.outer_clickable = Clickable(
            lambda xy, poly: poly.contains(Point(xy)),
            step,
            callback_args=[push],
            test_args=[self.outer_poly],
        )

    def determine_active(self):
        if self.color == COLORS["active"]:
            return ENTITY_STATES["LEVER_ACTIVE"]
        elif self.color == COLORS["inactive"]:
            return ENTITY_STATES["LEVER_INACTIVE"]
        else:
            raise ValueError(
                "Expected lever to be active or inactive, different color set"
            )


class Door(Object):
    # def __init__(self, door_fixture, door_joint, int_test, ext_test, name):
    def __init__(
        self, world_def, name, position, color, width=0.5, length=10, locked=True, effect_probability=1.0
    ):
        # Object.__init__(self, name, door_fixture, joint=door_joint, int_test=int_test, ext_test=ext_test)
        Object.__init__(self, name, effect_probability=effect_probability)

        # need reference to world def to destroy black dot when door becomes unlocked
        self.world_def = world_def

        # create a modified position to move the door so it's centered at the specified position
        # todo: this doesn't correclty handle shifts in theta
        x, y, theta = position.config
        x = x + width / 2
        y = y + length / 2
        new_position = ObjectPosition(TwoDConfig(x, y, theta), position.name)

        if world_def is not None:
            self.fixture, self.joint = self._create_door(
                world_def, new_position, locked=locked
            )

            # old
            # open_test = lambda door_hinge: ENTITY_STATES['DOOR_MOVED'] if abs(door_hinge.angle) > np.pi / 16 else ENTITY_STATES['DOOR_MOVED']
            # old
            # open_test = lambda door_hinge: abs(door_hinge.angle) > np.pi / 16

            self.int_test = self.open_test
            self.ext_test = self.open_test

        self.door_lock = Lock("door_lock", locked)

        if locked:
            self.lock()

        self.color = color
        self.name = "door"
        self.position = position

    def open_test_old(self, door_hinge):
        return abs(door_hinge.angle) > np.pi / 16

    def open_test_new(self, door_hinge):
        if abs(door_hinge.angle) > np.pi / 16:
            return ENTITY_STATES["DOOR_OPENED"]
        else:
            return ENTITY_STATES["DOOR_CLOSED"]

    def open_test(self, door_hinge):
        assert self.open_test_old(door_hinge) == self.open_test_new(door_hinge)
        return self.open_test_new(door_hinge)

    @property
    def locked(self):
        return self.door_lock.locked

    def lock_state(self):
        if self.locked:
            return ENTITY_STATES["DOOR_LOCKED"]
        else:
            return ENTITY_STATES["DOOR_UNLOCKED"]

    def lock(self):
        if self.world_def is not None:
            theta = self.fixture.body.angle
            length = max([v[0] for v in self.fixture.shape.vertices])
            x, y = self.fixture.body.position

            delta_x = np.cos(theta) * length
            delta_y = np.sin(theta) * length

            self.door_lock.lock(self.world_def, self.fixture.body, x, y, delta_x, delta_y)
        else:
            self.door_lock.lock()

    def unlock(self):
        self.door_lock.unlock(self.world_def)

    def _create_door(self, world_def, position, width=0.5, length=10, locked=True):
        # create door
        x, y, theta = position.config

        fixture_def = b2FixtureDef(
            shape=b2PolygonShape(
                vertices=[(0, -width), (0, width), (length, width), (length, -width)]
            ),
            density=1,
            friction=1.0,
            categoryBits=0x0010,
            maskBits=0x1101,
        )

        door_body = world_def.world.CreateDynamicBody(
            position=(x, y),
            angle=theta,
            angularDamping=0.8,
            linearDamping=0.8,
            userData=self,
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
            maxMotorTorque=500,
        )

        return door_fixture, door_hinge


class Lock(Object):
    def __init__(self, name, locked):
        Object.__init__(self, name)
        self.locked = locked

    def lock(
        self, world_def=None, body=None, x=None, y=None, delta_x=None, delta_y=None
    ):
        if world_def is not None:
            self.joint = world_def.world.CreateWeldJoint(
                bodyA=body,  # end of link A
                bodyB=world_def.ground,  # beginning of link B
                localAnchorB=(x + delta_x, y + delta_y),
            )

        self.locked = True

    def unlock(self, world_def=None):
        if world_def is not None:
            world_def.world.DestroyJoint(self.joint)

        self.locked = False


class Button(Object):
    def __init__(
        self,
        world_def,
        position,
        color,
        name,
        height,
        width,
        x_offset=0,
        y_offset=0,
        clickable=None,
    ):
        Object.__init__(self, name)
        self.position = position
        x, y, theta = position.config
        button = world_def.world.CreateStaticBody(
            position=(x + x_offset, y + y_offset),
            angle=theta,
            shapes=b2PolygonShape(box=(height, width)),
            userData=self,
        )
        self.fixture = button.fixtures[0]
        self.color = color
        self.clickable = None

    def create_clickable(self, step, callback_action):
        vertices = [
            self.fixture.body.GetWorldPoint(vertex)
            for vertex in self.fixture.shape.vertices
        ]
        poly = Polygon(vertices)
        self.clickable = Clickable(
            lambda xy, poly: poly.contains(Point(xy)),
            step,
            callback_args=[callback_action],
            test_args=[poly],
        )


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
    print("Hello! Welcome to the game!")

    # time.sleep(1)

    # time.sleep(1)
    print(
        """See that door on your right? It is the vertical vertical on your right, with the
             red circle (the door hinge) and black circle (it's lock). That is your only escape."""
    )
    # time.sleep(1)
    print(
        """To open it, you must manipulate the three locks (the rectangles above, below, and
             to your left). Their behavior is unknown! You'll know that you unlocked the door
             when the black circle goes away"""
    )
    # time.sleep(1)
    print("ready...")
    # time.sleep(1)
    print("set...")
    # time.sleep(1)
    print("go!")
