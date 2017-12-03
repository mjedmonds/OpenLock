import numpy as np
from Box2D import *
from Box2D import _Box2D
from types import MethodType

from gym_lock.common import TwoDConfig, TwoDForce, Lock, Door
from gym_lock.common import wrapToMinusPiToPi
from gym_lock.pid_central import PIDController
from gym_lock.settings import BOX2D_SETTINGS
from gym_lock.state_machine.multi_lock import MultiDoorLockFSM
from gym_lock.common import Button, COLORS


# TODO: cleaner interface than indices between bodies and lengths
# TODO: cleanup initialization/reset method
# TODO: no __ for class parameters
# TODO: add state machine here

# class Clickable(object):
#
#     def __init__(self, test, callback, args):
#         self.test = test
#         self.callback = callback
#         self.args = args
#
#     def test_region(self, world_xy):
#         return self.test(world_xy)
#
#     def call(self):
#         return self.callback(*args)

class ArmLockContactListener(b2ContactListener):
    def __init__(self, end_effector_fixture, timestep):
        b2ContactListener.__init__(self)
        self.__end_effector_fixture = end_effector_fixture
        self.__timestep = timestep

        self.__contacting = False
        self.__tan_force_vector = self.__norm_force_vector = None

        # self.__total_norm_force_vector = self.__total_tan_force_vector = b2Vec2(0, 0)
        # self.__iterations = 0

    @property
    def norm_force(self):
        return self.__norm_force_vector

    @property
    def tan_force(self):
        return self.__tan_force_vector

    def __filter_contact(self, contact):
        if self.__end_effector_fixture == contact.fixtureA:
            return 'A'
        elif self.__end_effector_fixture == contact.fixtureB:
            return 'B'
        else:
            return False

    def BeginContact(self, contact):
        if self.__filter_contact(contact):
            self.__contacting = True
            # self.__total_tan_force_vector = self.__total_norm_force_vector = b2Vec2(0, 0)
            # self.__iterations = 0

    def EndContact(self, contact):
        if self.__filter_contact(contact):
            self.__contacting = False
            self.__norm_force_vector = self.__tan_force_vector = None

    def PreSolve(self, contact, oldManifold):
        pass

    def PostSolve(self, contact, impulse):
        fixture_id = self.__filter_contact(contact)
        if fixture_id:
            manifold = contact.worldManifold

            # checking assumtion: Every contact.points list has two points: true contact point in world coordinates
            # and (0,0), checking assumption that former is always head of list
            assert manifold.points[0] != (0, 0)

            # checking assumptions...cannot find documentation on how/why length of impulses
            # would be greater than 1
            assert len(impulse.normalImpulses) == len(impulse.tangentImpulses) == 1

            norm_imp = impulse.normalImpulses[0]
            tan_imp = impulse.tangentImpulses[0]

            if fixture_id == 'A':
                transform = contact.fixtureA.body.GetLocalPoint(manifold.points[0])
                norm_vector = -manifold.normal
            else:
                transform = contact.fixtureB.body.transform
                norm_vector = manifold.normal

            norm_vector = transform.R * norm_vector
            norm_vector.Normalize()
            tan_vector = b2Vec2(-norm_vector[1], norm_vector[0])

            norm_force_vector = (norm_imp / self.__timestep) * norm_vector
            tan_force_vector = (tan_imp / self.__timestep) * tan_vector

            self.__norm_force_vector = norm_force_vector
            self.__tan_force_vector = tan_force_vector

            #
            # self.__total_norm_force_vector += norm_force_vector
            # self.__total_tan_force_vector += tan_force_vector
            #
            # self.__iterations += 1


class ArmLockDef(object):
    def __init__(self, chain, timestep, world_size):
        super(ArmLockDef, self).__init__()

        self.timestep = timestep
        self.chain = chain

        self.world = b2World(gravity=(0, -10),
                             doSleep=False)
        self.background = b2World(gravity=(0,0), dosleep=True)

        self.clock = 0
        self.target_arrow = None

        x0 = chain.get_abs_config()

        # create boundaries
        self.ground = self.world.CreateBody()
        self.ground.CreateEdgeChain([(-world_size, -world_size),
                                     (world_size, -world_size),
                                     (world_size, world_size),
                                     (-world_size, world_size),
                                     (-world_size, -world_size)])

        self.obj_map = dict()
        self.grasped_list = []
        self.fsm = MultiDoorLockFSM(self)
        self.__init_arm(x0)
        self._init_fsm_rep()
        self.__init_cascade_controller()

        self.contact_listener = ArmLockContactListener(self.end_effector_fixture, self.timestep)
        self.world.contactListener = self.contact_listener


        # for body in self.world.bodies:
        #     body.bullet = True

    def __init_arm(self, x0):
        # create arm links
        self.arm_bodies = []
        self.arm_lengths = []  # needed for joints

        # define all fixtures
        # define base properties
        base_fixture_def = b2FixtureDef(
            shape=b2PolygonShape(box=(1, 1)),
            density=10.0,
            friction=1.0,
            categoryBits=0x0001,
            maskBits=0x1110)

        # define link properties
        link_fixture_def = b2FixtureDef(  # all links have same properties
            density=1.0,
            friction=1.0,
            categoryBits=0x0001,
            maskBits=0x1110)
        # define end effector properties
        end_effector_fixture_def = b2FixtureDef(  # all links have same properties
            density=0.1,
            friction=1.0,
            categoryBits=0x0001,
            maskBits=0x1110,
            shape=b2CircleShape(radius=0.5))

        # create base
        self.arm_bodies.append(self.world.CreateBody(
            position=(x0[0].x, x0[0].y),
            angle=0))

        # add in "virtual" joint length so arm_bodies and arm_lengths are same length
        length = np.linalg.norm(np.array([0, 0]) - \
                                np.array([x0[0].x, x0[0].y]))

        self.arm_lengths.append(length)

        # create the rest of the arm
        # body frame located at each joint
        for i in range(1, len(x0)):
            length = np.linalg.norm(np.array([x0[i].x, x0[i].y] - \
                                             np.array([x0[i - 1].x, x0[i - 1].y])))
            self.arm_lengths.append(length)

            link_fixture_def.shape = b2PolygonShape(vertices=[(0, -BOX2D_SETTINGS['ARM_WIDTH'] / 2),
                                                              (-BOX2D_SETTINGS['ARM_LENGTH'],
                                                               -BOX2D_SETTINGS['ARM_WIDTH'] / 2),
                                                              (-BOX2D_SETTINGS['ARM_LENGTH'],
                                                               BOX2D_SETTINGS['ARM_WIDTH'] / 2),
                                                              (0, BOX2D_SETTINGS['ARM_WIDTH'] / 2)])
            arm_body = self.world.CreateDynamicBody(
                position=(x0[i].x, x0[i].y),
                angle=x0[i].theta,
                fixtures=link_fixture_def,
                linearDamping=0.5,
                angularDamping=1)

            self.arm_bodies.append(arm_body)
        self.end_effector_fixture = self.arm_bodies[-1].CreateFixture(end_effector_fixture_def)

        # create arm joints
        self.arm_joints = []
        for i in range(1, len(self.arm_bodies)):
            # enableMotor = True for motor friction, helps dampen oscillations
            self.arm_joints.append(self.world.CreateRevoluteJoint(
                bodyA=self.arm_bodies[i - 1],  # end of link A
                bodyB=self.arm_bodies[i],  # beginning of link B
                localAnchorA=(0, 0),
                localAnchorB=(-self.arm_lengths[i], 0),
                enableMotor=True,
                motorSpeed=0,
                maxMotorTorque=50,
                enableLimit=False))

    def _init_fsm_rep(self):
        # TODO: better setup interface

        door_config = TwoDConfig(15, 5, -np.pi / 2)
        door = Door(self, 'door', door_config)
        self.obj_map['door'] = door

        self.obj_map['door_right_button'] = Button(world=self.world, config=door_config, color=COLORS['static'], name='door_right_button', height=1.5, width=1.5, x_offset=3, y_offset=5)
        self.obj_map['door_left_button'] = Button(world=self.world, config=door_config, color=COLORS['static'], name='door_left_button', height=1.5, width=1.5, x_offset=-3, y_offset=5)

        button_config = TwoDConfig(-25, -27, -np.pi / 2)
        self.obj_map['save_button'] = Button(world=self.world, config=button_config, color=COLORS['save_button'], name='save_button', height=1.5, width=3)
        self.obj_map['reset_button'] = Button(world=self.world, config=button_config, color=COLORS['reset_button'], name='reset_button', height=1.5, width=3, x_offset=7)

        configs = [TwoDConfig(0, 15, 0), TwoDConfig(-15, 0, np.pi / 2), TwoDConfig(0, -15, -np.pi)]

        opt_params = [None, None, {'lower_lim': 0.0, 'upper_lim': 2.0}]

        for i in range(0, len(configs)):
            name = 'l{}'.format(i)
            lock = Lock(self, name, configs[i], opt_params[i])
            self.obj_map[name] = lock

    def _lock_door(self):
        theta = self.door.body.angle
        length = max([v[0] for v in self.door.shape.vertices])
        x, y = self.door.body.position

        delta_x = np.cos(theta) * length
        delta_y = np.sin(theta) * length

        self.door_lock = self.world.CreateWeldJoint(
            bodyA=self.door.body,  # end of link A
            bodyB=self.ground,  # beginning of link B
            localAnchorB=(x + delta_x, y + delta_y),
        )

    def _unlock_door(self):
        self.world.DestroyJoint(self.door_lock)
        self.door_lock = None

    def __init_cascade_controller(self):
        pts = [c.theta for c in self.chain.get_rel_config()[1:]]
        self.pos_controller = PIDController([10] * len(self.arm_joints),
                                            [1] * len(self.arm_joints),
                                            [0] * len(self.arm_joints),
                                            pts,
                                            self.timestep,
                                            max_out=1.5,
                                            err_wrap_func=wrapToMinusPiToPi)

        # initialize with zero velocity
        self.vel_controller = PIDController([17000] * len(self.arm_joints),
                                            [0] * len(self.arm_joints),
                                            [0] * len(self.arm_joints),
                                            [0] * len(self.arm_joints),
                                            self.timestep,
                                            max_out=30000)

    def get_state(self):

        # true iff out
        in_out_test = lambda joint: joint.translation < (joint.upperLimit + joint.lowerLimit) / 2.0

        return {
            'END_EFFECTOR_POS': self.get_abs_config()[-1],
            'END_EFFECTOR_FORCE': TwoDForce(self.contact_listener.norm_force, self.contact_listener.tan_force),
            # 'DOOR_ANGLE' : self.obj_map['door'][1].angle,
            # 'LOCK_TRANSLATIONS' : {name : val[1].translation for name, val in self.obj_map.items() if name != 'door'},
            'OBJ_STATES': {name: val.ext_test(val.joint) for name, val in self.obj_map.items() if 'button' not in name},  # ext state
            # 'OBJ_STATES': {name: val[3](val[1]) for name, val in self.obj_map.items() if 'button' not in name},
        # ext state
            'LOCK_STATE': self.obj_map['door'].int_test(self.obj_map['door'].joint),
            # 'LOCK_STATE': self.obj_map['door'][2](self.obj_map['door'][1]),
            '_FSM_STATE': self.fsm.observable_fsm.state + self.fsm.latent_fsm.state,

        }

    def update_cascade_controller(self):
        if self.clock % BOX2D_SETTINGS['POS_PID_CLK_DIV'] == 0:
            theta = [c.theta for c in self.get_rel_config()[1:]]
            vel_setpoints = self.pos_controller.update(theta)
            self.vel_controller.set_setpoint(vel_setpoints)
        joint_speeds = [joint.speed for joint in self.arm_joints]
        return self.vel_controller.update(joint_speeds)

    def set_controllers(self, setpoints):
        # make sure that angles are in [-pi, pi]
        new = [wrapToMinusPiToPi(c) for c in setpoints]

        # update position PID
        self.pos_controller.set_setpoint(new)

        # update velocity PID instead of waiting until next step
        theta = [c.theta for c in self.get_rel_config()[1:]]
        vel_setpoints = self.pos_controller.update(theta)
        self.vel_controller.set_setpoint(vel_setpoints)

    def lock_lever(self, lever):
        self.obj_map['l2'].joint.maxMotorForce = 100000

    def unlock_lever(self, lever):
        lock = self.obj_map['l2'].fixture
        joint = self.obj_map['l2'].joint
        joint_axis = (-np.sin(lock.body.angle), np.cos(lock.body.angle))
        joint.maxMotorForce = abs(b2Dot(lock.body.massData.mass * self.world.gravity, b2Vec2(joint_axis)))

    def get_abs_config(self):
        config = []

        for i in range(0, len(self.arm_bodies)):
            x = self.arm_bodies[i].position[0]
            y = self.arm_bodies[i].position[1]
            theta = self.arm_bodies[i].transform.angle

            config.append(TwoDConfig(x, y, theta))

        return config

    def get_rel_config(self):
        config = []
        x = y = theta = 0
        for i in range(0, len(self.arm_bodies)):
            next_x = self.arm_bodies[i].position[0]
            next_y = self.arm_bodies[i].position[1]
            next_theta = wrapToMinusPiToPi(self.arm_bodies[i].transform.angle)

            dx = next_x - x
            dy = next_y - y
            dtheta = wrapToMinusPiToPi(next_theta - theta)

            x = next_x
            y = next_y
            theta = next_theta

            config.append(TwoDConfig(dx, dy, dtheta))

        return config

    def apply_torque(self, idx, torque):

        # compute force if which applied at end of link would yield 'torque'
        force = torque / self.arm_lengths[idx]

        # compute normal to end of link where force will be exerted
        angle = self.arm_bodies[idx].transform.angle
        position = self.arm_bodies[idx].position
        yaxis = b2Vec2([-np.sin(angle), np.cos(angle)])

        force_vector = yaxis * force

        self.arm_bodies[idx].ApplyForce(force=force_vector, point=position, wake=True)

    def step(self, timestep, vel_iterations, pos_iterations):
        self.clock += 1

        if self.clock % BOX2D_SETTINGS['STATE_MACHINE_CLK_DIV'] == 0:
            self.fsm.update_state_machine()

        # update torques
        new_torque = self.update_cascade_controller()
        for i in range(0, len(new_torque)):
            self.apply_torque(i + 1, new_torque[i])

        self.world.Step(timestep, vel_iterations, pos_iterations)

    # TODO: implement
    def reset_world(self):
        """Returns the world to its intial state"""
        pass
