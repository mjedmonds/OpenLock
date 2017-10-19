import numpy as np
from Box2D import *
from Box2D import _Box2D
from types import MethodType

from gym_lock.common import TwoDConfig, TwoDForce
from gym_lock.common import wrapToMinusPiToPi
from gym_lock.pid_central import PIDController
from gym_lock.settings import BOX2D_SETTINGS
from gym_lock.state_machine.multi_lock import MultiDoorLockFSM


# TODO: cleaner interface than indices between bodies and lengths
# TODO: cleanup initialization/reset method
# TODO: no __ for class parameters
# TODO: add state machine here

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
        self.fsm = MultiDoorLockFSM()
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

        self.door, self.door_hinge, self.door_lock = self._create_door(TwoDConfig(15, 5, -np.pi / 2))

        open_test = lambda door_hinge: abs(door_hinge.angle) > np.pi / 16
        self.obj_map['door'] = [self.door, self.door_hinge, open_test, open_test]

        configs = [TwoDConfig(0, 15, 0), TwoDConfig(-15, 0, np.pi / 2), TwoDConfig(0, -15, -np.pi)]

        opt_params = [None, None, {'lower_lim': 0.0, 'upper_lim': 2.0}]

        for i in range(0, len(configs)):
            if opt_params[i]:
                lock, joint = self._create_lock(configs[i], **opt_params[i])
            else:
                lock, joint = self._create_lock(configs[i])

            # true iff out
            int_test = lambda joint: joint.translation < (joint.upperLimit + joint.lowerLimit) / 2.0

            # true iff in
            ext_test = lambda joint: joint.translation > (joint.upperLimit + joint.lowerLimit) / 2.0

            self.obj_map['l{}'.format(i)] = [lock, joint, int_test, ext_test]

        # modify l2, true iff in
        self.obj_map['l2'][2] = lambda joint: joint.translation > (joint.upperLimit + joint.lowerLimit) / 2.0
        self.obj_map['l2'][3] = lambda joint: joint.translation > (joint.upperLimit + joint.lowerLimit) / 2.0

    def _create_door(self, config, width=0.5, length=10, locked=True):
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

        door_body = self.world.CreateDynamicBody(
            position=(x, y),
            angle=theta,
            angularDamping=0.8,
            linearDamping=0.8)

        door = door_body.CreateFixture(fixture_def)

        door_hinge = self.world.CreateRevoluteJoint(
            bodyA=door.body,  # end of link A
            bodyB=self.ground,  # beginning of link B
            localAnchorA=(0, 0),
            localAnchorB=(x, y),
            enableMotor=True,
            motorSpeed=0,
            enableLimit=False,
            maxMotorTorque=500)

        door_lock = None
        if locked:
            delta_x = np.cos(theta) * length
            delta_y = np.sin(theta) * length
            door_lock = self.world.CreateWeldJoint(
                bodyA=door.body,  # end of link A
                bodyB=self.ground,  # beginning of link B
                localAnchorB=(x + delta_x, y + delta_y),
            )

        return door, door_hinge, door_lock

    def _create_lock(self, config, width=0.5, length=5, lower_lim=-2, upper_lim=0):
        x, y, theta = config

        fixture_def = b2FixtureDef(
            shape=b2PolygonShape(vertices=[(-length, -width),
                                           (-length, width),
                                           (length, width),
                                           (length, -width)]),
            density=1,
            friction=1.0,
            categoryBits=0x0010,
            maskBits=0x1101)

        lock_body = self.world.CreateDynamicBody(
            position=(x, y),
            angle=theta,
            angularDamping=0.8,
            linearDamping=0.8)

        lock = lock_body.CreateFixture(fixture_def)

        joint_axis = (-np.sin(theta), np.cos(theta))
        lock_joint = self.world.CreatePrismaticJoint(
            bodyA=lock.body,
            bodyB=self.ground,
            #anchor=(0, 0),
            anchor=lock.body.position,
            #localAnchorA=lock.body.position,
            #localAnchorB=self.ground.position,
            axis=joint_axis,
            lowerTranslation=lower_lim,
            upperTranslation=upper_lim,
            enableLimit=True,
            motorSpeed=0,
            maxMotorForce=abs(b2Dot(lock_body.massData.mass * self.world.gravity, b2Vec2(joint_axis))),
            enableMotor=True,
            userData={'plot_padding': width},
        )

        return lock, lock_joint

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
            'OBJ_STATES': {name: val[3](val[1]) for name, val in self.obj_map.items()},  # ext state
            'LOCK_STATE': self.obj_map['door'][2](self.obj_map['door'][1]),
            '_FSM_STATE': self.fsm.state,

        }

    def update_cascade_controller(self):
        if self.clock % BOX2D_SETTINGS['POS_PID_CLK_DIV'] == 0:
            theta = [c.theta for c in self.get_rel_config()[1:]]
            vel_setpoints = self.pos_controller.update(theta)
            self.vel_controller.set_setpoint(vel_setpoints)
        joint_speeds = [joint.speed for joint in self.arm_joints]
        return self.vel_controller.update(joint_speeds)

    def update_state_machine(self):

        # execute state transitions

        # check locks
        for name, val in self.obj_map.items():
            lock, joint, test, _ = val

            if test(joint):
                # unlocked
                action = 'unlock_{}'.format(name) if name != 'door' else 'open'
                if action in self.fsm.actions:
                    self.fsm.trigger(action)
            else:
                # unlock
                action = 'lock_{}'.format(name) if name != 'door' else 'close'
                if action in self.fsm.actions:
                    self.fsm.trigger(action)

        # check door

        if not '+' in self.fsm.state.split('o')[0] and self.door_lock is not None:
            # all locks open
            self._unlock_door()
        elif '+' in self.fsm.state.split('o')[0] and self.door_lock is None:
            self._lock_door()

        if 'l0-l1-' in self.fsm.state:
            lock, joint, _, _ = self.obj_map['l2']
            joint_axis = (-np.sin(lock.body.angle), np.cos(lock.body.angle))
            joint.maxMotorForce = abs(b2Dot(lock.body.massData.mass * self.world.gravity, b2Vec2(joint_axis)))
        else:
            self.obj_map['l2'][1].maxMotorForce = 100000




            # if 'o+' in self.fsm.state and self.door_lock is not None:
            #     self._unlock_door()
            # elif 'o-' in self.fsm.state and self.door_lock is None:
            #     self.world.CreateJoint()

    def set_controllers(self, setpoints):
        # make sure that angles are in [-pi, pi]
        new = [wrapToMinusPiToPi(c) for c in setpoints]

        # update position PID
        self.pos_controller.set_setpoint(new)

        # update velocity PID instead of waiting until next step
        theta = [c.theta for c in self.get_rel_config()[1:]]
        vel_setpoints = self.pos_controller.update(theta)
        self.vel_controller.set_setpoint(vel_setpoints)

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
            self.update_state_machine()

        # update torques
        new_torque = self.update_cascade_controller()
        for i in range(0, len(new_torque)):
            self.apply_torque(i + 1, new_torque[i])

        self.world.Step(timestep, vel_iterations, pos_iterations)

    # TODO: implement
    def reset_world(self):
        """Returns the world to its intial state"""
        pass
