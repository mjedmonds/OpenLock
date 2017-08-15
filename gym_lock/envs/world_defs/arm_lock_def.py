import Box2D as b2
import numpy as np

from gym_lock.common import TwoDConfig
from gym_lock.pid_central import PIDController
from gym_lock.common import wrapToMinusPiToPi

FPS = 30


# TODO: cleaner interface than indices between bodies and lengths
# TODO: cleanup initialization/reset method
# NOTE: action spaces are different..

# class stateMachineListener(b2.b2ContactListener):
#
#     def __init__(self):
#         b2.b2ContactListener.__init__(self)
#         print "creatded"

    # def BeginContact(self, contact):
    #     print 'begin'
    #     exit()

    # def EndContact(self, contact):
    #     print'end'
    #     exit()

    # def PreSolve(self, contact, oldManifold):
    #     print 'pre'
    #     exit()
    #
    # def PostSolve(self, contact, impulse):
    #     print 'post'
    #     exit()
    #     print impulse

class ArmLockDef(object):
    def __init__(self, chain, world_size):
        super(ArmLockDef, self).__init__()

        self.world = b2.b2World(gravity=(0, 0),
                                doSleep=False)

        self.clock = 0

        self.num_steps = 0
        self.start = np.array([0,0,0,0])


        self.x0 = chain.get_abs_config()
        self.chain = chain
        width = 1.0

        # create boundaries
        self.ground = self.world.CreateBody()
        self.ground.CreateEdgeChain([(-world_size, -world_size),
                                     (world_size, -world_size),
                                     (world_size, world_size),
                                     (-world_size, world_size),
                                     (-world_size, -world_size)])

        # create arm links
        self.arm_bodies = []
        motor_fixtures = []  # needed for torque control
        self.arm_lengths = []  # needed for joints

        # define all fixtures
        # define base properties
        base_fixture = b2.b2FixtureDef(
            shape=b2.b2PolygonShape(box=(1, 1)),
            density=100.0,
            friction=1.0,
            categoryBits=0x0001,
            maskBits=0x1110)

        # define link properties
        link_fixture = b2.b2FixtureDef(  # all links have same properties
            density=1.0,
            friction=1.0,
            categoryBits=0x0001,
            maskBits=0x1110)

        # create base
        self.arm_bodies.append(self.world.CreateBody(
            position=(self.x0[0].x, self.x0[0].y),
            angle=0))

        #TODO: remove?
        # add in "virtual" joint length so arm_bodies and arm_lengths are same length
        length = np.linalg.norm(np.array([0, 0]) - \
                                np.array([self.x0[0].x, self.x0[0].y]))

        self.arm_lengths.append(length)

        # create the rest of the arm
        # body frame located at each joint
        for i in range(1, len(self.x0)):
            length = np.linalg.norm(np.array([self.x0[i].x, self.x0[i].y] - \
                                             np.array([self.x0[i - 1].x, self.x0[i - 1].y])))
            self.arm_lengths.append(length)

            link_fixture.shape = b2.b2PolygonShape(vertices=[(0, -width / 2),
                                                             (-length, -width / 2),
                                                             (-length, width / 2),
                                                             (0, width / 2)
                                                             ])
            arm_body = self.world.CreateDynamicBody(
                position=(self.x0[i].x, self.x0[i].y),
                angle=self.x0[i].theta,
                fixtures=link_fixture,
                linearDamping=0.5,
                angularDamping=1)

            self.arm_bodies.append(arm_body)

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
                maxMotorTorque=0,
                enableLimit=False))


        # self.__init_door_lock()
        self.__init_cascade_controller()


    def __init_door_lock(self):

        # create door
        door_width = 0.5
        door_length = 10
        door_fixture = b2.b2FixtureDef(
            shape=b2.b2PolygonShape(vertices=[(0,-door_width),
                                             (0, door_width),
                                             (door_length, door_width),
                                             (door_length, -door_width)]),
            density=1,
            friction=1.0)
        self.door = self.world.CreateDynamicBody(
            position = (15, 10),
            angle = -np.pi/2,
            fixtures = door_fixture)

        self.door_hinge = self.world.CreateRevoluteJoint(
            bodyA=self.door,  # end of link A
            bodyB=self.ground,  # beginning of link B
            localAnchorA=(0, 0),
            localAnchorB=(15, 10),
            enableMotor=True,
            motorSpeed=0,
            maxMotorTorque=100000,
            enableLimit=True)


        lock_width = 0.5
        lock_length = 5
        lock_fixture = b2.b2FixtureDef(
            shape=b2.b2PolygonShape(vertices=[(0,-lock_width),
                                                    (0, lock_width),
                                                    (lock_length, lock_width),
                                                    (lock_length, -lock_width)]),
            density=1,
            friction=1.0)

        self.lock = self.world.CreateDynamicBody(
            position = (15, -5),
            angle = -np.pi/2,
            fixtures = lock_fixture)
        self.lock_joint = self.world.CreatePrismaticJoint(
            bodyA=self.lock,
            bodyB=self.ground,
            anchor=(0, 0),
            axis=(1, 0),
            lowerTranslation=0,
            upperTranslation=2,
            enableLimit = True
        )


    def __init_cascade_controller(self):
        pts = [c.theta for c in self.chain.get_rel_config()[1:]]
        pts = [np.pi, -np.pi, -np.pi/2, 0]
        self.pos_controller = PIDController([1] * len(self.arm_joints),
                                        [0.0000001] * len(self.arm_joints),
                                        [0.0] * len(self.arm_joints),
                                        pts)

        # initialize with zero velocity
        pts = [0, 0, 0, 0]
        self.vel_controller = PIDController([1000] * len(self.arm_joints),
                                            [0.0] * len(self.arm_joints),
                                            [0] * len(self.arm_joints),
                                            pts,
                                            max_out=5)

    def update_cascade_controller(self, theta):
        # TODO: formalize clockrate
        if self.clock % 50 == 0:
            # print 'here'
            # print self.pos_controller.setpoint
            # print self.pos_controller.integral
            vel_setpoints = self.pos_controller.update(theta)
            self.vel_controller.set_setpoint(vel_setpoints)
            print vel_setpoints

        joint_speeds = [joint.speed for joint in self.arm_joints]
        torques = self.vel_controller.update(joint_speeds)
        if self.clock % 50 == 0:
            print '_--------------'
            print torques
            print self.pos_controller.integral
            print self.pos_controller.i_term
        return torques



    def update_state_machine(self):
        if self.lock_joint.translation > 1.0 and self.bolt_joint:
            self.door_hinge.maxMotorTorque = 70
            self.bolt_joint = None

    def set_controllers(self, delta_setpoints):
        print 'wtf?'
        exit()
        new = [wrapToMinusPiToPi(c + n) \
               for c, n in zip(self.pos_controller.setpoint, delta_setpoints)]
        self.pos_controller.set_setpoint(new)

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
        force = torque / self.arm_lengths[idx]

        angle = self.arm_bodies[idx].transform.angle
        position = self.arm_bodies[idx].position

        yaxis = b2.b2Vec2([-np.sin(angle), np.cos(angle)])
        force_vector = yaxis * force

        self.arm_bodies[idx].ApplyForce(force=force_vector, point=position, wake=True)

    def step(self, timestep, vel_iterations, pos_iterations):
        self.clock += 1

        # self.update_state_machine()

        # update torques
        theta = [c.theta for c in self.get_rel_config()[1:]]
        new_torque = self.update_cascade_controller(theta)

        for i in range(0, len(new_torque)):
            self.apply_torque(i + 1, new_torque[i])

        self.world.Step(timestep, vel_iterations, pos_iterations)

    # TODO: implement
    def reset_world(self):
        """Returns the world to its intial state"""
        pass

    def get_state(self):
        state = {'joint_config': self.get_abs_config()}
        return state
