import Box2D as b2
import numpy as np

from gym_lock.common import TwoDConfig
from gym_lock.pid import PIDController

FPS = 30


# TODO: cleaner interface than indices between bodies and lengths
# TODO: cleanup initialization/reset method
# NOTE: action spaces are different..

class stateMachineListener(b2.b2ContactListener):

    def __init__(self):
        b2.b2ContactListener.__init__(self)
        print "creatded"

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
    def __init__(self, x0, world_size):
        super(ArmLockDef, self).__init__()

        listener = stateMachineListener()
        self.world = b2.b2World(gravity=(0, -2),
                                doSleep=False,
                                contactlistener=listener,
                                )
        self.world.contactListener = listener
        self.world.subStepping = True
        self.world.SetAllowSleeping(False)

        self.x0 = x0
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
            friction=1)
        # define link properties
        link_fixture = b2.b2FixtureDef(  # all links have same properties
            density=1.0,
            friction=1.0,
            categoryBits=0x0001,
            maskBits=0x1111)
        # define "motor" properties
        motor_fixture = b2.b2FixtureDef(  # all motors have same properties
            density=1.0,
            friction=1.0,
            categoryBits=0x0001,
            maskBits=0x1111)

        # create base
        self.arm_bodies.append(self.world.CreateBody(
            position=(x0[0].x, x0[0].y),
            angle=0,
            fixtures=base_fixture))

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

            link_fixture.shape = b2.b2PolygonShape(vertices=[(0, -width / 2),
                                                             (-length, -width / 2),
                                                             (-length, width / 2),
                                                             (0, width / 2)
                                                             ])
            arm_body = self.world.CreateDynamicBody(
                position=(x0[i].x, x0[i].y),
                angle=x0[i].theta,
                fixtures=link_fixture)

            self.arm_bodies.append(arm_body)

            motor_fixture.shape = b2.b2CircleShape(radius=(width), pos=(-length, 0))
            motor_fixtures.append(arm_body.CreateFixture(motor_fixture))
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
                maxMotorTorque=400,
                enableLimit=False))

        # create joint PID controllers and initialize to
        # angles specified in x0
        self.joint_controllers = []
        config = self.get_abs_config()[1:]  # ignore baseframe transform
        print config
        for i in range(0, len(self.arm_joints)):
            self.joint_controllers.append(PIDController(setpoint=config[i].theta,
                                                        dt=1.0 / FPS))


        # create door
        door_width = 0.5
        door_length = 10
        door_fixture = b2.b2FixtureDef(
            shape=b2.b2PolygonShape(vertices=[(0,-width),
                                             (0, width),
                                             (door_length, width),
                                             (door_length, -width)]),
            density=1,
            friction=1.0)
        self.door = self.world.CreateDynamicBody(
            position = (0, 10),
            fixtures = door_fixture)
        self.door_hinge = self.world.CreateRevoluteJoint(
            bodyA=self.door,  # end of link A
            bodyB=self.ground,  # beginning of link B
            localAnchorA=(0, 0),
            localAnchorB=(0, 10))

        # create lock
        # create door
        lock_width = 0.5
        lock_length = 10
        lock_fixture = b2.b2FixtureDef(
            shape=b2.b2PolygonShape(box=(5, 0.5)),
            density=1,
            friction=1.0)
        self.lock = self.world.CreateDynamicBody(
            position = (17, -world_size + 15.5),
            fixtures = lock_fixture,
            angle = np.pi / 2)
        self.lock_joint = self.world.CreatePrismaticJoint(
            bodyA=self.lock,
            bodyB=self.ground,
            anchor=(0, 0),
            axis=(0, 1),
            lowerTranslation=0,
            upperTranslation=1,
            enableLimit=True,
            motorSpeed=0.0,
            enableMotor=True,
        )

    def update_state_machine(self, input):
        pass

    def set_controllers(self, delta_setpoints):
        for i in range(0, len(self.joint_controllers)):
            cur = self.joint_controllers[i].setpoint
            new = cur + delta_setpoints[i]
            self.joint_controllers[i].set_setpoint(new)

    def get_abs_config(self):
        config = []

        for i in range(0, len(self.arm_bodies)):
            x = self.arm_bodies[i].position[0]
            y = self.arm_bodies[i].position[1]
            theta = self.arm_bodies[i].transform.angle
            print theta

            config.append(TwoDConfig(x, y, theta))

        return config

    def get_rel_config(self):
        config = []
        x = y = theta = 0
        for i in range(0, len(self.arm_bodies)):
            next_x = self.arm_bodies[i].position[0]
            next_y = self.arm_bodies[i].position[1]
            next_theta = self.arm_bodies[i].transform.angle

            dx = next_x - x
            dy = next_y - y
            dtheta = next_theta - theta

            x = next_x
            y = next_y
            theta = next_theta

            config.append(TwoDConfig(dx, dy, dtheta))

        return config

    def apply_torque(self, idx, torque):
        force = torque / self.arm_lengths[idx - 1]

        angle = self.arm_bodies[idx].transform.angle
        position = self.arm_bodies[idx].position

        yaxis = b2.b2Vec2([-np.sin(angle), np.cos(angle)])
        force_vector = yaxis * force

        self.arm_bodies[idx].ApplyForce(force=force_vector, point=position, wake=True)

    def step(self, timestep, vel_iterations, pos_iterations):
        # update torques
        self.update_state_machine(None)
        for i in range(1, len(self.arm_bodies)):
            body_angle = self.arm_bodies[i].transform.angle
            new_torque = self.joint_controllers[i - 1].update(body_angle)
            self.apply_torque(i, new_torque)
        self.world.Step(timestep, vel_iterations, pos_iterations)

    # TODO: implement
    def reset_world(self):
        """Returns the world to its intial state"""
        pass

    def get_state(self):
        state = {'joint_config': self.get_abs_config()}
        return state
