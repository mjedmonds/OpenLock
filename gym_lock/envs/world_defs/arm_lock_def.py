import Box2D as b2
import numpy as np

from gym_lock.common import TwoDConfig
from gym_lock.pid import PIDController

FPS = 30


# TODO: cleaner interface than indices between bodies and lengths
# TODO: cleanup initialization/reset method
# NOTE: action spaces are different..

class ArmLockDef(object):
    def __init__(self, x0):
        super(ArmLockDef, self).__init__()

        self.world = b2.b2World(gravity=(0, -2), doSleep=True)

        self.x0 = x0
        width = 1.0

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
            maskBits=0x0000)
        # define "motor" properties
        motor_fixture = b2.b2FixtureDef(  # all motors have same properties
            density=1.0,
            friction=1.0,
            categoryBits=0x0001,
            maskBits=0x0000)

        # create base
        self.arm_bodies.append(self.world.CreateBody(
            position=(x0[0].x, x0[0].y),
            angle=0,
            fixtures=base_fixture))

        # add in "virtual" joint length so arm_bodies and arm_lengths are same length
        length = np.linalg.norm(x0[1][0] - x0[0][0])
        self.arm_lengths.append(length)
        # create the rest of the arm
        # body frame located at each joint
        for i in range(1, len(x0)):
            length = np.linalg.norm(x0[i - 1][0] - x0[i][0])
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
            self.arm_joints.append(self.world.CreateRevoluteJoint(
                bodyA=self.arm_bodies[i - 1],  # end of link A
                bodyB=self.arm_bodies[i],  # beginning of link B
                localAnchorA=(0, 0),
                localAnchorB=(-self.arm_lengths[i], 0),
                enableMotor=True,
                maxMotorTorque=400,
                enableLimit=False))

        # create joint PID controllers and initialize to
        # angles specified in x0
        self.joint_controllers = []
        config = self.get_abs_config()[1:]  # ignore baseframe transform
        for i in range(0, len(self.arm_joints)):
            self.joint_controllers.append(PIDController(setpoint=config[i].theta,
                                                        dt=1.0 / FPS))

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
        for i in range(1, len(self.arm_bodies)):
            body_angle = self.arm_bodies[i].transform.angle
            new_torque = self.joint_controllers[i - 1].update(body_angle)
            self.apply_torque(i, new_torque)
        self.world.Step(timestep, vel_iterations, pos_iterations)

    # TODO: implement
    def reset_world(self):
        """Returns the world to its intial state"""
        pass

    # TODO: implement
    def get_state(self):
        """Retrieves the state of the point mass"""
        pass
