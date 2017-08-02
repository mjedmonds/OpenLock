import numpy as np
import Box2D as b2

FPS = 30

# TODO: cleaner interface than indices between bodies and lengths
# TODO: cleanup initialization/reset method
# NOTE: action spaces are different..

class ArmLockDef(object):

    def __init__(self, x0):
        super(ArmLockDef, self).__init__()

        self.world = b2.b2World(gravity=(0, -10), doSleep=True)

        self.x0 = x0
        width = 1.0 
        
    
        # create arm links 
        self.arm_bodies = []
        motor_fixtures = [] # needed for torque control
        self.arm_lengths = [] # needed for joints

        # define base properties
        base_fixture = b2.b2FixtureDef(
            shape=b2.b2PolygonShape(box=(1, 1)),
            density=100.0,
            friction=1,
        )
        # create base
        self.arm_bodies.append(self.world.CreateBody(
            position = x0[0].pos,
            angle = 0,
            fixtures = base_fixture))
        
        # define link properties
        link_fixture = b2.b2FixtureDef( # all links have same properties
            density = 1.0,
            friction = 1.0,
            categoryBits = 0x0001,
            maskBits = 0x0000)
        # define "motor" properties
        motor_fixture = b2.b2FixtureDef( # all motors have same properties
            density = 1.0,
            friction = 1.0,
            categoryBits = 0x0001,
            maskBits = 0x0000)
        
        # create the rest of the arm
        # body frame located at each joint
        for i in range(1, len(x0)):
            length = np.linalg.norm(x0[i - 1][0] - x0[i][0])
            self.arm_lengths.append(length) 
            link_fixture.shape=b2.b2PolygonShape(vertices=[(0,-width / 2), 
                                      (-length, -width / 2), 
                                      (-length, width / 2),
                                      (0, width / 2)
                                      ])
            arm_body = self.world.CreateDynamicBody(
                position = x0[i].pos,
                angle = x0[i].theta,
                fixtures = link_fixture)

            self.arm_bodies.append(arm_body)

            motor_fixture.shape = b2.b2CircleShape(radius=(width), pos=(-length, 0))
            motor_fixtures.append(arm_body.CreateFixture(motor_fixture))
       
        # create arm joints
        arm_joints = []
        for i in range (1, len(self.arm_bodies)):
            arm_joints.append(self.world.CreateRevoluteJoint(
                bodyA=self.arm_bodies[i - 1], # end of link A
                bodyB=self.arm_bodies[i], # beginning of link B 
                localAnchorA=(0, 0),
                localAnchorB=(-self.arm_lengths[i - 1], 0), # for n links, there are n + 1 bodies 
                enableMotor=True,
                maxMotorTorque=400,
                enableLimit=False))


    def apply_torque(self, idx, torque):
        force = torque / self.arm_lengths[idx - 1]
        angle = self.arm_bodies[idx].transform.angle
        position = self.arm_bodies[idx].position
        yaxis = b2.b2Vec2([-np.sin(angle), np.cos(angle)])
        force_vector = yaxis * force
        self.arm_bodies[idx].ApplyForce(force=force_vector, point=position, wake=True)

    def step(self, timestep, vel_iterations, pos_iterations):
        self.world.Step(timestep, vel_iterations, pos_iterations)

    # TODO: deprecated?
    def take_action(self, action):
        self.joint1.motorSpeed = action[0]
        self.joint2.motorSpeed = action[1]
    
    def reset_world(self):
        """Returns the world to its intial state"""
        self.world.ClearForces()
        self.joint1.motorSpeed = 0
        self.joint2.motorSpeed = 0
        self.body1.linearVelocity = (0, 0)
        self.body1.angularVelocity = 0
        self.body2.linearVelocity = (0, 0)
        self.body2.angularVelocity = 0
        self.set_joint_angles(self.body1, self.body2, self.x0[0], self.x0[1])

    def get_state(self):
        """Retrieves the state of the point mass"""
        state = {'joint_angles' : np.array([self.joint1.angle,
                                         self.joint2.angle]),
                 'joint_velocities' : np.array([self.joint1.speed,
                                             self.joint2.speed]),
                 'end_effector_points' : np.append(np.array(self.body2.position),[0])}
        return state

