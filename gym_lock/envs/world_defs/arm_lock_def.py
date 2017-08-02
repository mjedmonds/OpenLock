import numpy as np
import Box2D as b2

FPS = 30

# TODO: cleanup initialization/reset method
# NOTE: action spaces are different..

class ArmLockDef(object):

    def __init__(self, x0):
        super(ArmLockDef, self).__init__()

        self.world = b2.b2World(gravity=(0, -10), doSleep=True)

        fixture_length = 5
        #self.x0 = .array([0.75*np.pi, 0.5*np.pi, 0, 0, 0, 0, 0])
        self.x0 = x0
        self.target = np.array([0, 0]) 
        width = 1.0 
        
        
        # create arm links 
        link_fixture = b2.b2FixtureDef( # all links have same properties
            density = 1.0,
            friction = 1.0,
            categoryBits=0x0001,
            maskBits=0x0000)
        print 'hello'

        arm_bodies = []
        arm_lengths = [] # needed for joints

        # create base
        base_fixture = b2.b2FixtureDef(
            shape=b2.b2PolygonShape(box=(1, 1)),
            density=100.0,
            friction=1,
        )
        arm_bodies.append(self.world.CreateBody(
            position = x0[0].pos,
            angle = x0[0].theta,
            fixtures = base_fixture))

        # create the rest of the arm    
        for i in range(1, len(x0)):
            length = np.linalg.norm(x0[i][0] - x0[i-1][0])
            arm_lengths.append(length) 
            link_fixture.shape=b2.b2PolygonShape(vertices=[(0,-width / 2), 
                                      (-length, -width / 2), 
                                      (-length, width / 2),
                                      (0, width / 2)
                                      ])
            arm_bodies.append(self.world.CreateDynamicBody(
                position = x0[i].pos,
                angle = x0[i].theta,
                fixtures = link_fixture))
        
        # create arm joints
        arm_joints = []
        for i in range (1, len(arm_bodies)):
            arm_joints.append(self.world.CreateRevoluteJoint(
                bodyA=arm_bodies[i - 1], # end of joint A
                bodyB=arm_bodies[i], # end of joint B
                localAnchorA=(0, 0),
                localAnchorB=(-arm_lengths[i - 1], 0), # for n links, there are n + 1 bodies 
                enableMotor=True,
                maxMotorTorque=400,
                enableLimit=False))
        #self.joint2 = self.world.CreateRevoluteJoint(
        #    bodyA=self.body1,
        #    bodyB=self.body2,
        #    localAnchorA=(0, -(fixture_length - 0.5)),
        #    localAnchorB=(0, fixture_length - 0.5),
        #    enableMotor=True,
        #    maxMotorTorque=400,
        #    enableLimit=False,
        #)

        #self.set_joint_angles(self.body1, self.body2, self.x0[0], self.x0[1])
        #self.set_joint_angles(self.target1, self.target2, self.target[0], self.target[1])
        #self.target1.active = False
        #self.target2.active = False

        #self.joint1.motorSpeed = self.x0[2]
        #self.joint2.motorSpeed = self.x0[3]

    def step(self, timestep, vel_iterations, pos_iterations):
        self.world.Step(timestep, vel_iterations, pos_iterations)

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

