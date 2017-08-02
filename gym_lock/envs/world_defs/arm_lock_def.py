import numpy as np
import Box2D as b2

FPS = 30

# TODO: cleanup initialization/reset method
# NOTE: action spaces are different..

class ArmLockDef(object):

    def __init__(self, x0):
        super(ArmLockDef, self).__init__()

        self.world = b2.b2World(gravity=(0, 0), doSleep=True)

        fixture_length = 5
        #self.x0 = .array([0.75*np.pi, 0.5*np.pi, 0, 0, 0, 0, 0])
        self.x0 = x0
        self.target = np.array([0, 0]) 
        width = 1.0 
        base_location = x0[0]
        arm_link_bodies = []
        link_fixture = b2.b2FixtureDef(
            density = 1.0,
            friction = 1.0,
            categoryBits=0x0001,
            maskBits=0x0000)
        print 'hello'

        # create links
        for i in range(1, len(x0)):
            length = np.linalg.norm(x0[i][0] - x0[i-1][0])
            link_fixture.shape=b2.b2PolygonShape(vertices=[(0,0), 
                                      (-length, 0), 
                                      (-length, width),
                                      (0, width)
                                      ])
            body = self.world.CreateDynamicBody(
                position = x0[i].pos,
                angle = x0[i].theta,
                fixtures = link_fixture)

        #rectangle_fixture = b2.b2FixtureDef(
        #    shape=b2.b2PolygonShape(box=(.5, fixture_length)),
        #    density=.5,
        #    friction=1,
        #)
        #square_fixture = b2.b2FixtureDef(
        #    shape=b2.b2PolygonShape(box=(1, 1)),
        #    density=100.0,
        #    friction=1,
        #)
        #self.base = self.world.CreateBody(
        #    position=(0, 15),
        #    fixtures=square_fixture,
        #)

        #self.body1 = self.world.CreateDynamicBody(
        #    position=(0, 2),
        #    fixtures=rectangle_fixture,
        #    angle=b2.b2_pi,
        #)

        #self.body2 = self.world.CreateDynamicBody(
        #    fixtures=rectangle_fixture,
        #    position=(0, 2),
        #    angle=b2.b2_pi,
        #)
        #self.target1 = self.world.CreateDynamicBody(
        #    fixtures=rectangle_fixture,
        #    position=(0, 0),
        #    angle=b2.b2_pi,
        #)
        #self.target2 = self.world.CreateDynamicBody(
        #    fixtures=rectangle_fixture,
        #    position=(0, 0),
        #    angle=b2.b2_pi,
        #)

        #self.joint1 = self.world.CreateRevoluteJoint(
        #    bodyA=self.base,
        #    bodyB=self.body1,
        #    localAnchorA=(0, 0),
        #    localAnchorB=(0, fixture_length),
        #    enableMotor=True,
        #    maxMotorTorque=400,
        #    enableLimit=False,
        #)

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

    def set_joint_angles(self, body1, body2, angle1, angle2):
        """ Converts the given absolute angle of the arms to joint angles"""
        pos = self.base.GetWorldPoint((0, 0))
        body1.angle = angle1 + np.pi
        new_pos = body1.GetWorldPoint((0, 5))
        body1.position += pos - new_pos
        body2.angle = angle2 + body1.angle
        pos = body1.GetWorldPoint((0, -4.5))
        new_pos = body2.GetWorldPoint((0, 4.5))
        body2.position += pos - new_pos

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

