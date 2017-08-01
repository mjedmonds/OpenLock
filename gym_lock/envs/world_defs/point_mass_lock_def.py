import numpy as np

from abc import ABCMeta, abstractmethod
    
import Box2D as b2

FPS = 30

# TODO: cleanup initialization/reset method
# NOTE: action spaces are different..

class PointMassLockDef(object):

    def __init__(self):
        super(PointMassLockDef, self).__init__()

        self.world = b2.b2World(gravity=(0, -10), doSleep=True)

        self.initial_position = (0, 5)
        self.initial_angle = b2.b2_pi
        self.initial_linear_velocity = (0., 0.)  # (x0[2], x0[3])
        self.initial_target_obj_position = (5., 20.)  # (x0[4], x0[5])
        self.initial_target_obj_linear_velocity = (0., 0.)  # (x0[6], x0[7]),
        self.initial_angular_velocity = 0
        self.target_initial = (0, 0)

        ground = self.world.CreateBody(position=(0, 0))
        ground.CreateEdgeChain(
            [(-20, -20),
             (-20, 20),
             (20, 20),
             (20, -20),
             (-20, -20)]
        )

        xf1 = b2.b2Transform()
        xf1.angle = 0.3524 * b2.b2_pi
        xf1.position = b2.b2Mul(xf1.R, (1.0, 0.0))

        xf2 = b2.b2Transform()
        xf2.angle = -0.3524 * b2.b2_pi
        xf2.position = b2.b2Mul(xf2.R, (-1.0, 0.0))
        self.body = self.world.CreateDynamicBody(
            position=self.initial_position,
            angle=self.initial_angle,
            linearVelocity=self.initial_linear_velocity,
            angularVelocity=self.initial_angular_velocity,
            angularDamping=5,
            linearDamping=0.1,
            shapes=[b2.b2CircleShape(radius=1)],
            shapeFixture=b2.b2FixtureDef(density=1.0),
        )

        # Create piston

        # create piston head
        self.target_obj = self.world.CreateDynamicBody(
            position=(0, 0),
            angle=0,
            linearVelocity=self.initial_linear_velocity,
            angularVelocity=0,
            angularDamping=5,
            linearDamping=0.3,
            shapes=[b2.b2PolygonShape(box=(2, 1))],
            shapeFixture=b2.b2FixtureDef(density=1.0),
        )

        # piston head desired location
        self.target = self.world.CreateStaticBody(
            position=(0, -5),
            angle=0,
            shapes=[b2.b2PolygonShape(box=(2, 1))],
        )
        self.target.active = False
        # ghost body which IS active for piston
        self.target_ghost = self.world.CreateStaticBody(
            position=(0, -5),
        )

        # piston constraining walls
        eta = 0.1
        self.piston_walls = self.world.CreateBody(position=(0, 0))
        ground.CreateEdgeChain(
            [(-3, 1), (-3, -10), (3, -10), (3, 1), (2 + eta, 1), (2 + eta, -9), (-2 - eta, -9), (-2 - eta, 1), (-3, 1)])

        # constrain piston movement
        self.world.CreatePrismaticJoint(
            bodyA=self.target_obj,
            bodyB=self.target_ghost,
            anchor=(self.target.position + self.target_obj.position) / 2,
            axis=(0, 1),
            lowerTranslation=0.0,
            upperTranslation=5.0,
            enableLimit=True,
            # motorForce=1.0,
            motorSpeed=0.0,
            enableMotor=True,
        )
        # attach spring to piston head
        self.world.CreateDistanceJoint(
            bodyA=self.target_obj,
            bodyB=self.target_ghost,
            anchorA=self.target_obj.position,
            anchorB=self.target_ghost.position,
            frequencyHz=0.5,
            dampingRatio=0.8,
            collideConnected=True)

    def reset_world(self):
        # override parent settings in the case of GPS
        self.world.gravity = (0.0, 0.0)
        self.doSleep = True

        """ This resets the world to its initial state"""
        self.world.ClearForces()

        # reset manipulator
        self.body.position = self.initial_position
        self.body.angle = self.initial_angle
        self.body.angularVelocity = self.initial_angular_velocity
        self.body.linearVelocity = self.initial_linear_velocity

        # reset piston head
        self.target_obj.position = (0,0)
        self.target_obj.initial_angle = 0
        self.target_obj.angularVelocity = 0
        self.target_obj.linearVelocity = (0, 0)

        # reset piston head
        self.target.position = (0,0)
        self.target.initial_angle = 0
        self.target.angularVelocity = 0
        self.target.linearVelocity = (0, 0)

    def step(self, timestep, vel_iterations, pos_iterations):
        self.world.Step(timestep, vel_iterations, pos_iterations)

    def take_action(self, action):
        self.body.ApplyForce(force=(action[0], action[1]), point=self.body.position, wake=True)

    def get_state(self):
        return [self.body.position,
                self.body.linearVelocity,
                self.target_obj.position]
