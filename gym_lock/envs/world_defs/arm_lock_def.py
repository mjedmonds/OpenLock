from Box2D import *
import numpy as np

from gym_lock.common import TwoDConfig, POS_PID_CLK_DIV
from gym_lock.pid_central import PIDController
from gym_lock.common import wrapToMinusPiToPi

FPS = 30


# TODO: cleaner interface than indices between bodies and lengths
# TODO: cleanup initialization/reset method
# NOTE: action spaces are different..

# class stateMachineListener(b2ContactListener):
#
#     def __init__(self):
#         b2ContactListener.__init__(self)
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

class ArmLockContactListener(b2ContactListener):
    
    def __init__(self):
        b2ContactListener.__init__(self)
    def BeginContact(self, contact):
        pass
    def EndContact(self, contact):
        pass
    def PreSolve(self, contact, oldManifold):
        print contact
    def PostSolve(self, contact, impulse):
        print contact
        print impulse
        exit()
    

class ArmLockDef(object):
    def __init__(self, chain, timestep, world_size):
        super(ArmLockDef, self).__init__()

        self.world = b2World(gravity=(0, -10),
                             doSleep=False,
                             contactListener=ArmLockContactListener())



        self.clock = 0
        self.timestep=timestep

        self.num_steps = 0
        self.start = np.array([0,0,0,0])
        self.target_arrow = None
        self.grasp = []

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
        base_fixture = b2FixtureDef(
            shape=b2PolygonShape(box=(1, 1)),
            density=100.0,
            friction=1.0,
            categoryBits=0x0001,
            maskBits=0x1110)

        # define link properties
        link_fixture = b2FixtureDef(  # all links have same properties
            density=1.0,
            friction=1.0,
            categoryBits=0x0001,
            maskBits=0x1110)
        # define end effector properties
        end_effector_fixture = b2FixtureDef(  # all links have same properties
            density=0.1,
            friction=1.0,
            categoryBits=0x0001,
            maskBits=0x1110,
            shape=b2CircleShape(radius=0.5))

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

            link_fixture.shape = b2PolygonShape(vertices=[(0, -width / 2),
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
        self.arm_bodies[-1].CreateFixture(end_effector_fixture)

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


        self.__init_door_lock()
        self.__init_cascade_controller()

        # TEST
        ball_fixture = b2FixtureDef(  # all links have same properties
            density=0.1,
            friction=1.0,
            shape=b2CircleShape(radius=1.5),
            categoryBits=0x0100,
            maskBits=0x1011,
        )

        self.world.CreateDynamicBody(
            position=(-10, 0),
            fixtures=ball_fixture,
            linearDamping=5.0
        )
        self.world.CreateDynamicBody(
            position=(0, -10),
            fixtures=ball_fixture,
            linearDamping=5.0

        )

        for body in self.world.bodies:
            body.bullet = True



    def draw_target_arrow(self, x, y, theta):
        if self.target_arrow:
            self.world.DestroyBody(self.target_arrow)
        self.target_arrow = self.world.CreateBody(position=(x, y),
                                     angle=theta,
                                     active=False,
                                     shapes=[b2PolygonShape(vertices=[(0, 0.25), (0, -0.25), (1, 0) ])])

    def __init_door_lock(self):

        # create door
        door_width = 0.5
        door_length = 10
        door_fixture = b2FixtureDef(
            shape=b2PolygonShape(vertices=[(0,-door_width),
                                             (0, door_width),
                                             (door_length, door_width),
                                             (door_length, -door_width)]),
            density=1,
            friction=1.0,
            categoryBits=0x0010,
            maskBits=0x1101)
        self.door = self.world.CreateDynamicBody(
            position = (15, 10),
            angle = -np.pi/2,
            fixtures = door_fixture,
            angularDamping = 0.5,
            linearDamping = 0.5)

        self.door_hinge = self.world.CreateRevoluteJoint(
            bodyA=self.door,  # end of link A
            bodyB=self.ground,  # beginning of link B
            localAnchorA=(0, 0),
            localAnchorB=(15, 10),
            enableMotor=True,
            motorSpeed=0,
            enableLimit=False,
            maxMotorTorque=20)
        self.door_lock = None
        # self.door_lock = self.world.CreateWeldJoint(
        #     bodyA=self.door,  # end of link A
        #     bodyB=self.ground,  # beginning of link B
        #     localAnchorB=(15, 5),
        # )


        lock_width = 0.5
        lock_length = 5
        lock_fixture = b2FixtureDef(
            shape=b2PolygonShape(vertices=[(0,-lock_width),
                                                    (0, lock_width),
                                                    (lock_length, lock_width),
                                                    (lock_length, -lock_width)]),
            density=1,
            friction=1.0,
            categoryBits=0x0010,
            maskBits=0x1101)

        self.lock = self.world.CreateDynamicBody(
            position = (15, -5),
            angle = -np.pi/2,
            fixtures = lock_fixture)
        self.lock_joint = self.world.CreatePrismaticJoint(
            bodyA=self.lock,
            bodyB=self.ground,
            anchor=(0, 0),
            axis=(1, 0),
            lowerTranslation=-2,
            upperTranslation=0,
            enableLimit = True
        )


        # test



    def __init_cascade_controller(self):
        pts = [c.theta for c in self.chain.get_rel_config()[1:]]
        # pts = [np.pi, -np.pi/2, np.pi/2, -np.pi/2, np.pi/2]
        # pts = [0] * len(self.arm_joints)
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



    def end_effector_grasp(self):
        if len(self.grasp) > 0:
            for connection in self.grasp:
                self.world.DestroyJoint(connection)
            self.grasp = []
        else:
            print '--------------------'

            for contact_edge in self.arm_bodies[-1].contacts:

                # print 'ours'
                # print contact_edge
                # print contact_edge.contact
                # for other_contact in contact_edge.other.contacts:
                #     print 'theirs'
                #     print other_contact
                #     print other_contact.contact

                # pointA, pointB, distance
                fix_A = contact_edge.contact.fixtureA
                fix_B = contact_edge.contact.fixtureB
                dist_result = b2Distance(shapeA=fix_A.shape,
                                            shapeB=fix_B.shape,
                                            transformA=fix_A.body.transform,
                                            transformB=fix_B.body.transform)

                point_A = fix_A.body.GetLocalPoint(dist_result.pointA)
                point_B = fix_B.body.GetLocalPoint(dist_result.pointB)

                self.grasp.append(self.world.CreateDistanceJoint(bodyA=fix_A.body,
                                                                 bodyB=fix_B.body,
                                                                 localAnchorA = point_A,
                                                                 localAnchorB = point_B,
                                                                 frequencyHz=0.5,
                                                                 dampingRatio=0.8,
                                                                 collideConnected=True
                                                              ))

            print '------end--------'

    def update_cascade_controller(self):
        # TODO: formalize clockrate
        if self.clock % POS_PID_CLK_DIV == 0:
            theta = [c.theta for c in self.get_rel_config()[1:]]
            vel_setpoints = self.pos_controller.update(theta)
            self.vel_controller.set_setpoint(vel_setpoints)

        joint_speeds = [joint.speed for joint in self.arm_joints]
        torques = self.vel_controller.update(joint_speeds)
        # if self.clock % 10 == 0:
        #     print self.clock
        #     print vel_setpoints
        #     print torques


        return torques

    def update_state_machine(self):
        if self.lock_joint.translation < -1.0 and self.door_lock:
            self.world.DestroyJoint(self.door_lock)
            self.door_lock = None
            # self.door_hinge.enableLimit = False

    def set_controllers(self, setpoints):
        new = [wrapToMinusPiToPi(c) for c in setpoints]
        self.pos_controller.set_setpoint(new)
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
        force = torque / self.arm_lengths[idx]
        # print torque
        # print self.arm_lengths[idx]

        angle = self.arm_bodies[idx].transform.angle
        position = self.arm_bodies[idx].position
        # print position

        yaxis = b2Vec2([-np.sin(angle), np.cos(angle)])
        # print yaxis
        force_vector = yaxis * force

        self.arm_bodies[idx].ApplyForce(force=force_vector, point=position, wake=True)

    def step(self, timestep, vel_iterations, pos_iterations):
        self.clock += 1
        # self.update_state_machine()

        # update torques
        new_torque = self.update_cascade_controller()
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
