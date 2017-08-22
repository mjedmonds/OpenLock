import numpy as np
from Box2D import *



from gym_lock.common import TwoDConfig, TwoDForce
from gym_lock.common import wrapToMinusPiToPi
from gym_lock.pid_central import PIDController
from gym_lock.settings import BOX2D_SETTINGS


# TODO: cleaner interface than indices between bodies and lengths
# TODO: cleanup initialization/reset method

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

            # Every contact.points list has two points: true contact point in world coordinates
            # and (0,0), checking assumption that former is always head of list
            assert contact.worldManifold.points[0] != (0,0)


            # checking assumptions...cannot find documentation on how/why length of impulses
            # would be greater than 1
            assert len(impulse.normalImpulses) == len(impulse.tangentImpulses) == 1

            norm_imp = impulse.normalImpulses[0]
            tan_imp = impulse.tangentImpulses[0]


            if fixture_id == 'A':
                transform = contact.fixtureA.body.GetLocalPoint(contact.worldManifold.points[0])
                norm_vector = -manifold.normal
            else:
                # print contact.worldManifold.normal
                transform = contact.fixtureB.body.transform
                norm_vector = manifold.normal

            norm_vector = transform.R * norm_vector
            norm_vector.Normalize()
            tan_vector = b2Vec2(-norm_vector[1], norm_vector[0])

            norm_force_vector = (norm_imp / self.__timestep) * norm_vector
            tan_force_vector = (tan_imp / self.__timestep) * tan_vector

            self.__norm_force_vector = norm_force_vector
            self.__tan_force_vector = tan_force_vector

class ArmLockDef(object):
    def __init__(self, chain, timestep, world_size):
        super(ArmLockDef, self).__init__()

        self.timestep = timestep
        self.chain = chain

        self.world = b2World(gravity=(0, -10),
                             doSleep=False)

        self.clock = 0
        self.target_arrow = None
        self.grasped_list = []

        x0 = chain.get_abs_config()

        # create boundaries
        self.ground = self.world.CreateBody()
        self.ground.CreateEdgeChain([(-world_size, -world_size),
                                     (world_size, -world_size),
                                     (world_size, world_size),
                                     (-world_size, world_size),
                                     (-world_size, -world_size)])

        self.__init_arm(x0)
        self.__init_door_lock()
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
                maxMotorTorque=0,
                enableLimit=False))

    def draw_target_arrow(self, x, y, theta):
        if self.target_arrow:
            self.world.DestroyBody(self.target_arrow)
        self.target_arrow = self.world.CreateBody(position=(x, y),
                                                  angle=theta,
                                                  active=False,
                                                  shapes=[b2PolygonShape(vertices=[(0, 0.25), (0, -0.25), (1, 0)])])

    def __init_door_lock(self):
        # TODO: add relocking ability
        # create door
        door_width = 0.5
        door_length = 10
        door_fixture = b2FixtureDef(
            shape=b2PolygonShape(vertices=[(0, -door_width),
                                           (0, door_width),
                                           (door_length, door_width),
                                           (door_length, -door_width)]),
            density=1,
            friction=1.0,
            categoryBits=0x0010,
            maskBits=0x1101)

        self.door = self.world.CreateDynamicBody(
            position=(15, 10),
            angle=-np.pi / 2,
            fixtures=door_fixture,
            angularDamping=0.5,
            linearDamping=0.5)

        self.door_hinge = self.world.CreateRevoluteJoint(
            bodyA=self.door,  # end of link A
            bodyB=self.ground,  # beginning of link B
            localAnchorA=(0, 0),
            localAnchorB=(15, 10),
            enableMotor=True,
            motorSpeed=0,
            enableLimit=False,
            maxMotorTorque=20)

        self.door_lock = self.world.CreateWeldJoint(
            bodyA=self.door,  # end of link A
            bodyB=self.ground,  # beginning of link B
            localAnchorB=(15, 5),
        )

        lock_width = 0.5
        lock_length = 5
        lock_fixture = b2FixtureDef(
            shape=b2PolygonShape(vertices=[(0, -lock_width),
                                           (0, lock_width),
                                           (lock_length, lock_width),
                                           (lock_length, -lock_width)]),
            density=1,
            friction=1.0,
            categoryBits=0x0010,
            maskBits=0x1101)

        self.lock = self.world.CreateDynamicBody(
            position=(15, -5),
            angle=-np.pi / 2,
            fixtures=lock_fixture)

        self.lock_joint = self.world.CreatePrismaticJoint(
            bodyA=self.lock,
            bodyB=self.ground,
            anchor=(0, 0),
            axis=(1, 0),
            lowerTranslation=-2,
            upperTranslation=0,
            enableLimit=True
        )

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
        return {
            'END_EFFECTOR_POS': self.get_abs_config()[-1],
            'LOCK_STATE': self.door_lock != None,
            'END_EFFECTOR_FORCE' : TwoDForce(self.contact_listener.norm_force, self.contact_listener.tan_force)
        }

    def end_effector_grasp_callback(self):
        # NOTE: It's a little tricky to grab objects when you're EXACTLY
        # touching, instead, we compute the shortest distance between the two
        # shapes once the bounding boxes start to overlap. This let's us grab
        # objects which are close. See: http://www.iforce2d.net/b2dtut/collision-anatomy

        if len(self.grasped_list) > 0:
            print 'detatch!'
            # we are already holding something
            for connection in self.grasped_list:
                self.world.DestroyJoint(connection)
            self.grasped_list = []
        else:
            if len(self.arm_bodies[-1].contacts) > 0:
                print 'grab!'
                # grab all the things!
                for contact_edge in self.arm_bodies[-1].contacts:
                    fix_A = contact_edge.contact.fixtureA
                    fix_B = contact_edge.contact.fixtureB

                    # find shortest distance between two shapes
                    dist_result = b2Distance(shapeA=fix_A.shape,
                                             shapeB=fix_B.shape,
                                             transformA=fix_A.body.transform,
                                             transformB=fix_B.body.transform)

                    point_A = fix_A.body.GetLocalPoint(dist_result.pointA)
                    point_B = fix_B.body.GetLocalPoint(dist_result.pointB)

                    # TODO experiment with other joints
                    self.grasped_list.append(self.world.CreateDistanceJoint(bodyA=fix_A.body,
                                                                            bodyB=fix_B.body,
                                                                            localAnchorA=point_A,
                                                                            localAnchorB=point_B,
                                                                            frequencyHz=1,
                                                                            dampingRatio=1,
                                                                            collideConnected=True
                                                                            ))
            else:
                print 'nothing to grab!'

    def update_cascade_controller(self):
        if self.clock % BOX2D_SETTINGS['POS_PID_CLK_DIV'] == 0:
            theta = [c.theta for c in self.get_rel_config()[1:]]
            vel_setpoints = self.pos_controller.update(theta)
            self.vel_controller.set_setpoint(vel_setpoints)
        joint_speeds = [joint.speed for joint in self.arm_joints]
        return self.vel_controller.update(joint_speeds)

    # TODO: move to contact listener
    def update_state_machine(self):
        if self.lock_joint.translation < -1.0 and self.door_lock:
            self.world.DestroyJoint(self.door_lock)
            self.door_lock = None

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
