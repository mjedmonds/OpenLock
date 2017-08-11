import numpy as np

# defined named tuples
from gym_lock.common import TwoDConfig, wrapToMinusPiToPi, transform_to_theta, clamp_mag

# turns on all assertions
DEBUG = True

# TODO: move elsewhere, or get rid of dict config and just pass in list of links?
def generate_valid_config(t1, t2, t3, t4):
    joint_config = [{'name' : '0-0'},
                    {'name' : '0+1-', 'theta' : t1, 'screw' : [0, 0, 0, 0, 0, 1]},
                    {'name' : '1-1+', 'x' : 5},
                    {'name' : '1+2-', 'theta' : t2, 'screw' : [0, 0, 0, 0, 0, 1]},
                    {'name' : '2-2+', 'x' : 5},
                    {'name' : '2+3-', 'theta' : t3, 'screw' : [0, 0, 0, 0, 0, 1]},
                    {'name' : '3-3+', 'x' : 5},
                    {'name': '3+4-', 'theta': t4, 'screw': [0, 0, 0, 0, 0, 1]},
                    {'name': '4-4+', 'x': 5}
                    ]
    return joint_config

def get_adjoint(transform):
    rot = transform[:3, :3]
    trans = transform[:3, 3]

    matrix_rep = np.array([[0, -trans[2], trans[1]],
                           [trans[2], 0, -trans[0]],
                           [-trans[1], trans[0], 0]])

    res = np.zeros((6, 6))
    res[:3, :3] = res[3:, 3:] = rot
    res[:3, 3:] = matrix_rep.dot(rot)
    return res

def discretize_path(cur, action, step_delta):

    # calculate number of discretized steps
    cur = cur.get_total_delta_config()
    targ = action.get_total_delta_config()

    delta = [t - c for t, c in zip(targ, cur)]

    num_steps = max([int(abs(d / step_delta)) for d in delta])

    if num_steps == 0:
        return None

    # generate discretized path
    waypoints = []
    for i in range(0, num_steps + 1):
        waypoints.append(KinematicLink(x=cur.x + i * delta[0] / num_steps,
                                     y=cur.y + i * delta[1] / num_steps,
                                     theta=wrapToMinusPiToPi(cur.theta + i * delta[2] / num_steps)))
    # sanity check: we actually reach the target config
    assert np.allclose(waypoints[-1].get_transform(), action.get_transform())

    return waypoints

class InverseKinematics(object):
    def __init__(self, kinematic_chain, target):
        self.kinematic_chain = kinematic_chain
        self.target = target

    def set_target(self, new_target):
        self.target = new_target

    def set_current_config(self, current_config):
        self.kinematic_chain = current_config

    def get_error_vec(self, clamp=False):
        err_mat = self.target.get_transform() \
                      .dot(np.linalg.inv(self.kinematic_chain.get_transform())) \
                  - np.eye(4)
        err_vec = np.zeros(6)
        err_vec[:3] = err_mat[:3, 3]
        err_vec[3] = err_mat[2, 2] + err_mat[2, 1] + err_mat[1, 1]
        err_vec[4] = err_mat[0, 0] + err_mat[0, 2] + err_mat[2, 2]
        err_vec[5] = err_mat[1, 1] + err_mat[1, 0] + err_mat[0, 0]

        if clamp:
            err_vec = clamp_mag(err_vec, clamp)

        return err_vec

    def get_error(self):
        return np.linalg.norm(self.get_error_vec())

    def get_delta_theta_dls(self, lam=3, clamp_err=False, clamp_theta=False):
        err = self.get_error_vec(clamp=clamp_err)
        jac = self.kinematic_chain.get_jacobian()
        jac_t = jac.transpose()
        dtheta = np.linalg.inv(jac_t.dot(jac) \
                               + (lam ** 2) * np.eye(jac.shape[1])).dot(jac_t).dot(err)
        if clamp_theta:
            dtheta = clamp_mag(dtheta, clamp_theta)

        return dtheta

    def get_delta_theta_trans(self, alpha=0.01, clamp_err=False, clamp_theta=False):
        err = self.get_error_vec(clamp=clamp_err)
        jacob = self.kinematic_chain.get_jacobian()
        dtheta = jacob.transpose().dot(err)
        dtheta = alpha * dtheta #/ max(1, np.linalg.norm(dtheta))

        if clamp_theta:
            dtheta = clamp_mag(dtheta, clamp_theta)

        return dtheta

class KinematicChain(object):
    def __init__(self, initial_configuration, cog_links=None):
        self.configuration = initial_configuration
        self.chain = []
        for link in self.configuration:
            self.chain.append(KinematicLink(**link))

        self.cog_links = cog_links
        self._check_rep()

    def _check_rep(self):
        if DEBUG:
            if self.cog_links:
                # there should be one cog_link for every link
                assert len(self.cog_links) == (len(self.chain) - 1) / 2

            # there should always be an odd number of links
            # 1 virtual link + 1 rot and 1 trans for every link
            assert len(self.chain) % 2 == 1

            # a chain always has 1 virutal link and 2 real links
            assert len(self.chain) >= 3


    def update_chain(self, new_config):
        assert len(new_config) * 2 - 1 == len(self.chain)

        # update baseframe
        self.chain[0].set_x(new_config[0].x)
        self.chain[0].set_y(new_config[0].y)
        self.chain[0].set_theta(new_config[0].theta)

        # update angles at each joint
        for i in range(1, len(new_config)):
            self.chain[2 * i - 1].set_theta(new_config[i].theta)

    def get_abs_config(self):
        total_transform = np.eye(4)
        link_locations = []
        theta = 0
        for link in self.chain:
            # print link.get_transform()
            total_transform = total_transform.dot(link.get_transform())
            if link.screw is None:
                # link is a translation
                theta = transform_to_theta(total_transform)
                theta = wrapToMinusPiToPi(theta)
                link_locations.append(TwoDConfig(total_transform[:2, 3][0], total_transform[:2, 3][1], theta))
        return link_locations

    def get_rel_config(self):
        link_locations = []
        base = self.chain[0].get_transform()
        theta = np.arccos(base[0, 0]) \
                * np.sign(np.arcsin(base[1, 0]))
        base_conf = TwoDConfig(base[0, 3], base[1, 3], theta)
        link_locations.append(base_conf)
        for i in range(1, len(self.chain)):
            if self.chain[i].screw is None:
                # i - 1 is static link
                # i is rotational joint
                trans = self.chain[i].get_transform()
                rot = self.chain[i - 1].get_transform()
                x = trans[0, 3]
                y = trans[1, 3]
                theta = transform_to_theta(rot)

                link_locations.append(TwoDConfig(x, y, theta))
        return link_locations

    def get_total_delta_config(self):
        trans = self.get_transform()
        x = trans[0, 3]
        y = trans[1, 3]
        theta = transform_to_theta(trans)
        return TwoDConfig(x, y, theta)


    def get_transform(self, name=None):
        total_transform = np.eye(4)
        i=0
        for link in self.chain:
            i = i +1
            total_transform = total_transform.dot(link.get_transform())
            if link.name == name:
                print 'break'
                break
            print i
        return total_transform

    def get_jacobian(self):
        transform = np.eye(4)
        jacobian = []
        for i in range(0, len(self.chain)):
            transform = transform.dot(self.chain[i].get_transform())
            if self.chain[i].screw is not None:
                # end of a link
                screw = self.chain[i].screw
                adj = get_adjoint(transform)
                jacobian_i = adj.dot(screw)
                jacobian.append(jacobian_i)
        return np.array(jacobian).transpose()


class KinematicLink(object):
    def __init__(self, theta=0, x=0, y=0, scale=1, screw=None, name='KinematicLink'):
        self.transform = np.asarray([[np.cos(theta), -np.sin(theta), 0, x],
                                     [np.sin(theta), np.cos(theta), 0, y],
                                     [0, 0, 1, 0],
                                     [0, 0, 0, scale]])
        self.theta = theta
        self.x = x
        self.y = y
        self.scale = scale
        self.screw = np.array(screw) if screw != None else None
        self.name = name

    def set_theta(self, theta):
        self.transform[:2, :2] = [[np.cos(theta), -np.sin(theta)],
                                  [np.sin(theta), np.cos(theta)]]
        self.theta = theta

    def set_x(self, x):
        self.transform[0, 3] = x
        self.x = x

    def set_y(self, y):
        self.transform[1, 3] = y
        self.y = y

    def set_scale(self, scale):
        self.transform[2, 3] = scale
        self.scale = scale

    def get_transform(self):
        return self.transform




def main():

    import matplotlib.pyplot as plt
    from Queue import Queue
    import time

    # params
    epsilon = 0.01
    i = 0
    delta_step = 0.5

    # setup
    plt.ion()
    current_chain = KinematicChain(generate_valid_config(0, 0, 0, 0))

    t_1minus_l1 = KinematicLink(x=2.5).get_transform()
    t_0_1minus = current_chain.get_transform(name='0+1-')
    print t_0_1minus

    t_0_l1 = t_0_1minus.dot(t_1minus_l1)
    print t_0_l1
    adj_l1_0 = np.linalg.inv(get_adjoint(t_0_l1))

    jac = current_chain.get_jacobian()
    print jac

    jac_1 = adj_l1_0.dot(jac)

    print jac_1

    # jac_b = ad
    # print jac_b


    exit()

    targ = KinematicChain(generate_valid_config(np.pi / 2, 0, 0, np.pi / 2))
    poses = discretize_path(current_chain, targ, delta_step)


    # initialize with target and current the same
    invk = InverseKinematics(current_chain, current_chain)

    print len(poses)

    for i in range(1, len(poses)):

        # get next waypoint
        next_waypoint = poses[i]
        # set inverse kinematics to have next waypoint
        invk.set_target(next_waypoint)

        # print next_waypoint.get_transform()
        # print poses[-1].get_transform()

        # while err > epsilon, converge
        err = invk.get_error() # prime the loop
        print 'converging'
        a = 0
        while (err > epsilon):
            a = a + 1
            # get delta theta
            d_theta = invk.get_delta_theta_dls(lam=20)

            # d_theta = invk.get_delta_theta_trans()
            
            # get current config
            cur_theta = [c.theta for c in invk.kinematic_chain.get_rel_config()[1:]] # ignore virtual base link

            # create new config
            new_theta = [cur + delta for cur, delta in zip(cur_theta, d_theta)]

            # update inverse kinematics model
            invk.set_current_config(KinematicChain(generate_valid_config(new_theta[0],
                                                                         new_theta[1],
                                                                         new_theta[2],
                                                                         new_theta[3])))

            # update err
            err = invk.get_error()

            if a > 500:
                print err

        print 'converged in {} iterations'.format(a)
        # converged on that waypoint

        # plot
        if i % 1 == 0:
            print i
            con = invk.kinematic_chain.get_abs_config()
            x = [c.x for c in con]
            y = [c.y for c in con]
            plt.plot(x, y)
            plt.xlim([-20, 20])
            plt.ylim([-20, 20])

            if i == len(poses) - 1:
                plt.pause(100)
            else:
                plt.pause(0.1)
                plt.cla()



if __name__ == "__main__":
    main()
