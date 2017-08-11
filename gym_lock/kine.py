from __future__ import division
import numpy as np

# defined named tuples
from gym_lock.common import TwoDConfig, wrapToMinusPiToPi, transform_to_theta, clamp_mag

# turns on all assertions
DEBUG = True


# TODO: move elsewhere, or get rid of dict config and just pass in list of links?
def generate_four_arm(t1, t2, t3, t4, length=5):
    return [KinematicLink(TwoDKinematicTransform(name='0+1-' ,theta=t1, screw=[0, 0, 0, 0, 0, 1]),
                          TwoDKinematicTransform(name='1_cog', x=length / 2),
                          TwoDKinematicTransform(name='1-1+', x=length),
                          np.eye(6)),

            KinematicLink(TwoDKinematicTransform(name='0+2-', theta=t2, screw=[0, 0, 0, 0, 0, 1]),
                          TwoDKinematicTransform(name='2_cog', x=length / 2),
                          TwoDKinematicTransform(name='2-2+', x=length),
                          np.eye(6)),

            KinematicLink(TwoDKinematicTransform(name='0+3-', theta=t3, screw=[0, 0, 0, 0, 0, 1]),
                          TwoDKinematicTransform(name='3_cog', x=length / 2),
                          TwoDKinematicTransform(name='3-3+', x=length),
                          np.eye(6)),

            KinematicLink(TwoDKinematicTransform(name='0+4-', theta=t4, screw=[0, 0, 0, 0, 0, 1]),
                          TwoDKinematicTransform(name='4_cog', x=length / 2),
                          TwoDKinematicTransform(name='4-4+', x=length),
                          np.eye(6))]


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
        waypoints.append(TwoDKinematicTransform(x=cur.x + i * delta[0] / num_steps,
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
        dtheta = alpha * dtheta  # / max(1, np.linalg.norm(dtheta))

        if clamp_theta:
            dtheta = clamp_mag(dtheta, clamp_theta)

        return dtheta

class KinematicChain(object):
    def __init__(self, base, chain):
        self.chain = chain
        self.base = base

        self._check_rep()

    def _check_rep(self):
        if DEBUG:
            # a chain always has at least 1 virutal link and 1 real links
            assert len(self.chain) >= 2

    def update_chain(self, new_config):
        assert len(new_config) == len(self.chain) + 1

        # update baseframe
        self.base.set_theta(new_config[0].theta)
        self.base.set_x(new_config[0].x)
        self.base.set_y(new_config[0].y)


        # update angles at each joint
        for link, conf in zip(self.chain, new_config):
            link.minus.set_theta(conf.theta)

    def get_abs_config(self):
        total_transform = self.base.get_transform()
        link_locations = []

        # add base
        link_locations.append(TwoDConfig(self.base.x, self.base.y, self.base.theta))

        # add arm links
        for link in self.chain:
            total_transform = total_transform.dot(link.minus.transform).dot(link.plus.transform)
            theta = transform_to_theta(total_transform)
            theta = wrapToMinusPiToPi(theta)
            link_locations.append(TwoDConfig(total_transform[:2, 3][0], total_transform[:2, 3][1], theta))
        return link_locations

    def get_rel_config(self):
        link_locations = []

        link_locations.append(TwoDConfig(self.base.x, self.base.y, self.base.theta))

        for link in self.chain:
            theta = link.minus.theta
            x = link.plus.x
            y = link.plus.y
            link_locations.append(TwoDConfig(x, y, theta))

        return link_locations

    def get_total_delta_config(self):
        total = self.get_transform()
        x = total[0, 3]
        y = total[1, 3]
        theta = transform_to_theta(total)
        return TwoDConfig(x, y, theta)

    def get_transform(self, name=None):
        total_transform = self.base.get_transform()
        for link in self.chain:
            total_transform = total_transform.dot(link.minus.transform).dot(link.plus.transform)
        return total_transform

    def get_jacobian(self):
        total_transform = self.base.get_transform()
        jacobian = []

        for link in self.chain:

            # transform to proximal end
            total_transform = total_transform.dot(link.minus.transform)

            if link.minus.screw is not None:
                # screw is on proximal end (i.e. revolute joint)

                # transform screw to inertial frame
                jacobian.append(get_adjoint(total_transform).dot(link.minus.screw))

            # transform to distal end
            total_transform = total_transform.dot(link.plus.transform)

            # if link.plus.screw is not None:
            #     
            #     # TODO: remove this, should never get here with current config?
            #     assert 1 == 0
            #     
            #     # screw is on distal end (i.e. translational joint)
            #     
            #     # transform screw to inertial frame
            #     jacobian.append(get_adjoint(total_transform).dot(link.plus.screw))

        res = np.array(jacobian).transpose()

        # jacobian should have a column for every link in kinematic chain (except virtual)
        # and row for each w_{x, y z} and v_{x, y, z}
        assert res.shape == (6, len(self.chain))

        return res

    def get_inertia_matrix(self):

        # compute body jacobians at each COG
        jacobians = [np.zeros((6, len(self.chain))) for i in range(0, len(self.chain))]
        for i in range(0, len(self.chain)):
            total_transform = np.eye(4)

            # get transform up to i'th frame
            jacobians[i][:, i] = np.linalg.inv(get_adjoint(total_transform.dot(self.chain[i].cog.transform))).dot(
                self.chain[i].minus.screw)

            # fill in i'th column of every jacobian
            for j in range(i + 1, len(self.chain)):

                total_transform = total_transform.dot(self.chain[i].plus.transform).dot(self.chain[i + 1].minus.transform)
                # T_ipjm

                jacobians[j][:, i] = np.linalg.inv(get_adjoint(total_transform.dot(self.chain[j].cog.transform))).dot(self.chain[i].minus.screw)


        return jacobians
class KinematicLink(object):
    def __init__(self, minus, cog, plus, inertia_matrix):
        self.minus = minus
        self.cog = cog
        self.plus = plus
        self.inertia_matrix = inertia_matrix

        self._check_rep()

    def _check_rep(self):
        assert self.minus.transform.shape == self.cog.transform.shape == self.plus.transform.shape == (4, 4)
        assert self.inertia_matrix.shape == (6, 6)


class TwoDKinematicTransform(object):
    def __init__(self, theta=0, x=0, y=0, scale=1, screw=None, name=None):
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
    base = TwoDKinematicTransform()
    current_chain = KinematicChain(base, generate_four_arm(0, 0, 0, 0))

    jacs = current_chain.get_inertia_matrix()
    print jacs
    exit()


    targ = KinematicChain(base, generate_four_arm(np.pi / 2, 0, 0, 0))
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
        err = invk.get_error()  # prime the loop
        print 'converging'
        a = 0
        while (err > epsilon):
            a = a + 1
            # get delta theta
            d_theta = invk.get_delta_theta_dls(lam=20)

            # d_theta = invk.get_delta_theta_trans()

            # get current config
            cur_theta = [c.theta for c in invk.kinematic_chain.get_rel_config()[1:]]  # ignore virtual base link

            # create new config
            new_theta = [cur + delta for cur, delta in zip(cur_theta, d_theta)]

            # update inverse kinematics model
            invk.set_current_config(KinematicChain(base, generate_four_arm(new_theta[0],
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

