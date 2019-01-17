import numpy as np

# defined named tuples
from openlock.common import TwoDConfig, wrapToMinusPiToPi, transform_to_theta, clamp_mag
from openlock.settings_render import BOX2D_SETTINGS

# turns on all assertions
DEBUG = True


# TODO: move elsewhere, or get rid of dict config and just pass in list of links?
def generate_five_arm(t1, t2, t3, t4, t5):
    length = BOX2D_SETTINGS["ARM_LENGTH"]
    return [
        KinematicLink(
            TwoDKinematicTransform(name="0+1-", theta=t1, screw=[0, 0, 0, 0, 0, 1]),
            TwoDKinematicTransform(name="1_cog", x=length / 2),
            TwoDKinematicTransform(name="1-1+", x=length),
            None,
        ),
        KinematicLink(
            TwoDKinematicTransform(name="1+2-", theta=t2, screw=[0, 0, 0, 0, 0, 1]),
            TwoDKinematicTransform(name="2_cog", x=length / 2),
            TwoDKinematicTransform(name="2-2+", x=length),
            None,
        ),
        KinematicLink(
            TwoDKinematicTransform(name="2+3-", theta=t3, screw=[0, 0, 0, 0, 0, 1]),
            TwoDKinematicTransform(name="3_cog", x=length / 2),
            TwoDKinematicTransform(name="3-3+", x=length),
            None,
        ),
        KinematicLink(
            TwoDKinematicTransform(name="3+4-", theta=t4, screw=[0, 0, 0, 0, 0, 1]),
            TwoDKinematicTransform(name="4_cog", x=length / 2),
            TwoDKinematicTransform(name="4-4+", x=length),
            None,
        ),
        KinematicLink(
            TwoDKinematicTransform(name="4+5-", theta=t5, screw=[0, 0, 0, 0, 0, 1]),
            TwoDKinematicTransform(name="5_cog", x=length / 2),
            TwoDKinematicTransform(name="5-5+", x=length),
            None,
        ),
    ]


def get_adjoint(transform):
    rot = transform[:3, :3]
    trans = transform[:3, 3]

    matrix_rep = np.array(
        [[0, -trans[2], trans[1]], [trans[2], 0, -trans[0]], [-trans[1], trans[0], 0]]
    )

    res = np.zeros((6, 6))
    res[:3, :3] = res[3:, 3:] = rot
    res[:3, 3:] = matrix_rep.dot(rot)
    return res


def discretize_path(cur, targ, step_delta):
    # calculate number of discretized steps
    delta = [t - c for t, c in zip(targ, cur)]
    delta[-1] = wrapToMinusPiToPi(delta[-1])

    num_steps = max([int(abs(d / step_delta)) for d in delta])

    waypoints = []

    if num_steps == 0:
        # we're already within step_delta of our desired config in all dimensions
        return waypoints

    # generate discretized path
    for i in range(0, num_steps + 1):
        waypoints.append(
            TwoDKinematicTransform(
                x=cur.x + i * delta[0] / num_steps,
                y=cur.y + i * delta[1] / num_steps,
                theta=wrapToMinusPiToPi(cur.theta + i * delta[2] / num_steps),
            )
        )

    # sanity check: we actually reach the target config

    assert np.isclose(waypoints[-1].x, targ.x, rtol=1e-01, atol=1e-02)

    # TODO: handle +/- pi case
    # assert np.isclose(waypoints[-1].y, targ.y, rtol=1e-01, atol=1e-02)
    # if np.isclose(abs(waypoints[-1].theta), np.pi) or np.isclose(abs(targ.theta), np.pi):
    #     assert np.isclose(abs(waypoints[1].theta), abs(targ.theta), rtol=1e-01, atol=1e-02)
    # else:
    #     assert np.isclose(waypoints[-1].theta, targ.theta, rtol=1e-01, atol=1e-02)

    return waypoints


class InverseKinematics(object):
    def __init__(self, kinematic_chain, target):
        self.kinematic_chain = kinematic_chain
        self.target = target

    def set_current_config(self, current_config):
        self.kinematic_chain = current_config

    def get_error_vec(self, clamp=False):
        err_mat = self.target.get_transform().dot(
            np.linalg.inv(self.kinematic_chain.get_transform())
        ) - np.eye(4)
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
        dtheta = (
            np.linalg.inv(jac_t.dot(jac) + (lam ** 2) * np.eye(jac.shape[1]))
            .dot(jac_t)
            .dot(err)
        )
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
            assert len(self.chain) >= 1

    def update_chain(self, new_config):
        # len(new_config) should be equal to len(base config) + len(chain config)
        assert len(new_config) == len(self.chain) + 1

        # update baseframe
        self.base.theta = new_config[0].theta
        self.base.x = new_config[0].x
        self.base.y = new_config[0].y

        # update angles at each joint
        for link, conf in zip(self.chain, new_config[1:]):
            link.minus.theta = conf.theta

    def get_abs_config(self):
        total_transform = self.base.get_transform()
        link_locations = []

        # add base
        link_locations.append(TwoDConfig(self.base.x, self.base.y, self.base.theta))

        # add arm links
        for link in self.chain:
            total_transform = total_transform.dot(link.minus.transform).dot(
                link.plus.transform
            )
            theta = transform_to_theta(total_transform)
            theta = wrapToMinusPiToPi(theta)
            link_locations.append(
                TwoDConfig(total_transform[:2, 3][0], total_transform[:2, 3][1], theta)
            )

        return link_locations

    def get_rel_config(self):
        link_locations = []

        # add base
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

    def get_transform(self):
        total_transform = self.base.get_transform()
        for link in self.chain:
            total_transform = total_transform.dot(link.minus.transform).dot(
                link.plus.transform
            )
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

        res = np.array(jacobian).transpose()

        # jacobian should have a column for every link in kinematic chain (except virtual)
        # and row for each w_{x, y z} and v_{x, y, z}
        assert res.shape == (6, len(self.chain))

        return res

    def get_inertia_matrix(self):
        # one jacobian for every link in chain
        jacobians = [np.zeros((6, len(self.chain))) for i in range(0, len(self.chain))]

        # ith column of every jacobian shares preceding transforms, so for j jacobians,
        # fill in ith column
        for i in range(0, len(self.chain)):
            total_transform = np.eye(4)

            # get transform up to i'th frame
            jacobians[i][:, i] = np.linalg.inv(
                get_adjoint(total_transform.dot(self.chain[i].cog.transform))
            ).dot(self.chain[i].minus.screw)

            # fill in i'th column of every jacobian. Note that for jth link from base,
            # all columns > j are zero since more distant joints do not effect the jth link
            for j in range(i + 1, len(self.chain)):
                total_transform = total_transform.dot(self.chain[i].plus.transform).dot(
                    self.chain[i + 1].minus.transform
                )
                jacobians[j][:, i] = np.linalg.inv(
                    get_adjoint(total_transform.dot(self.chain[j].cog.transform))
                ).dot(self.chain[i].minus.screw)

        # finally sum J_i^T * M_i * J_i for all i to compute generalized inertia matrix
        ret = sum(
            [
                jacobian.transpose().dot(link.inertia_matrix).dot(jacobian)
                for link, jacobian in zip(self.chain, jacobians)
            ]
        )

        return ret


class KinematicLink(object):
    def __init__(self, minus, cog, plus, inertia_matrix, density=1):
        self.minus = minus
        self.cog = cog
        self.plus = plus
        self.density = density
        self.inertia_matrix = inertia_matrix

        self._check_rep()

    def _check_rep(self):
        assert (
            self.minus.transform.shape
            == self.cog.transform.shape
            == self.plus.transform.shape
            == (4, 4)
        )


class TwoDKinematicTransform(object):
    def __init__(self, theta=0, x=0, y=0, scale=1, screw=None, name=None):
        self.transform = np.asarray(
            [
                [np.cos(theta), -np.sin(theta), 0, x],
                [np.sin(theta), np.cos(theta), 0, y],
                [0, 0, 1, 0],
                [0, 0, 0, scale],
            ]
        )
        self.__theta = theta
        self.__x = x
        self.__y = y
        self.__scale = scale
        self.__screw = np.array(screw) if screw != None else None
        self.name = name

    @property
    def theta(self):
        return self.__theta

    @property
    def x(self):
        return self.__x

    @property
    def y(self):
        return self.__y

    @property
    def scale(self):
        return self.__scale

    @property
    def screw(self):
        return self.__screw

    @theta.setter
    def theta(self, theta):
        self.transform[:2, :2] = [
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
        ]
        self.__theta = theta

    @x.setter
    def x(self, x):
        self.transform[0, 3] = x
        self.__x = x

    @y.setter
    def y(self, y):
        self.transform[1, 3] = y
        self.__y = y

    @scale.setter
    def scale(self, scale):
        self.transform[2, 3] = scale
        self.__scale = scale

    @screw.setter
    def scale(self, screw):
        self.__screw = np.array(screw)

    def get_transform(self):
        return self.transform


def main():
    import openlock
    import matplotlib.pyplot as plt

    # params
    epsilon = 0.01
    i = 0
    delta_step = 0.5

    # setup
    plt.ion()
    base = TwoDKinematicTransform()
    current_chain = KinematicChain(base, generate_five_arm(0, 0, 0, 0))

    targ = KinematicChain(base, generate_five_arm(np.pi / 4, -np.pi / 4, 0, 0))
    poses = discretize_path(current_chain, targ, delta_step)

    # initialize with target and current the same
    invk = InverseKinematics(current_chain, current_chain)

    for i in range(1, len(poses)):

        # get next waypoint
        next_waypoint = poses[i]
        # set inverse kinematics to have next waypoint
        invk.target = next_waypoint

        # print next_waypoint.get_transform()
        # print poses[-1].get_transform()

        # while err > epsilon, converge
        err = invk.get_error()  # prime the loop
        print("converging")
        a = 0
        while err > epsilon:
            a = a + 1
            # get delta theta
            d_theta = invk.get_delta_theta_dls(lam=20)

            # d_theta = invk.get_delta_theta_trans()

            # get current config
            cur_theta = [
                c.theta for c in invk.kinematic_chain.get_rel_config()[1:]
            ]  # ignore virtual base link

            # create new config
            new_theta = [cur + delta for cur, delta in zip(cur_theta, d_theta)]

            # update inverse kinematics model
            invk.set_current_config(
                KinematicChain(
                    base,
                    generate_five_arm(
                        new_theta[0], new_theta[1], new_theta[2], new_theta[3]
                    ),
                )
            )

            # update err
            err = invk.get_error()

        print("converged in {} iterations".format(a))
        # converged on that waypoint

        # plot
        if i % 1 == 0:
            print(i)
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
