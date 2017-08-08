import numpy as np

# defined named tuples
from gym_lock.common import TwoDConfig, wrapToMinusPiToPi

def generate_valid_config(t1, t2, t3):
    joint_config = [{'name' : '0-0'},
                    {'name' : '0+1-', 'theta' : t1, 'screw' : [0, 0, 0, 0, 0, 1]},
                    {'name' : '1-1+', 'x' : 5},
                    {'name' : '1+2-', 'theta' : t2, 'screw' : [0, 0, 0, 0, 0, 1]},
                    {'name' : '2-2+', 'x' : 5},
                    {'name' : '2+3-', 'theta' : t3, 'screw' : [0, 0, 0, 0, 0, 1]},
                    {'name' : '3-3+', 'x' : 5}]
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


class InverseKinematics(object):
    def __init__(self, kinematic_chain, target, alpha=0.01, lam=25):
        self.kinematic_chain = kinematic_chain
        self.target = target
        self.alpha = alpha
        self.lam = lam

    def set_target(self, new_target):
        self.target = new_target

    def set_current_config(self, current_config):
        self.kinematic_chain = current_config

    def get_error(self):
        err_mat = self.target.get_transform() \
                      .dot(np.linalg.inv(self.kinematic_chain.get_transform())) \
                  - np.eye(4)
        err_vec = np.zeros(6)
        err_vec[:3] = err_mat[:3, 3]
        err_vec[3] = err_mat[2, 2] + err_mat[2, 1] + err_mat[1, 1]
        err_vec[4] = err_mat[0, 0] + err_mat[0, 2] + err_mat[2, 2]
        err_vec[5] = err_mat[1, 1] + err_mat[1, 0] + err_mat[0, 0]
        return err_vec

    def get_delta_theta(self):
        # jacob = self.kinematic_chain.get_jacobian()
        # dtheta = jacob.transpose().dot(self.get_error())
        # dtheta = self.alpha * dtheta #/ max(1, np.linalg.norm(dtheta))


        err = self.get_error()
        jac = self.kinematic_chain.get_jacobian()
        jac_t = jac.transpose()
        dtheta = np.linalg.inv(jac_t.dot(jac) \
                               + (self.lam ** 2) * np.eye(jac.shape[1])).dot(jac_t).dot(err)




        return dtheta


class KinematicChain(object):
    def __init__(self, initial_configuration):
        self.configuration = initial_configuration
        self.chain = []
        for link in self.configuration:
            self.chain.append(KinematicLink(**link))

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
                theta = np.arccos(total_transform[0, 0]) \
                        * np.sign(np.arcsin(total_transform[1, 0]))
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
                theta = np.arccos(rot[0, 0]) \
                        * np.sign(np.arcsin(rot[1, 0]))
                link_locations.append(TwoDConfig(x, y, theta))
        return link_locations


    def get_transform(self):
        total_transform = np.eye(4)
        for link in self.chain:
            total_transform = total_transform.dot(link.get_transform())
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
    import time

    # delta = [-np.pi/3, np.pi/4, np.pi/2]
    # start = [np.pi/4, np.pi/4, 0]
    # start = [wrapToMinusPiToPi(s + d) for s,d in zip(start, delta)]
    # delta = [-np.pi, -np.pi, np.pi/2]
    # print [wrapToMinusPiToPi(s + d) for s,d in zip(start, delta)]

    delta = [-np.pi/4, np.pi, np.pi/2]
    start = [np.pi/4, -np.pi, -np.pi/2]

    poses = dict()
    leng = 10000
    for i in range(0, leng):
        poses[i] = generate_valid_config(wrapToMinusPiToPi(start[0] + delta[0] * 1.0 * i / leng),
                                         wrapToMinusPiToPi(start[1] + delta[1] * 1.0 * i / leng),
                                         wrapToMinusPiToPi(start[2] + delta[2] * 1.0 * i / leng))
    #
    # start = [s + d for s,d in zip(start, delta)]
    # delta = [-np.pi, -np.pi, np.pi/2]
    # print [s + d for s,d in zip(start, delta)]
    #
    # for i in range(0, leng):
    #     poses[i + leng] = generate_valid_config(start[0] + delta[0] * 1.0 * i / leng,
    #                                      start[1] + delta[1] * 1.0 * i / leng,
    #                                      start[2] + delta[2] * 1.0 * i / leng)

    plt.ion()

    chain = KinematicChain(poses[0])


    for i in range(1, leng):

        targ = KinematicChain(poses[i])
        invk = InverseKinematics(chain, targ)

        if i % 500 == 0:
            con = invk.kinematic_chain.get_abs_config()
            x = [c.x for c in con]
            y = [c.y for c in con]

            plt.plot(x, y)
            plt.xlim([-15, 15])
            plt.ylim([-15, 15])
            plt.pause(0.001)
            # plt.cla()



        dtheta = invk.get_delta_theta()
        # print 'dtheta'
        # print dtheta

        cur = [c.theta for c in invk.kinematic_chain.get_rel_config()[1:]]
        new = [dt + c for dt, c in zip(dtheta, cur)]
        # print 'cur'
        # print cur
        # print 'new'
        # print new

        new_config = generate_valid_config(new[0], new[1], new[2])

        chain = KinematicChain(new_config)

        invk.set_current_config(chain)
        print i
        if i == leng - 1:
            con = invk.kinematic_chain.get_abs_config()
            x = [c.x for c in con]
            y = [c.y for c in con]

            plt.plot(x, y)
            plt.xlim([-15, 15])
            plt.ylim([-15, 15])
            plt.title('0 0 0 to 45 45 45, dls')
            plt.pause(0.001)
            time.sleep(100)
        # print np.linalg.norm(invk.get_error())







if __name__ == "__main__":
    main()
