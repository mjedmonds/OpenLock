import numpy as np
from collections import namedtuple

# defined named tuples
Config = namedtuple('Config', 'pos theta')

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

    def __init__(self, kinematic_chain, target, alpha=0.1, eta=0.001):
        self.kinematic_chain = kinematic_chain
        self.target = target
        self.alpha = alpha
        self.eta = eta

    def get_error(self):
        err_mat = self.kinematic_chain.get_transform() \
                       .dot(np.linalg.inv(self.target.get_transform())) \
                       - np.eye(4)
        err_vec = np.zeros(6)
        err_vec[:3] = err_mat[:3, 3]
        err_vec[3] = err_mat[2, 2] + err_mat[2, 1] + err_mat[1, 1]
        err_vec[4] = err_mat[0, 0] + err_mat[0, 2] + err_mat[2, 2]
        err_vec[5] = err_mat[1, 1] + err_mat[1, 0] + err_mat[0, 0]
        return err_vec

    def get_delta_theta(self):
        return self.kinematic_chain.get_jacobian().transpose().dot(self.get_error())

class KinematicChain(object):

    def __init__(self, initial_configuration):
        self.configuration = initial_configuration
        self.chain = []
        for link in self.configuration:
            self.chain.append(KinematicLink(**link))
    
    def get_link_config(self):
        total_transform = np.eye(4)
        link_locations = []
        for link in self.chain:
            total_transform = total_transform.dot(link.get_transform())
            if link.screw is None:
                # link is a translation
                link_locations.append(Config(total_transform[:2, 3],
                                             np.arccos(total_transform[0, 0])))
        return link_locations

    def get_transform(self):
        total_transform = np.eye(4)
        for link in self.chain:
            total_transform = total_transform.dot(link.get_transform())
        return total_transform

    def get_jacobian(self):
        transform = self.chain[0].get_transform()
        jacobian = []
        for i in range(1, len(self.chain)):
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
    joint_config = [{'name' : '0-0+'},
                    {'name' : '0+1-', 'theta' : 0, 'screw' : [0, 0, 0, 0, 0, 1]},
                    {'name' : '1-1+', 'x' : 1},
                    {'name' : '1+2-', 'theta' : np.pi / 2, 'screw' : [0, 0, 0, 0, 0, 1]}, 
                    {'name' : '2-2+', 'x' : 1},
                    {'name' : '2+3-', 'theta' : np.pi / 2, 'screw' : [0, 0, 0, 0, 0, 1]},
                    {'name' : '3-3+', 'x' : 1}]
    target_config = [{'name' : '0-0+'},
                {'name' : '0+1-', 'theta' : 0, 'screw' : [0, 0, 0, 0, 0, 1]},
                {'name' : '1-1+', 'x' : 1},
                {'name' : '1+2-', 'theta' : 0, 'screw' : [0, 0, 0, 0, 0, 1]}, 
                {'name' : '2-2+', 'x' : 1},
                {'name' : '2+3-', 'theta' : 0, 'screw' : [0, 0, 0, 0, 0, 1]},
                {'name' : '3-3+', 'x' : 1}]
    
    chain = KinematicChain(joint_config)
    target = KinematicChain(target_config)
    
    invk = InverseKinematics(chain, target, 0.1, 0.001)
    alpha = -0.5
    #TODO choose alpha
    print chain.get_link_config()

#    for i in range(0, 1000):
#        print chain.get_transform().dot(np.array([0,0,0,1]))
#        print np.linalg.norm((invk.get_error()))
#        delta_theta = invk.get_delta_theta()
#        chain.chain[1].set_theta(chain.chain[1].theta + alpha * delta_theta[0])
#        chain.chain[3].set_theta(chain.chain[3].theta + alpha * delta_theta[1])

if __name__ == "__main__":
    main()


