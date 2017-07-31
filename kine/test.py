import numpy as np
from abc import ABCMeta, abstractmethod

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

class TwoDKinematicChain(object):

    def __init__(self, initial_configuration):
        self.configuration = initial_configuration
        self.chain = []
        for link in self.configuration:
            self.chain.append(TwoDKinematicLink(**link))

    def compute_total_transform(self):
        total_transform = np.eye(4)
        for link in self.chain:
            total_transform = total_transform.dot(link.get_transform())
        return total_transform

    def compute_jacobian(self):
        transform = self.chain[0].get_transform()
        jacobian = []
        for i in range(1, len(self.chain)):
            transform = transform.dot(self.chain[i].get_transform())
            if self.chain[i - 1].screw is not None:
                # end of a link
                screw = self.chain[i - 1].screw
                adj = get_adjoint(transform)
                jacobian_i = adj.dot(screw)
                jacobian.append(jacobian_i)
        return np.array(jacobian).transpose()

class TwoDKinematicLink(object):

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
        print self.screw
        self.name = name
        
    def set_theta(self, theta):
        self.transform[:2, :2] = np.asarray([[np.cos(theta), -np.sin(theta)],
                                    [np.sin(theta), np.cos(theta)]])
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

joint_config = [{'name' : '0-0+'},
                {'name' : '0+1-', 'theta' : np.pi / 4, 'screw' : [0, 0, 0, 0, 0, 1]},
                {'name' : '1-1+', 'x' : 1},
                {'name' : '1+2-', 'theta' : np.pi / 4, 'screw' : [0, 0, 0, 0, 0, 1]}, 
                {'name' : '2-2+', 'x' : 1}]
#                {'name' : '2+3-', 'theta' : np.pi / 2, 'screw' : [0, 0, 0, 0, 0, 1]},
#                {'name' : '3-3+', 'x' : 30}]

chain = TwoDKinematicChain(joint_config)
print chain.compute_jacobian()
transform = chain.compute_total_transform()
print transform.dot(np.asarray([0, 0, 0, 1]))

