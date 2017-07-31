import sympy as sp
import numpy as np
from abc import ABCMeta, abstractmethod


class TwoDKinematicChain(object):

    def __init__(self, initial_configuration):
        self.configuration = initial_configuration
        self.chain = []
        for link in self.configuration:
            self.chain.append(TwoDKinematicLink(**link))

    def compute_total_transform(self):
        total_transform = np.eye(3)
        for link in self.chain:
            total_transform = total_transform.dot(link.get_transform())
        return total_transform

class TwoDKinematicLink(object):

    def __init__(self, theta=0, x=0, y=0, scale=1, name='KinematicLink'):
        self.transform = np.array([[np.cos(theta), -np.sin(theta), x],
                                  [np.sin(theta), np.cos(theta), y],
                                  [0, 0, scale]])
        self.theta = theta
        self.x = x
        self.y = y
        self.scale = scale
        self.name = name
        
    def set_theta(self, theta):
        self.transform[0:2, 0:2] = [[np.cos(theta), -np.sin(theta)],
                                    [np.sin(theta), np.cos(theta)]]
        self.theta = theta
        
    def set_x(self, x):
        self.transform[0, 2] = x
        self.x = x
    
    def set_y(self, y):
        self.transform[1, 2] = y
        self.y = y

    def set_scale(self, scale):
        self.transform[2, 2] = scale
        self.scale = scale

    def get_transform(self):
        return self.transform

joint_config = [{'name' : '0-0+'},
                {'name' : '0+1-', 'theta' : np.pi / 2},
                {'name' : '1-1+', 'x' : 10},
                {'name' : '1+2-', 'theta' : np.pi / 2}, 
                {'name' : '2-2+', 'x' : 20},
                {'name' : '2+3-', 'theta' : np.pi / 2},
                {'name' : '3-3+', 'x' : 30}]

chain = TwoDKinematicChain(joint_config)
transform = chain.compute_total_transform()
print transform.dot([0, 0, 1])

