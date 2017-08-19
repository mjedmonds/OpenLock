from collections import namedtuple

import numpy as np

TwoDConfig = namedtuple('Config', 'x y theta')
Color = namedtuple('Color', 'r g b')

FPS = 500
RENDER_CLK_DIV = 10
POS_PID_CLK_DIV = 10
VIEWPORT_W = 800
VIEWPORT_H = 800
SCALE = 15.0

# map [-inf, inf] -> [-pi, pi]
# note that pi -> -pi
def wrapToMinusPiToPi(original):
    return (original + np.pi) % (2 * np.pi) - np.pi

def transform_to_theta(transform):
    return np.arccos(transform[0, 0]) * np.sign(np.arcsin(transform[1, 0]))

def clamp_mag(array_like, clamp_mag):
    for i in range(0, len(array_like)):
        if abs(array_like[i]) > clamp_mag:
            array_like[i] = clamp_mag * np.sign(array_like[i])
    return array_like
