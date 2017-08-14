# TODO prepend terms _
import numpy as np

from common import wrapToMinusPiToPi

class PIDController(object):
    def __init__(self, kp, ki, kd, setpoint, dt=1, max_out=2000, max_int=500):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint
        self.dt = dt
        self.max_out = max_out
        self.steps = 0

        self.previous_error = self.error = self.integral = \
            self.differential = self.previous_value = [0] * len(setpoint)

    def update(self, current_value):
        if not len(self.setpoint) == len(current_value):
            print self.setpoint
            print current_value
        self.error = [wrapToMinusPiToPi(s - c) for s, c in zip(self.setpoint, current_value)]
        self.integral = [(i + e) * self.dt for i, e in zip(self.integral, self.error)]
        self.differential = [(e - p) / self.dt for e, p in zip(self.error, self.previous_error)]
        p_term = [kp * e for kp, e in zip(self.kp, self.error)]
        i_term = [ki * e for ki, e in zip(self.ki, self.integral)]
        d_term = [kd * e for kd, e in zip(self.kd, self.differential)]

        self.previous_error = self.error

        # TODO: incorporate dynamics?
        return [p + i + d for p, i, d in zip(p_term, i_term, d_term)]


    def set_setpoint(self, setpoint):
        self.previous_error = self.error = self.integral = \
            self.differential = self.previous_value = [0] * len(setpoint)
        self.setpoint = setpoint

    def set_kp(self, kp):
        self.kp = kp

    def set_kd(self, kd):
        self.kd = kd

    def set_ki(self, ki):
        self.ki = ki

    def set_dt(self, dt):
        self.dt = dt

    def set_max_out(self, max_out):
        self.max_out = max_out
