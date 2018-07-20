# TODO prepend terms _

from .common import wrapToMinusPiToPi


class PIDController(object):
    def __init__(self, kp=2500, ki=0, kd=200, setpoint=0, dt=1, max_out=2000):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint
        self.dt = dt
        self.max_out = max_out

        self.previous_error = 0
        self.error = 0
        self.integral = 0
        self.differential = 0
        self.previous_value = 0

    def update(self, current_value):
        self.error = wrapToMinusPiToPi(self.setpoint - current_value)
        self.integral = self.integral + self.error * self.dt
        self.differential = (self.error - self.previous_error) / self.dt

        p_term = self.kp * self.error
        i_term = self.ki * self.integral
        d_term = self.kd * self.differential

        out = p_term + i_term + d_term

        # clamp_mag range
        out = max(-self.max_out, min(out, self.max_out))

        self.previous_error = self.error

        print(out)
        print(self.setpoint)
        return out

    def set_setpoint(self, setpoint):
        self.previous_error = 0
        self.error = 0
        self.integral = 0
        self.differential = 0
        self.previous_value = 0
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
