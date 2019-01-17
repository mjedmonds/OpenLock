# TODO prepend terms _


class PIDController(object):
    def __init__(
        self, kp, ki, kd, setpoint, dt, max_out=None, max_int=500, err_wrap_func=None
    ):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint
        self.dt = dt
        self.max_out = max_out
        self.steps = 0
        self.error = 0
        self.err_wrap_func = err_wrap_func
        self.i_term = 0

        self.previous_error = (
            self.error
        ) = self.integral = self.differential = self.previous_value = [0] * len(
            setpoint
        )

    def update(self, current_value):
        if self.err_wrap_func:
            self.error = [
                self.err_wrap_func(s - c) for s, c in zip(self.setpoint, current_value)
            ]
        else:
            self.error = [s - c for s, c in zip(self.setpoint, current_value)]

        self.integral = [(i + e) * self.dt for i, e in zip(self.integral, self.error)]
        self.differential = [
            (e - p) / self.dt for e, p in zip(self.error, self.previous_error)
        ]

        p_term = [kp * e for kp, e in zip(self.kp, self.error)]
        i_term = [ki * e for ki, e in zip(self.ki, self.integral)]
        # i_term = [max(-100, min(i, 100)) for o in out]
        self.i_term = i_term

        d_term = [kd * e for kd, e in zip(self.kd, self.differential)]

        self.previous_error = self.error

        # TODO: incorporate dynamics?
        out = [p + i + d for p, i, d in zip(p_term, i_term, d_term)]

        if self.max_out:
            out = [max(-self.max_out, min(o, self.max_out)) for o in out]

        return out

    def set_setpoint(self, setpoint):
        self.previous_error = (
            self.error
        ) = self.integral = self.differential = self.previous_value = [0] * len(
            setpoint
        )
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
