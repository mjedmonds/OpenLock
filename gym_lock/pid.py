#TODO prepend terms _

class PIDController(object):

    def __init__(self, kp=2.0, ki=0.5, kd=1.0, setpoint=0, dt=1):

        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint
        self.dt = dt

        self.previous_error = 0
        self.error = 0 
        self.integral = 0
        self.differential = 0

    
    def update(self, current_value):
        
        self.previous_error = self.error

        error = self.set_point - current_value
        self.integral = self.self.integral + self.error * dt
        self.differential = (self.error - self.previous_error) / self.dt

        p_term = self.kp * self.error
        i_term = self.ki * self.integral
        d_term = self.kd * self.differential

        return p_term + i_term + d_term

    def change_setpoint(self, setpoint):
        self.setpoint = setpoint
        self.error = 0
        self.integral = 0
        self.differential = 0

    def set_kp(self, kp):
        self.kp = kp

    def set_kd(self, kd):
        self.kd = kd
    
    def set_ki(self, ki):
        self.ki = ki

    def set_dt(self, dt):
        self.dt = dt


        
