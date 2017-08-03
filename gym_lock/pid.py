#TODO prepend terms _
import time

class PIDController(object):

    def __init__(self, kp=10000, ki=5000, kd=2000, setpoint=0, dt=1):

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

        self.error = self.setpoint - current_value
        self.integral = self.integral + self.error * self.dt
        self.differential = (self.error - self.previous_error) / self.dt

        p_term = self.kp * self.error
        i_term = self.ki * self.integral
        d_term = self.kd * self.differential
      
        #print 'start'
        #print self.error
        #print p_term
        #print i_term
        #print d_term
        #print p_term + i_term + d_term
        #print '--------'
        return p_term + i_term + d_term

    def change_setpoint(self, setpoint):
        self.setpoint = setpoint
        self.error = 0
        self.integral = 0
        self.differential = 0
        self.previous_error = 0

    def set_kp(self, kp):
        self.kp = kp

    def set_kd(self, kd):
        self.kd = kd
    
    def set_ki(self, ki):
        self.ki = ki

    def set_dt(self, dt):
        self.dt = dt


        
