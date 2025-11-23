from casadi import MX, vertcat, integrator


class PendulumDynamics:
    def __init__(self, m=1.0, l=1.0, g=9.81,dt=0.01):
        self.m = m
        self.l = l
        self.g = g
        self.dt = dt

        # States
        theta = MX.sym('theta')
        theta_dot = MX.sym('theta_dot')
        self.x = vertcat(theta, theta_dot)

        # Input torque
        u = MX.sym('u')
        self.u = u

        # Dynamics
        theta_ddot = (u / (m*l*l)) - (g/l)*MX.sin(theta)
        self.f = vertcat(theta_dot, theta_ddot)

