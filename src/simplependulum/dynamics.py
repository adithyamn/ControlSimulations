from casadi import MX, vertcat, integrator
import numpy as np

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

        # Dynamics (clean version)
        theta_ddot = (u / (self.m*self.l*self.l)) - (self.g/self.l)*MX.sin(theta)
        self.f = vertcat(theta_dot, theta_ddot)

        # RK4 integrator
        opts = {'tf': dt}
        self.integrator = integrator('I', 'rk', {'x': self.x, 'p': self.u, 'ode': self.f}, opts)

    def step(self, x_curr, u_val):
        res = self.integrator(x0=x_curr, p=u_val)
        return res['xf']

    def simulate(self, x0, u_val, T):
        N_steps = int(T / self.dt)
        x_curr = x0
        trajectory = [x_curr]
        for _ in range(N_steps):
            x_curr = self.step(x_curr, u_val)
            trajectory.append(x_curr)
        return np.array([xi.full().flatten() for xi in trajectory])
