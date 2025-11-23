from casadi import MX, vertcat, integrator
import numpy as np

class PendulumDynamics:
    def __init__(self, m=1.0, l=1.0, g=9.81,b=1.5,dt=0.01):
        self.m = m
        self.l = l
        self.g = g
        self.b = b
        self.dt = dt

        # States
        theta = MX.sym('theta')
        theta_dot = MX.sym('theta_dot')
        self.x = vertcat(theta, theta_dot)

        # Input torque
        u = MX.sym('u')
        self.u = u

        # Dynamics
        theta_ddot = (-self.b*theta_dot + u / (self.m*self.l*self.l)) - (self.g/self.l)*MX.sin(theta)
        self.f = vertcat(theta_dot, theta_ddot)

        # RK4 integrator
        opts = {'tf': dt}
        self.integrator = integrator('I', 'rk', {'x': self.x, 'p': self.u, 'ode': self.f}, opts)

    def step(self, x_curr, u_val):
        res = self.integrator(x0=x_curr, p=u_val)
        return res['xf']

    def simulate(self, x0, u_val, T):
        N_steps = int(T / self.dt) + 1
        trajectory = np.zeros((N_steps, 2))  # pre-allocate array
        trajectory[0, :] = np.array(x0).flatten()  # ensure 1D array

        for i in range(1, N_steps):
            next_state = self.step(trajectory[i-1, :], u_val)
            trajectory[i, :] = next_state.full().flatten()

        return trajectory
