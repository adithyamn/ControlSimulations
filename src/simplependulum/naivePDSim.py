from casadi import DM
from dynamics import PendulumDynamics
from plot import PendulumPlot
import numpy as np

# First Step - Targeting upright

# Parameters
dt = 0.01
T = 100.0
x0 = DM([0.0, 0.0])  # initial [theta, theta_dot]

umax = 100.0         # torquelimit
Kp = 10.0
Kd = 2.0

# System 
pendulum = PendulumDynamics(dt=dt, b=0.5)
plotter = PendulumPlot(dt=dt)

# Simulation
N_steps = int(T/dt) + 1
x = x0.full().flatten()
trajectory = np.zeros((N_steps, 2))
trajectory[0, :] = x

for i in range(1, N_steps):
    theta, theta_dot = x

    # Naive PD controller toward upright
    u = -Kp * (theta - np.pi/2) - Kd * theta_dot
       
    # Torque enforcement
    if u > umax:
        u = umax
    elif u < -umax:
        u = -umax

    # Step
    x_next = pendulum.step(x, u)
    x = x_next.full().flatten()
    trajectory[i, :] = x

#Plot results
#plotter.plot_trajectory(trajectory, T)
plotter.plot_phase(trajectory)
plotter.animate(trajectory)
