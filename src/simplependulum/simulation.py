from casadi import DM
from dynamics import PendulumDynamics
from plot import PendulumPlot

import numpy as np

# Parameters
dt = 0.01
T = 10.0
x0 = DM([0.2, 0.0])  # initial [theta, theta_dot]
u_val = 0          # torque

pendulum = PendulumDynamics(dt=dt)

#Test Simulation
num_steps = int(T / dt) + 1
X_np = np.zeros((num_steps, 2))
x0 = np.array(x0).flatten()  
X_np[0, :] = x0

X_np[0, :] = np.array(x0)

for i in range(1, num_steps):
    theta, theta_dot = X_np[i-1, :]
    theta_ddot = (u_val / (pendulum.m * pendulum.l**2)) - (pendulum.g / pendulum.l) * np.sin(theta)
    X_np[i, 0] = theta + dt * theta_dot
    X_np[i, 1] = theta_dot + dt * theta_ddot

print("Simulated trajectory:")
print(X_np)
