import numpy as np
from dynamics import PendulumDynamics
from plot import PendulumPlot, DPPlot


# Load State Feedback Law
J = np.load("src/simplependulum/controlLaw/J_value.npy")
policy = np.load("src/simplependulum/controlLaw/policy.npy")

theta_grid = np.linspace(-np.pi, np.pi, J.shape[0])
theta_dot_grid = np.linspace(-4.0, 4.0, J.shape[1])

def wrap_angle(angle):
    return (angle + np.pi) % (2*np.pi) - np.pi

# Input lookup from discretized grid
def select_input(theta, theta_dot):
    i = np.searchsorted(theta_grid, wrap_angle(theta))
    j = np.searchsorted(theta_dot_grid, theta_dot)
    i = np.clip(i, 0, J.shape[0]-1)
    j = np.clip(j, 0, J.shape[1]-1)
    return policy[i, j]


# Define Dynamics
dyn = PendulumDynamics(b=0.1, dt=0.05)

T = 30.0
steps = int(T / dyn.dt)

# Start from downward position
x = np.array([0.0, 0.0])
trajectory = []

for k in range(steps):
    trajectory.append(x.copy())
    u = float(select_input(x[0], x[1]))
    x = dyn.step(x, u).full().flatten()

trajectory = np.array(trajectory)

# Plot
plotHelper = PendulumPlot(l=1.0, dt=dyn.dt)

plotHelper.plot_trajectory(trajectory)
plotHelper.plot_phase(trajectory)
plotHelper.animate(trajectory)

dp_plotHelper = DPPlot()

dp_plotHelper.plot_value_function(J, theta_grid, theta_dot_grid)
dp_plotHelper.plot_policy_heatmap(policy, theta_grid, theta_dot_grid)
dp_plotHelper.plot_value_contours_with_trajectory(J, theta_grid, theta_dot_grid, trajectory)
dp_plotHelper.plot_value_surface_3d(J, theta_grid, theta_dot_grid, trajectory)



