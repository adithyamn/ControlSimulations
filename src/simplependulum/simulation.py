from casadi import DM
from dynamics import PendulumDynamics
from plot import PendulumPlot

# --- Parameters ---
dt = 0.01
T = 10.0
x0 = DM([0.7, 0.0])  # initial [theta, theta_dot]
u_val = 0          # torque

# --- Simulation ---
pendulum = PendulumDynamics(dt=dt)
X_np = pendulum.simulate(x0, u_val, T)
print(X_np)

#Plot
plotter = PendulumPlot(dt=dt)
#plotter.plot_trajectory(X_np, T)
plotter.plot_phase(X_np)
plotter.animate(X_np)
