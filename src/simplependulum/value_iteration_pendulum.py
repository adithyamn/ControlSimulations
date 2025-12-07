import numpy as np
from dynamics import PendulumDynamics

def wrap_angle(a):
    """Wrap angle to (-pi, pi]."""
    return (a + np.pi) % (2*np.pi) - np.pi

def stage_cost(theta, theta_dot, u,
               q_th=10.0, q_dth=1.0, r_u=0.1):
    """Quadratic cost for value iteration."""
    th_err = wrap_angle(theta - np.pi)  # target upright position
    return q_th*(th_err**2) + q_dth*(theta_dot**2) + r_u*(u**2)


dyn = PendulumDynamics(
    m=1.0,
    l=1.0,
    g=9.81,
    b=0.1,  # lighter damping for swing-up
    dt=0.05 # DP Timestep
)

# Discretize the state and input to grids for DP
N_th = 101
N_dth = 101

theta_grid = np.linspace(-np.pi, np.pi, N_th)
theta_dot_grid = np.linspace(-4.0, 4.0, N_dth)

# Torque inputs (discrete)
U = np.array([-6.0, 0.0, 6.0])

# DP arrays
J = np.zeros((N_th, N_dth))
policy = np.zeros((N_th, N_dth))

gamma = 0.98
MAX_ITERS = 500
TOL = 1e-3


# Value Iteration Loop

for iter in range(MAX_ITERS):
    print(f"Iteration {iter}")
    J_new = np.zeros_like(J)

    for i, th in enumerate(theta_grid):
        for j, dth in enumerate(theta_dot_grid):

            costs = []

            for u in U:
                x_next = dyn.step(np.array([th, dth]), u).full().flatten()
                th_next, dth_next = x_next

                # wrap the angle after RK4
                th_next = wrap_angle(th_next)

                # nearest grid indices
                i2 = np.searchsorted(theta_grid, th_next)
                j2 = np.searchsorted(theta_dot_grid, dth_next)

                # ensure within bounds
                i2 = np.clip(i2, 0, N_th-1)
                j2 = np.clip(j2, 0, N_dth-1)

                # Bellman update target
                Jn = J[i2, j2]
                c = stage_cost(th, dth, u) + gamma * Jn
                costs.append(c)

            # minimize over torque
            best = np.argmin(costs)
            J_new[i, j] = costs[best]
            policy[i, j] = U[best]

    diff = np.max(np.abs(J_new - J))
    print(f" diff = {diff}")

    J = J_new

    if diff < TOL:
        print("Converged.")
        break


np.save("src/simplependulum/controlLaw/J_value.npy", J)
np.save("src/simplependulum/controlLaw/policy.npy", policy)

print("Value Iteration Completed.")
