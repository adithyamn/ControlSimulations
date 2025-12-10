import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D


# pendulum simulation plots
class PendulumPlot:
    def __init__(self, l=1.0, dt=0.01):
        self.l = l
        self.dt = dt

    def plot_trajectory(self, X_np):
        time = np.arange(len(X_np)) * self.dt
        plt.figure(figsize=(10,4))
        plt.plot(time, X_np[:,0], label='theta (rad)')
        plt.plot(time, X_np[:,1], label='theta_dot (rad/s)')
        plt.xlabel('Time [s]')
        plt.ylabel('State')
        plt.title('Pendulum Trajectory')
        plt.grid(True)
        plt.legend()
        plt.show()

    def plot_phase(self, X_np):
        plt.figure(figsize=(8,6))
        plt.plot(X_np[:,0], X_np[:,1], color='red', lw=2)
        plt.xlabel(r'$\theta$ [rad]')
        plt.ylabel(r'$\dot{\theta}$ [rad/s]')
        plt.title('Pendulum Phase Plot')
        plt.grid(True)
        plt.show()

    def animate(self, X_np):
        theta_traj = X_np[:, 0]

        # Normalize angle so upright is visually correct
        theta_traj = ((theta_traj + np.pi) % (2*np.pi)) - np.pi

    # Pendulum coordinates
        x_pend = self.l * np.sin(theta_traj)
        y_pend = -self.l * np.cos(theta_traj)

        fig, ax = plt.subplots(figsize=(5, 5))

    # Axis limits so pendulum never gets cut off
        margin = 0.3 * self.l
        ax.set_xlim(-self.l - margin, self.l + margin)
        ax.set_ylim(-self.l - margin, self.l + margin)

        ax.set_aspect('equal')
        ax.grid(True)
        ax.set_title("Simple Pendulum - Swing Up Policy")
        ax.set_xticks([])
        ax.set_yticks([])
        
        rod, = ax.plot([], [], 'o-', lw=1, color='black') 
        bob, = ax.plot([], [], 'o',
                   markersize=18,
                   color='maroon')
        
        # Init function
        def init():
            rod.set_data([], [])
            bob.set_data([], [])
            return rod, bob

        # Update function
        def update(frame):
            rod.set_data([0, x_pend[frame]], [0, y_pend[frame]])
            bob.set_data([x_pend[frame]], [y_pend[frame]])
            return rod, bob

        ani = FuncAnimation(
            fig, update,
            frames=len(theta_traj),
            init_func=init,
            blit=True,
            interval=self.dt * 1000
        )

        plt.show()

# dp value-iteration plots
class DPPlot:

    def plot_value_function(self, J, theta_grid, theta_dot_grid):
        plt.figure(figsize=(8,6))
        plt.imshow(
            J.T,
            extent=[theta_grid[0], theta_grid[-1],
                    theta_dot_grid[0], theta_dot_grid[-1]],
            origin='lower',
            aspect='auto',
            cmap='viridis'
        )
        plt.colorbar(label="J")
        plt.xlabel(r'$\theta$')
        plt.ylabel(r'$\dot{\theta}$')
        plt.title("Value Function")
        plt.show()

    def plot_policy_heatmap(self, policy, theta_grid, theta_dot_grid):
        plt.figure(figsize=(8,6))
        plt.imshow(
            policy.T,
            extent=[theta_grid[0], theta_grid[-1],
                    theta_dot_grid[0], theta_dot_grid[-1]],
            origin='lower',
            aspect='auto',
            cmap='coolwarm'
        )
        plt.colorbar(label="u*")
        plt.xlabel(r'$\theta$')
        plt.ylabel(r'$\dot{\theta}$')
        plt.title("Optimal Policy")
        plt.show()

    def plot_value_contours_with_trajectory(self, J, theta_grid, theta_dot_grid, trajectory):
        TH, DTH = np.meshgrid(theta_grid, theta_dot_grid, indexing='ij')
        plt.figure(figsize=(8,6))
        plt.contour(TH, DTH, J.T, levels=40, cmap='viridis')
        plt.plot(trajectory[:,0], trajectory[:,1], 'r', lw=2)
        plt.xlabel(r'$\theta$')
        plt.ylabel(r'$\dot{\theta}$')
        plt.title("Trajectory on Value Function")
        plt.grid(True)
        plt.show()

    def plot_value_surface_3d(self, J, theta_grid, theta_dot_grid, trajectory=None):
        TH, DTH = np.meshgrid(theta_grid, theta_dot_grid, indexing='ij')

        fig = plt.figure(figsize=(10,7))
        ax = fig.add_subplot(111, projection='3d')

        surf = ax.plot_surface(
            TH, DTH, J,
            cmap='viridis',
            edgecolor='none',
            alpha=0.9
        )

        if trajectory is not None:
            th = trajectory[:, 0]
            dth = trajectory[:, 1]

            J_traj = np.zeros(len(th))
            for k in range(len(th)):
                i = np.searchsorted(theta_grid, th[k])
                j = np.searchsorted(theta_dot_grid, dth[k])
                i = np.clip(i, 0, len(theta_grid)-1)
                j = np.clip(j, 0, len(theta_dot_grid)-1)
                J_traj[k] = J[i, j]

            ax.plot(th, dth, J_traj, 'r', lw=3)

        ax.set_xlabel(r'$\theta$')
        ax.set_ylabel(r'$\dot{\theta}$')
        ax.set_zlabel('J')
        plt.title("Value Function Surface and Trajectory")
        fig.colorbar(surf, shrink=0.5, aspect=10)
        plt.show()