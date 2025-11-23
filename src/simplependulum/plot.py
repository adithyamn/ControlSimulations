# plotting.py
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

class PendulumPlot:
    def __init__(self, l=1.0, dt=0.01):
        self.l = l
        self.dt = dt

    def plot_trajectory(self, X_np, T):
        time = np.arange(0, T+self.dt, self.dt)
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
        theta_traj = X_np[:,0]
        x_pend = self.l * np.sin(theta_traj)
        y_pend = -self.l * np.cos(theta_traj)

        fig, ax = plt.subplots(figsize=(5,5))
        ax.set_xlim(-self.l-0.2, self.l+0.2)
        ax.set_ylim(-self.l-0.2, 0.2)
        ax.set_aspect('equal')
        ax.grid(True)
        plt.title("Simple Pendulum - With Damping")
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        line, = ax.plot([], [], 'o-', lw=1)

        def init():
            line.set_data([], [])
            return line,

        def update(frame):
            line.set_data([0, x_pend[frame]], [0, y_pend[frame]])
            return line,

        ani = FuncAnimation(fig, update, frames=len(theta_traj),
                            init_func=init, blit=True, interval=self.dt*1000)
        plt.show()
