import matplotlib.pyplot as plt
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
