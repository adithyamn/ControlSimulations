from casadi import *

# NaivePDSim.py - Simulation
'''
Some theory for energy shaping control for pendulum
Total Enegry E = 1/2ml^2thetadot^2 - mglcostheta

To understand how to control energy (or shape it) we take its derivative
'''
theta = SX.sym('theta')
thetadot = SX.sym('thetadot')
m = SX.sym('m')
l = SX.sym('l')
g = SX.sym('g')

# Total Energy

E = 0.5 * m * l**2 * thetadot**2 - m * g * l * cos(theta)

thetaddot = SX.sym('thetaddot')  # angular acceleration (control input u/moment)
Edot = gradient(E, theta) * thetadot + gradient(E, thetadot) * thetaddot

# Adding and Removing Energy into the system is simple based on the direction of torque
# E_dot = u theta_dot

print("Total Energy E:", E)
print("Time derivative of Energy dE/dt:", Edot)

# To Swing up the pendulum even with torque limits, we can use this observation to push the system towards its homoclinic orbit
# Homoclinic orbit - Leaves a saddle equilibrium and returns to it at T = inf

E_desired = m * g * l

# Considering a feedback controller of the form
# u = -k * thetadot * E

# This however is sensitive to the model parameters (l - length)
# In summary, the key idea of energy-shaping control for the pendulum is to directly influence its total mechanical energy.
# Rather than just the position or velocity from the NaivePD simulation
