import casadi as ca

# Problem setup
N = 50                     # number of intervals
x0 = [2.0 , 2.0]            # initial state [q, qdot]
nx = 2
nu = 1

# Optimization
opti = ca.Opti()

# Decision variables
X = opti.variable(nx, N+1)         # states
U = opti.variable(nu, N)           # controls
tf = opti.variable()               # final time

# Time step
dt = tf / N

# Dynamics function
def f(x, u):
    q, qdot = x[0], x[1]
    return ca.vertcat(qdot, u)     # dq = qdot,  dqdot = u

# Initial constraint
opti.subject_to(X[:,0] == x0)

# Final state constraint
opti.subject_to(X[:,N] == ca.DM.zeros(nx))

# Dynamics constraints: forward Euler
for k in range(N):
    xk = X[:,k]
    uk = U[:,k]
    x_next = xk + dt * f(xk, uk)
    opti.subject_to(X[:,k+1] == x_next)

# Control bound
opti.subject_to(opti.bounded(-1, U, 1))

# Final time constraints
opti.subject_to(tf >= 0.01) 
opti.minimize(tf)

# Solver
p_opts = {"expand": True}
s_opts = {"max_iter": 1000}
opti.solver("ipopt", p_opts, s_opts)

# Solve
sol = opti.solve()

# Extract solution
X_opt = sol.value(X)
U_opt = sol.value(U)
tf_opt = sol.value(tf)

print("Minimum Time from Start to End :", tf_opt)


