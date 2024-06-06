import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from scipy.io import loadmat


# Define system matrices
AB9_0 = np.array(loadmat('AB_from_ball_kick_9_states_n0.mat')['AB'])
vine = np.array(loadmat('AB_from_ball_kick_3_states.mat')['vine'])[0, 0]
A = AB9_0[:9, :9]
B = AB9_0[:9, 9:]
AB = np.array(loadmat('AB_from_ball_kick_9_states.mat')['AB'])
AB = np.array(loadmat('AB_from_ball_kick_3_states.mat')['AB'])
Q = np.eye(9)  # State cost
R = np.eye(3)  # Input cost

# Time discretization
T = 10
dt = 0.1
N = int(T / dt)  # Number of time steps

# Initial and target states
x0 = np.array([[0],[0],[1.5],[0],[0],[0],[0],[0],[1.5-vine]]).squeeze()
x_target = np.array([[1.5],[0],[1.5],[0],[0],[0],[1.5],[0],[1.5-vine]]).squeeze()

# Define optimization variables
x = cp.Variable((9, N + 1))
u = cp.Variable((3, N))

# Define the cost function
cost = 0
constraints = [x[:, 0] == x0]
u_min = -2
u_max = 2

def calc(u, n=5, AB=None, vine=0, state_size=9):
    # Define initial positions
    if state_size == 9:
        x0 = np.array([[0], [0], [1.5], [0], [0], [0], [0], [0], [1.5 - vine]])
        
        # Check and set initial u
        u0 = np.array([[0], [0], [1.5]])
        if not np.all(np.equal(u[:, 0].value, u0.squeeze())):  # Use .value to extract NumPy array from cvxpy variable
            u = cp.hstack([u0, u])
        t = u.shape[1]
    elif state_size == 3:
        x0 = np.array([[0], [0], [1.5 - vine]])
        
        # Check and set initial u
        u0 = np.array([[0]])
        if not np.all(np.equal(u[0].value, u0.squeeze())):  # Use .value to extract NumPy array from cvxpy variable
            u = cp.hstack([u0, u])
        t = u.shape[1]
    
    # Set n history initial positions
    z0 = np.tile(x0, (n, 1)).flatten()
    z = cp.Variable((state_size * n, t))
    
    # Initialize x
    x = cp.Variable((state_size, t + 1))
    z[:, 0].value = z0
    x[:, 0].value = x0.flatten()
    
    for i in range(t):
        z_concat = cp.vstack([z[:, i], u[:, i]])
        x[:, i + 1] = AB @ z_concat
        z_next = cp.vstack([x[:, i + 1], z[:-(state_size), i]])
        z[:, i + 1] = z_next
    
    return x

data = loadmat('data_measured1.mat')
local_position = data['local_position']
local_setpoint = data['local_setpoint']
tip_pose = data['tip_pose']


# Parameters
num_skips = 1
train_size = 360

# Data slicing and resampling
local_position = local_position[::num_skips, :train_size]
local_setpoint = local_setpoint[::num_skips, :train_size]
tip_pose = tip_pose[::num_skips, :train_size]


# Constructing X_all and U_all
X_all = np.transpose(np.hstack((local_position[:, [1, 2, 3, 5, 6, 7]], 
                                tip_pose[:, 1:4])))
U_all = np.transpose(local_setpoint[:, 1:4])
x_init = X_all[:,164:164+N+2]
u_init = U_all[:,164:164+N+2]


for t in range(N):
    cost += cp.quad_form(x[:, t] - x0, Q)
    cost += cp.quad_form(x[:, t] - x_init[:, t], np.eye(9))  # Penalty for deviation from x_init
    cost += cp.quad_form(u[:, t] - u_init[:, t], np.eye(3))  # Penalty for deviation from u_init
    constraints += [x[:, t + 1] == A @ x[:, t] + B @ u[:, t]]
    # constraints += [x[:, t + 1] == calc(u, n=5, AB=AB, vine=vine, state_size=3)]
    constraints += [u[:, t] >= u_min, u[:, t] <= u_max] 
constraints += [x[:, -1] >= x_target-0.05, x[:, -1] <= x_target+0.05]
cost += cp.quad_form(x[:, -1] - x0, Q)
cost += cp.quad_form(x[:, int(N/2)] - x_target, Q)
cost += cp.quad_form(x[:, int(N/2 + 1)] - x_target, Q)
# Define and solve the optimization problem
problem = cp.Problem(cp.Minimize(cost), constraints)
problem.solve()

# Extract optimal control inputs and states
x_opt = x.value
print(x_opt[:,-1])
u_opt = u.value

plt.plot(range(N), u_opt[0, :], '-o')
plt.xlabel('State x1')
plt.ylabel('State x2')
plt.title('Optimal Trajectory of the LTI System')
plt.grid(True)
plt.show()

# Optionally, you can plot more states or other aspects of the solution
# For example, plotting all states over time:
# for i in range(9):
#     plt.plot(np.arange(N + 1) * dt, x_opt[i, :], label=f'x{i+1}')
# plt.xlabel('Time')
# plt.ylabel('States')
# plt.title('State Trajectories over Time')
# plt.legend()
# plt.grid(True)


plt.plot(np.arange(N + 1) * dt, x_opt[0, :], label=f'x{0}')
plt.plot(np.arange(N + 1) * dt, x_opt[6, :], label=f'x{6}')
plt.plot(np.arange(N + 1) * dt, x_opt[8, :], label=f'x{8}')
plt.xlabel('Time')
plt.ylabel('States')
plt.title('State Trajectories over Time')
plt.legend()
plt.grid(True)
plt.show()
