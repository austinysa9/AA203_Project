import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.io import savemat
import os
from scipy.signal import savgol_filter
# from filterpy.kalman import UnscentedKalmanFilter as UKF
# from filterpy.kalman import MerweScaledSigmaPoints


# Define system matrices
# Get the directory of the current script
current_directory = os.path.dirname(os.path.abspath(__file__))
# Define the relative path to your .mat file
# AB_file = os.path.join(current_directory, 'AB_from_pipe_enter_9_states.mat')
# data_file = os.path.join(current_directory, 'data_measured_pipe.mat')

# AB_file = os.path.join(current_directory, 'AB_from_ball_kick_9_states.mat')
# data_file = os.path.join(current_directory, 'data_measured1.mat')

# AB9_0 = np.array(loadmat("AB_from_ball_kick_9_states_n0.mat")['AB'])
# vine = np.array(loadmat('AB_from_ball_kick_3_states.mat')['vine'])[0, 0]
# A = AB9_0[:9, :9]
# B = AB9_0[:9, 9:]
# AB = np.array(loadmat('AB_from_ball_kick_9_states.mat')['AB'])
# AB = np.array(loadmat(AB_file)['AB'])
# vine = np.array(loadmat(AB_file)['vine'])[0, 0]

AB = np.array(loadmat('C:/Users/98502/OneDrive/Documents/GitHub/flyingSysID/Spring24_AY/AB_from_pipe_enter_9_states.mat')['AB'])
vine = np.array(loadmat('C:/Users/98502/OneDrive/Documents/GitHub/flyingSysID/Spring24_AY/AB_from_pipe_enter_9_states.mat')['vine'])[0, 0]
print(AB.shape)
# AB = np.array(loadmat('AB_from_ball_kick_3_states.mat')['AB'])
Q = np.eye(9)  # State cost
R = np.eye(3)  # Input cost

# Time discretization
T = 10
dt = 0.1
N = int(T / dt)  # Number of time steps
N = 310-151

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

# _1 for ball kick
# data = loadmat(data_file)
data = loadmat('C:/Users/98502/OneDrive/Documents/GitHub/flyingSysID/Spring24_AY/data_measured_pipe.mat')
local_position = data['local_position']
local_setpoint = data['local_setpoint']
tip_pose = data['tip_pose']

# Parameters
num_skips = 1
train_size = 340

# Data slicing and resampling
local_position = local_position[::num_skips, :train_size]
local_setpoint = local_setpoint[::num_skips, :train_size]
tip_pose = tip_pose[::num_skips, :train_size]


# Constructing X_all and U_all
X_all = np.transpose(np.hstack((local_position[:, [1, 2, 3, 5, 6, 7]], 
                                tip_pose[:, 1:4])))
U_all = np.transpose(local_setpoint[:, 1:4])
x_init = X_all[:,151:310]
u_init = U_all[:,151:310]




Q1 = [
    [10, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 10, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 10],
]

n = 9
state_history_size = 5
for t in range(N):
    if t >= 4:
        # z_ = cp.vstack([x[:, t], x[:, t-1], x[:, t-2], x[:, t-3], x[:, t-4], u[:, t]])
        z_ = z_ = cp.vstack([
        cp.reshape(x[:,t], (9, 1)),
        cp.reshape(x[:, t-1], (9, 1)),
        cp.reshape(x[:, t-2], (9, 1)),
        cp.reshape(x[:, t-3], (9, 1)),
        cp.reshape(x[:, t-4], (9, 1)),
        cp.reshape(u[:, t], (3, 1))
    ])
        z_next = AB @ z_
        # print(z_next.shape)
        z_next = cp.reshape(z_next, (9,))
        constraints += [x[:, t + 1] == z_next]
        constraints += [cp.norm(u[0:3, t - 1] - u[0:3, t]) <= 0.1]
        if t>=20:
            cost += cp.quad_form(x[:, t] - x_init[:, t], Q1)  # Penalty for deviation from x_init
    #cost += cp.quad_form(u[:, t] - u_init[:, t], np.eye(3))  # Penalty for deviation from u_init
    # constraints += [x[:, t + 1] == calc(u, n=5, AB=AB, vine=vine, state_size=3)]
    constraints += [u[:, t] >= u_min, u[:, t] <= u_max]
    constraints += [u[2, t] >= 1.4, u[2, t] <= u_max]
constraints += [x[:, 0] == x0]
constraints += [x[:, 1] == x0]
constraints += [x[:, 2] == x0]
constraints += [x[:, 3] == x0]
constraints += [x[:, 4] == x0]
# constraints += [x[:, -1] >= x_target-0.05, x[:, -1] <= x_target+0.05]
# cost += cp.quad_form(x[:, -1] - x0, Q)
# cost += cp.quad_form(x[:, int(N/2)] - x_target, Q)
# cost += cp.quad_form(x[:, int(N/2 + 1)] - x_target, Q)
# Define and solve the optimization problem
problem = cp.Problem(cp.Minimize(cost), constraints)
problem.solve()

# Extract optimal control inputs and states
x_opt = x.value
u_opt = u.value

u_opt = np.array(u_opt)
savemat('Test_0604_2.mat', {'u_opt': u_opt})

# Apply Savitzky-Golay filter
# u_opt_filtered = savgol_filter(u_opt, window_length=11, polyorder=2, axis=1)
u_opt_filtered = savgol_filter(u_opt, window_length=15, polyorder=2, axis=1)

# Segmented Savitzky-Golay filter
u_opt_filtered = np.zeros_like(u_opt)
u_opt_filtered[:, :8] = savgol_filter(u_opt[:, :8], window_length=8, polyorder=2, axis=1)
u_opt_filtered[:, 8:29] = savgol_filter(u_opt[:, 8:29], window_length=10, polyorder=2, axis=1)
u_opt_filtered[:, 29:100] = savgol_filter(u_opt[:, 29:100], window_length=30, polyorder=2, axis=1)
u_opt_filtered[:, 100:] = savgol_filter(u_opt[:, 100:], window_length=20, polyorder=1, axis=1)

# Calculate filtered trajectory
# x_filtered = np.zeros((9, N + 1))
x_filtered = np.zeros((9, N + 1))
for t in range(5):
    x_filtered[:, t] = x0
print(x_filtered[:, 0:5])

# # UKF setup for control input filtering
# def fx(u, dt):
#     return u  # The state transition function for the control input is identity

# def hx(u):
#     return u  # The measurement function is also identity

# control_dim = 3  # Dimension of the control input
# points = MerweScaledSigmaPoints(control_dim, alpha=0.1, beta=2., kappa=0)
# ukf = UKF(dim_x=control_dim, dim_z=control_dim, fx=fx, hx=hx, dt=dt, points=points)
# ukf.x = u_opt[:, 0]
# ukf.P *= 0.1
# ukf.R = np.eye(control_dim) * 0.1
# ukf.Q = np.eye(control_dim) * 0.1

# # Filter the control inputs using UKF
# u_opt_filtered = np.zeros_like(u_opt)
# for t in range(N):
#     ukf.predict()
#     ukf.update(u_opt[:, t])
#     u_opt_filtered[:, t] = ukf.x

# # Calculate the state trajectory using the filtered control inputs
# x_filtered = np.zeros((9, N + 1))
# x_filtered[:, 0] = x0

# Need to be change @AUstin
for t in range(N):
    if t >= 4:
        z_ = np.vstack([
            np.reshape(x_filtered[:, t], (9, 1)),
            np.reshape(x_filtered[:, t-1], (9, 1)),
            np.reshape(x_filtered[:, t-2], (9, 1)),
            np.reshape(x_filtered[:, t-3], (9, 1)),
            np.reshape(x_filtered[:, t-4], (9, 1)),
            np.reshape(u_opt_filtered[:, t], (3, 1))
        ])
        if t == 4:
            print(z_)
        z_next = AB @ z_
        x_filtered[:, t + 1] = np.reshape(z_next, (9,))

# print(u_opt.shape)

plt.plot(range(N), u_opt[0, :], '-o', label='u_x generated')
plt.plot(range(N), u_init[0, :], '-*', label='u_x tested')
plt.plot(range(N), u_opt_filtered[0, :], '-*', label='Filtered u_opt')
plt.xlabel('Time Steps')
plt.ylabel('U_X')
plt.title('Control Commands of X Direction')
plt.grid(True)
plt.legend()
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

print(np.arange(N + 1).shape)

# plt.plot(np.arange(N) * dt, x_opt[0, :], label=f'x{0}')
# plt.plot(np.arange(N) * dt, x_opt[6, :], label=f'x{6}')
# plt.plot(np.arange(N) * dt, x_opt[8, :], label=f'x{8}')
# plt.plot(np.arange(N) * dt, x_init[0, :], label=f'x{0} from init')
# plt.plot(np.arange(N) * dt, x_init[6, :], label=f'x{6} from init')
# plt.plot(np.arange(N) * dt, x_init[8, :], label=f'x{8} from init')
# plt.xlabel('Time')
# plt.ylabel('States')
# plt.title('State Trajectories over Time')
# plt.legend()
# plt.grid(True)
# plt.show()


plt.plot(np.arange(N + 1), x_opt[0, :], label='x_drone')
plt.plot(np.arange(N + 1), x_opt[6, :], label='x_tip')
plt.plot(np.arange(N + 1), x_opt[8, :], label='z_tip')
plt.plot(np.arange(N), x_init[0, :], label='init x_drone')
plt.plot(np.arange(N), x_init[6, :], label='init x_tip')
plt.plot(np.arange(N), x_init[8, :], label='init z_tip')
plt.plot(np.arange(N + 1), x_filtered[0, :], '--', label='Filtered x_drone')
plt.plot(np.arange(N + 1), x_filtered[6, :], '--', label='Filtered x_tip')
plt.plot(np.arange(N + 1), x_filtered[8, :], '--', label='Filtered z_tip')
plt.xlabel('Time')
plt.ylabel('States')
plt.title('State Trajectories over Time')
plt.legend()
plt.grid(True)
plt.show()



def calc(u, n=5, AB=None, vine=0, state_size=9):
    # Set initial position
    if state_size == 9:
        x0 = np.array([[0],[0],[1.5],[0],[0],[0],[0],[0],[1.5-vine]])
        
        # Check and set initial u
        u0 = np.array([[0], [0], [1.5]])
        if not np.all(np.equal(u[:,0:1].value, u0)):  # Use .value to extract NumPy array from cvxpy variable
            u = cp.hstack([u0, u])
        t = u.shape[1]
    elif state_size == 3:
        x0 = np.array([[0],[0],[1.5-vine]])
        
        # Check and set initial u
        u0 = np.array([[0]])
        if not np.all(np.equal(u[0].value, u0)):  # Use .value to extract NumPy array from cvxpy variable
            u = cp.hstack([u0, u])
        t = u.shape[1]
    
    # Set n history initial positions
    z0 = np.tile(x0, (n, 1)).flatten()
    z = cp.Variable((state_size * n, t))
    constraints = [z[:, 0] == z0]
    
    # Initialize x
    x = cp.Variable((state_size, t + 1))
    constraints += [x[:, 0] == x0.flatten()]
    
    for i in range(t):
        z_concat = cp.vstack([z[:, i], u[:, i]])
        constraints += [x[:, i + 1] == AB @ z_concat]
        z_next = cp.vstack([x[:, i + 1], z[:-(state_size), i]])
        constraints += [z[:, i + 1] == z_next]
    
    return x, constraints