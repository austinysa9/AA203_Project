import numpy as np
from scipy.linalg import solve_continuous_are, inv
from scipy.io import loadmat
import matplotlib.pyplot as plt
from Simulation import calc
from Animation import animate_trajectories
# Define system matrices
AB3_0 = np.array(loadmat('AB_from_ball_kick_3_states_n0.mat')['AB'])
#AB9_0 = np.array(loadmat('AB_from_ball_kick_9_states_n0.mat')['AB'])
vine = np.array(loadmat('AB_from_ball_kick_3_states.mat')['vine'])[0, 0]
A = AB3_0[:, :3]
B = AB3_0[:, 3:]
# print(A, B)
# print(A.shape, B.shape)

# Define cost matrices
Q = 0.1*np.eye(3)  # State cost
R = np.array([[0.01]])  # Input cost

# Solve the Continuous-time Algebraic Riccati Equation (CARE)
P = solve_continuous_are(A, B, Q, R)

# Compute the LQR gain
K = inv(R) @ B.T @ P

# Define initial and target states
x0 = np.array([0, 0, 1.5-vine])   # Initial state
x_target = np.array([1.5, 1.5, 1.5-vine])  # Target state

# Simulate the system
dt = 0.05  # Time step
T = 8     # Total time
x = x0
states = [x0]
controls = []
u_min = -2
u_max = 2

for t in np.arange(0, T, dt):
    u = -K @ (x - x_target)  # Control input
    u = np.clip(u, u_min, u_max)
    x = A @ x + B @ u    # State update
    print(f'Time: {t:.2f}, State: {x}')
    states.append(x)
    controls.append(u)
    
# Plot the results
states = np.array(states)
plt.figure()
plt.plot(np.arange(0, T, dt), controls)
plt.plot(np.arange(0, T+dt, dt), states[:, 0])
plt.plot(np.arange(0, T+dt, dt), states[:, 1])
plt.plot(np.arange(0, T+dt, dt), states[:, 2])
plt.show()

