import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from Simulation import calc
from Animation import animate_trajectories

AB3 = np.array(loadmat('AB_from_ball_kick_3_states.mat')['AB'])
AB9 = np.array(loadmat('AB_from_ball_kick_9_states.mat')['AB'])
AB3_0 = np.array(loadmat('AB_from_ball_kick_3_states_n0.mat')['AB'])
AB9_0 = np.array(loadmat('AB_from_ball_kick_9_states_n0.mat')['AB'])
vine = np.array(loadmat('AB_from_ball_kick_3_states.mat')['vine'])[0, 0]

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
x1 = calc(U_all, n=5, AB=AB9, state_size=9)
x2 = calc(U_all, n=1, AB=AB9_0, state_size=9)

# Constructing X_all and U_all
X_all = np.transpose(np.hstack((local_position[:, [1, 2, 3, 5, 6, 7]], 
                                tip_pose[:, 1:4])))
U_all = np.transpose(local_setpoint[:, 1:2])


x = calc(U_all, AB=AB3, state_size=3)
x3 = calc(U_all, n=1, AB=AB3_0, state_size=3)

# 2D Plot
t = range(1, X_all.shape[1]+1)
t1 = range(1, x.shape[1]+1)
# Create a figure and axis
fig, ax = plt.subplots()

# Customizing the plot
ax.plot(t, X_all[-1,:], color='red', label="Real Data")
ax.plot(t1, x[-1,:], color='blue', label="Test Data from 3-state model with history size 5")
ax.plot(t1, x1[-1,:], color='yellow', label="Test Data from 9-state model with history size 5")
ax.plot(t1, x2[-1,:], color='black', label="Test Data from 9-state model without history")
ax.plot(t1, x3[-1,:], color='purple', label="Test Data from 3-state model without history")
# Adding title and labels
ax.set_title('Tip Z')
ax.set_xlabel('Time')
ax.set_ylabel('Z')
plt.legend()

# 2D Plot
t = range(1, X_all.shape[1]+1)
t1 = range(1, x.shape[1]+1)
# Create a figure and axis
fig, ax = plt.subplots()

# Customizing the plot
ax.plot(t, X_all[-3,:], color='red', label="Real Data")
ax.plot(t1, x[-2,:], color='blue', label="Test Data from 3-state model with history size 5")
ax.plot(t1, x1[-3,:], color='yellow', label="Test Data from 9-state model with history size 5")
ax.plot(t1, x2[-3,:], color='black', label="Test Data from 9-state model without history")
ax.plot(t1, x3[-2,:], color='purple', label="Test Data from 3-state model without history")
# Adding title and labels
ax.set_title('Tip X')
ax.set_xlabel('Time')
ax.set_ylabel('X')
plt.legend()

# 2D Plot
t = range(1, X_all.shape[1]+1)
t1 = range(1, x.shape[1]+1)
# Create a figure and axis
fig, ax = plt.subplots()

# Customizing the plot
ax.plot(t, X_all[0,:], color='red', label="Real Data")
ax.plot(t1, x[0,:], color='blue', label="Test Data from 3-state model with history size 5")
ax.plot(t1, x1[0,:], color='yellow', label="Test Data from 9-state model with history size 5")
ax.plot(t1, x2[0,:], color='black', label="Test Data from 9-state model without history")
ax.plot(t1, x3[0,:], color='purple', label="Test Data from 3-state model without history")
# Adding title and labels
ax.set_title('Drone X')
ax.set_xlabel('Time')
ax.set_ylabel('X')

plt.legend()

# Create a figure and axes for subplots
t = range(1, X_all[:,150:300].shape[1]+1)
t1 = range(1, x[:,150:300].shape[1]+1)

fig, axs = plt.subplots(2, 1, figsize=(10, 8))

# # Customizing the first subplot
# axs[0].plot(t, X_all[-1, 100:300], color='red', label="Real Data")
# axs[0].plot(t1, x[-1, 100:300], color='blue', label="Test Data from 3-state model with history size 5")
# axs[0].plot(t1, x1[-1, 100:300], color='yellow', label="Test Data from 9-state model with history size 5")
# axs[0].plot(t1, x2[-1, 100:300], color='black', label="Test Data from 9-state model without history")
# axs[0].plot(t1, x3[-1, 100:300], color='purple', label="Test Data from 3-state model without history")
# axs[0].set_title('Tip Z')
# axs[0].set_xlabel('Time')
# axs[0].set_ylabel('Z')
# axs[0].legend()

# Customizing the second subplot
axs[1].plot(t, X_all[-3, 150:300], color='red', label="Real Data")
axs[1].plot(t1, x[-2, 150:300], color='blue', label="Test Data from 3-state model with history size 5")
axs[1].plot(t1, x1[-3, 150:300], color='yellow', label="Test Data from 9-state model with history size 5")
axs[1].plot(t1, x2[-3, 150:300], color='black', label="Test Data from 9-state model without history")
axs[1].plot(t1, x3[-2, 150:300], color='purple', label="Test Data from 3-state model without history")
axs[1].set_title('Tip X')
axs[1].set_xlabel('Time')
axs[1].set_ylabel('X')
axs[1].legend()

# Customizing the third subplot
axs[0].plot(t, X_all[0, 150:300], color='red', label="Real Data")
axs[0].plot(t1, x[0, 150:300], color='blue', label="Test Data from 3-state model with history size 5")
axs[0].plot(t1, x1[0, 150:300], color='yellow', label="Test Data from 9-state model with history size 5")
axs[0].plot(t1, x2[0, 150:300], color='black', label="Test Data from 9-state model without history")
axs[0].plot(t1, x3[0, 150:300], color='purple', label="Test Data from 3-state model without history")
axs[0].set_title('Drone X')
axs[0].set_xlabel('Time')
axs[0].set_ylabel('X')
axs[0].legend()

# Adjust layout and show the plot
plt.tight_layout()
plt.show()
  
# # Show the plot
plt.show()