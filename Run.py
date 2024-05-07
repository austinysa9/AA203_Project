import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from Simulation import calc
from Animation import animate_trajectories

AB = np.array(loadmat('Model_AB_10.mat')['AB'])
vine = np.array(loadmat('Model_AB_10.mat')['vine'])[0, 0]
data = loadmat('data_measured.mat')
local_position = data['local_position']
local_setpoint = data['local_setpoint']
tip_pose = data['tip_pose']
vision_pose = data['vision_pose']  # Assuming you might need this later

# Parameters
num_skips = 1
train_size = 1400

# Data slicing and resampling
local_position = local_position[::num_skips, :train_size]
local_setpoint = local_setpoint[::num_skips, :train_size]
tip_pose = tip_pose[::num_skips, :train_size]
vision_pose = vision_pose[::num_skips, :train_size]

# Constructing X_all and U_all
X_all = np.transpose(np.hstack((local_position[:, [1, 2, 3, 5, 6, 7]], 
                                tip_pose[:, 1:4])))
U_all = np.transpose(local_setpoint[:, 1:4])

x = calc(U_all, n=10, AB=AB, vine=vine)
print(X_all.shape, x.shape, U_all.shape)
t = range(1, X_all.shape[1]+1)
t1 = range(1, x.shape[1]+1)
# Create a figure and axis
fig, ax = plt.subplots()

# # Customizing the plot
ax.plot(t, X_all[-1,:], color='red', marker='o', linestyle='--')
ax.plot(t1, x[-1,:], color='blue', marker='o', linestyle='--')
# Adding title and labels
ax.set_title('Customized Line Plot')
ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')

# Show the plot
plt.show()

# Example usage of the function
time = np.linspace(0, 20, num=1402)
x_drone = x[0,:]
y_drone = x[1,:]
z_drone = x[2,:]
x_tip = x[-3,:]
y_tip = x[-2,:]
z_tip = x[-1,:]

first_row = U_all[:, 0]
last_row = U_all[:, -1] 
first_row = first_row.reshape(3, 1)
last_row = last_row.reshape(3, 1)

# Concatenate the first row at the beginning and the last row at the end
u_paddle = np.hstack((first_row, U_all, last_row))

u_x_paddle = U_all[0,:]
u_y_paddle = U_all[1,:]
u_z_paddle = U_all[2,:]
animate_trajectories(time, x_drone, y_drone, z_drone, x_tip, y_tip, z_tip, u_x_paddle, u_y_paddle, u_z_paddle)