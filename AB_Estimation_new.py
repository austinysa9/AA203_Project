import numpy as np
from scipy.io import loadmat, savemat

# Load data
data = loadmat('2024-02-15-15-04-46_resampled_0-05.mat')
local_position = data['local_position']
local_setpoint = data['local_setpoint']
tip_pose = data['tip_pose']
vision_pose = data['vision_pose']

# Constants
vine = 0.9723
num_skips = 1
train_size = 1400

# Data slicing
local_position = local_position[::num_skips, :train_size]
local_setpoint = local_setpoint[::num_skips, :train_size]
tip_pose = tip_pose[::num_skips, :train_size]
vision_pose = vision_pose[::num_skips, :train_size]


# Data preparation
X_all = np.concatenate((local_position[:, [1, 2, 3, 5, 6, 7]], tip_pose[:, 1:4]),axis=1).T
U_all = local_setpoint[:, 1:4].T


# Calculate AB
n = 5
w = 1395
t0 = n + w
AB = 0

for j in range(t0, len(X_all[0]) + 1):
    X_plus = X_all[:, j-w-1:j-1]
    U = U_all[:, j-w-2:j-2]
    X_i = X_all[:, j-w-1:j-1]
    for i in range(1, n):
        print(X_all[:, j-w-i-1:j-1-i].shape)
        Z = np.concatenate((X_i, X_all[:, j-w-i-1:j-1-i]), axis=0)
        X_i = Z
    
    ZU = np.concatenate((Z, U), axis=0)
    print(ZU.shape)
    # Solve least squares
    ABT, residuals, rank, s = np.linalg.lstsq(ZU.T, X_plus.T, rcond=0.5e-1)
    AB = ABT.T
    # ZU_pinv = np.linalg.pinv(ZU)
    # AB = X_plus @ ZU_pinv



print(AB)

# Save results
savemat('Model_AB_new.mat', {'AB': AB, 'vine': vine})
