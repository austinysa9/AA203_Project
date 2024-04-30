import numpy as np
from scipy.io import loadmat

# Load data from the .mat file
data = loadmat('austinysa9/AA203_Project/Data.mat')
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
X_all = np.transpose(np.hstack((local_position[:, [1, 2, 3, 5, 6, 7]], tip_pose[:, 1:4])))
U_all = np.transpose(local_setpoint[:, 1:4])

# Initialize parameters for system identification
n = 5  # History size
w = 1395  # Window size
t0 = n + w + 1
len_X = X_all.shape[1]

# Loop to compute AB5
AB5 = 0
for j in range(t0, len_X + 1):
    X_plus = X_all[:, j-w:j]
    U = U_all[:, j-w-1:j-1]
    Z = np.array([]).reshape(0, w)  # Initialize empty array with 0 rows and w columns
    for i in range(1, n+1):
        Z = np.vstack((Z, X_all[:, j-w-i:j-i-1]))

    # Compute AB5 using least squares if Z and U are not singular
    if np.linalg.matrix_rank(np.vstack((Z, U))) == np.vstack((Z, U)).shape[0]:
        AB5 = np.linalg.lstsq(np.vstack((Z, U)), X_plus, rcond=None)[0]

print("AB5 Matrix:")
print(AB5)
