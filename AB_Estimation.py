import numpy as np
from scipy.io import loadmat

# Load data from the .mat file
data = loadmat('Data.mat')
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


##
def AB(X_all, U_all):
    len = X_all.shape[1]
    # History size
    n = 5
    # Window size
    w = 1395
    # % Predict (Totally need w + n data)
    # % At time step t0 = n + w + 1
    # % [x(n+1)...x(w+n)] = [A B] [x(n) ... x(w+n-1)]
    # %                             ..
    # %                            x(1) ... x(w)
    # %                            u(n) ... u(w+n-1)
    # % Size: 9 * w = [A B] * (9n + 3) * w
    t0 = n + w + 1
    # Fixed amount of data m
    AB1 = np.zeros((9, 9 * n + 3, 1200))
    for j in range(t0, len):
        X_plus = X_all[:, j-w:j-1]
        U = U_all[:, j-w-1:j-2]
        Z = np.vstack([X_all[:, j-w-i:j-1-i] for i in range(1, n+1)])
        AB1[:, :, j] = np.hstack((X_plus,)) @ np.linalg.pinv(np.concatenate((Z, U),axis=0))
    return AB1
  
def main():
    # Your main program logic goes here
    AB1 = AB(X_all,U_all)
    print(AB1)

# Call the main function if this script is run directly
if __name__ == "__main__":
    main()