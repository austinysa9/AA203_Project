import numpy as np
from scipy.io import loadmat

AB = np.array(loadmat('Model_AB.mat')['AB'])
vine = np.array(loadmat('Model_AB.mat')['vine'])[0, 0]


def calc(u, n=5, AB=AB, vine=vine, state_size=9):
    # Set initial position
    if state_size == 9:
        x0 = np.array([[0],[0],[1.5],[0],[0],[0],[0],[0],[1.5-vine]])
        
        # Check and set initial u
        u0 = np.array([[0], [0], [1.5]])
        if not np.all(np.equal(u[:,0:1], u0)):
            u = np.hstack((u0, u))
        t = u.shape[1]
    elif state_size == 3:
        x0 = np.array([[0],[0],[1.5-vine]])
        
        # Check and set initial u
        u0 = np.array([[0]])
        if not np.all(np.equal(u[0], u0)):
            u = np.hstack((u0, u))
        t = u.shape[1]
    
    # Set n history initial positions
    z0 = np.tile(x0, (n, 1))
    z0.squeeze()
    z = np.zeros((state_size * n, t))
    z[:,0:1] = z0
    
    
    # Initialize x
    x = np.zeros((state_size, t + 1))
    x[:,0:1] = x0
    
    for i in range(t):
        x[:,i+1:i+2] = AB @ np.concatenate((z[:,i:i+1], u[:,i:i+1]), axis=0)
        z[:, i+1:i+2] = np.concatenate((x[:, i+1:i+2], z[:-state_size,i:i+1]), axis=0)
    return x
    