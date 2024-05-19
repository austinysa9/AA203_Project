import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from scipy.linalg import solve_discrete_are, inv
from scipy.io import loadmat

AB = np.array(loadmat('Model_AB.mat')['AB'])
vine = np.array(loadmat('Model_AB.mat')['vine'])[0, 0]

n = 9
m = 3

A = AB[:,0:45]
B = AB[:,45:48]

x = np.random.uniform(low=0.0, high=1.0, size=(45, 1))
u = np.random.uniform(low=0.0, high=1.0, size=(3, 1))
z = np.concatenate((x, u), axis=0)

print( AB @ z-(A @ x  + B@u))