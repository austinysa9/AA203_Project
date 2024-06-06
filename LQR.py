import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from scipy.linalg import solve_discrete_are, inv
from scipy.io import loadmat
import scipy.io

# AB = np.array(loadmat('Model_AB_1.mat')['AB'])
# vine = np.array(loadmat('Model_AB_1.mat')['vine'])[0, 0]
# A = AB[:,0:9]
# B = AB[:,9:12]
# n = 9
# m = 3
n = 3
m = 1


def ricatti_recursion(
    A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray
) -> np.ndarray:
    """Compute the gain matrix K through Ricatti recursion

    Args:
        A (np.ndarray): Dynamics matrix, shape (n, n)
        B (np.ndarray): Controls matrix, shape (n, m)
        Q (np.ndarray): State cost matrix, shape (n, n)
        R (np.ndarray): Control cost matrix, shape (m, m)

    Returns:
        np.ndarray: Gain matrix K, shape (m, n)
    """
    eps = 1e-4  # Riccati recursion convergence tolerance
    max_iters = 1000  # Riccati recursion maximum number of iterations
    P_prev = np.zeros((n, n))  # initialization
    converged = False
    for i in range(max_iters):
        # PART (b) ##################################################
        # INSTRUCTIONS: Apply the Ricatti equation until convergence
        K = -np.linalg.inv(R + B.T @ P_prev @ B) @ B.T @ P_prev @ A
        P_now = Q + A.T @ P_prev @ (A + B @ K)
        diff = P_now - P_prev
        P_prev = P_now
        if np.max(np.abs(diff)) < eps:
            print(i)
            converged = True
            break
        
        # END PART (b) ##############################################
    
    if not converged:
        raise RuntimeError("Ricatti recursion did not converge!")
    K_print = np.round(K, 2)
    print("K:", K_print)
    return K

def simulate(
    t: np.ndarray, s_ref: np.ndarray, u_ref: np.ndarray, s0: np.ndarray, K: np.ndarray, A, B
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate the cartpole

    Args:
        t (np.ndarray): Evaluation times, shape (num_timesteps,)
        s_ref (np.ndarray): Reference state s_bar, evaluated at each time t. Shape (num_timesteps, n)
        u_ref (np.ndarray): Reference control u_bar, shape (m,)
        s0 (np.ndarray): Initial state, shape (n,)
        K (np.ndarray): Feedback gain matrix (Ricatti recursion result), shape (m, n)

    Returns:
        tuple[np.ndarray, np.ndarray]: Tuple of:
            np.ndarray: The state history, shape (num_timesteps, n)
            np.ndarray: The control history, shape (num_timesteps, m)
    """

    # PART (c) ##################################################
    # INSTRUCTIONS: Complete the function to simulate the cartpole system
    # Hint: use the cartpole wrapper above with odeint
    # s = np.zeros((n, t.shape[0]))
    # u = np.zeros((m, t.shape[0]))
    # # u0 = np.array([[0], [0], [1.5]])
    # u0 = np.array([[0]])
    # s[:,0:1] = s0
    # for k in range(len(t) - 1):
    #     u[:,k] = K @ (s[:,k] - s_ref[:,k])
    #     s[:,k+1] = A @ s[:,k] + B @ u[:,k]
    # u[:,-1] = u[:,-2]
    
    s = np.zeros((t.shape[0], n))
    u = np.zeros((t.shape[0], m))
    s[0,:] = np.squeeze(s0)
    for k in range(len(t) - 1):
        u[k] = K @ (s[k] - s_ref[k])
        s[k+1] = A @ s[k] + B @ u[k]
    u[-1,:] = u[-2,:]
    
    # # Set 5 history initial positions
    # z0 = np.tile(x0, (5, 1))
    # z0.squeeze()
    # z = np.zeros((9 * 5, t))
    # z[:,0:1] = z0
    
    # for i in range(5):
    #     s[i, :] = s0
    
    # for k in range(5, len(t) - 1):
    #     u[k] = K @ (s[k] - s_ref[k])
    #     s[:,i+1:i+2] = AB @ np.concatenate((z[:,i:i+1], u[:,i:i+1]), axis=0)
    # u[-1,:] = u[-2,:]
    # for i in range(5):
    #     u[i,:] = u0
        
    # x0 = np.array([[0],[0],[1.5],[0],[0],[0],[0],[0],[1.5-vine]])
    
    # # Check and set initial u
    # u0 = np.array([[0], [0], [1.5]])
    # if not np.all(np.equal(u[:,0:1], u0)):
    #     u = np.hstack((u0, u))
    # t = u.shape[1]
    
    # # Set n history initial positions
    # z0 = np.tile(x0, (n, 1))
    # z0.squeeze()
    # z = np.zeros((9 * n, t))
    # z[:,0:1] = z0
    
    
    # # Initialize x
    # x = np.zeros((9, t + 1))
    # x[:,0:1] = x0
    
    # for i in range(t):
    #     x[:,i+1:i+2] = AB @ np.concatenate((z[:,i:i+1], u[:,i:i+1]), axis=0)
    #     z[:, i+1:i+2] = np.concatenate((x[:, i+1:i+2], z[:-9,i:i+1]), axis=0)
    
    
    return s, u
def reference(t: float) -> np.ndarray:
    """Compute the reference state (s_bar) at time t

    Args:
        t (float): Evaluation time

    Returns:
        np.ndarray: Reference state, shape (n,)
    """
    a = 10.0  # Amplitude\
    T = 10.0  # Period
    
    

    # PART (d) ##################################################
    # INSTRUCTIONS: Compute the reference state for a given time
    return np.array([a*np.sin(2*np.pi*t/T), a*np.sin(2*np.pi*t/T), 0])
    
    
def main():
    # AB = np.array(loadmat('Model_AB_1.mat')['AB'])
    # vine = np.array(loadmat('Model_AB_1.mat')['vine'])[0, 0]
    # A = AB[:,0:9]
    # B = AB[:,9:12]
    # n = 9
    # m = 3
    
    Q = 10*np.eye(n)  # state cost matrix 10*
    R = np.eye(m)  # control cost matrix
    K = ricatti_recursion(A, B, Q, R)
    # Part C
    t = np.arange(0.0, 5.0, 1 / 20)
    print(t.size)
    #print(t.shape)
    # s_ref = np.array([[0],[0],[1.5],[0],[0],[0],[0],[0],[1.5-vine]]) * np.ones((1, t.size))
    # u_ref = np.array([[0], [0], [1.5]])
    
    s_ref = np.array([0, 0, 0]) * np.ones((t.size, 1))
    u_ref = np.array([0.0])
    vine = 1.0  # Example value for 'vine'
    #s_ref = np.array([reference(ti) for ti in t])

    # # Define the repeated updates
    # s_update = np.array([[0], [0], [1.5 + vine], [0], [0], [0], [0], [0], [1.5]])
    # u_update = np.array([[0], [0], [1.5]])

    # # Update all columns
    # for s in range(1400):
    #     s_ref[:, s] += np.array([s, 0, 1.5, 0, 0, 0, s, 0, 1.5-vine])
    #     u_ref[:, s] += np.array([s, 0, 1.5])
    #s0 = np.array([0.0, 3 * np.pi / 4, 0.0, 0.0])
    # s0 = np.array([[0],[0],[1.5],[0],[0],[0],[0],[0],[1.5-vine]])
    s0 = np.array([[0], [0], [1.5-vine]])
    s, u = simulate(t, s_ref, u_ref, s0, K, A, B)
    plt.plot(t, s[:, 0], label='x')
    plt.plot(t, s[:, 1], label='x_tip')
    plt.plot(t, s[:, 2], label='z_tip')
    plt.plot(t, u, label='control')
    plt.legend()
    # scipy.io.savemat('control_inputs.mat', {'control_inputs': u})
    plt.show()
    
if __name__ == "__main__":
    main()