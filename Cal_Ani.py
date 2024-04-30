import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

def split_and_animate(data):
    # Splitting the N x 6 array into two N x 3 arrays
    data1 = data[:, :3]
    data2 = data[:, 3:]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    def update(frame):
        ax.clear()
        ax.scatter(data1[frame, 0], data1[frame, 1], data1[frame, 2], color='b', label='Set 1')
        ax.scatter(data2[frame, 0], data2[frame, 1], data2[frame, 2], color='r', label='Set 2')
        ax.legend()
        ax.set_xlim([np.min(data[:, :3]), np.max(data[:, :3])])
        ax.set_ylim([np.min(data[:, :3]), np.max(data[:, :3])])
        ax.set_zlim([np.min(data[:, :3]), np.max(data[:, :3])])
        ax.set_title(f"Time step: {frame+1}")

    ani = FuncAnimation(fig, update, frames=len(data), repeat=True)
    plt.show()

# Example usage
# Let's assume you have an N x 6 array, where N is the number of timesteps
N = 100
data = np.random.rand(N, 6)  # Random data for illustration
split_and_animate(data)
