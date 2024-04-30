import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

def split_and_animate(data):
    # Splitting the N x 6 array into two N x 3 arrays
    data1 = data[:, :3]
    data2 = data[:, 3:]

    # Prepare the 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim([np.min(data[:, :3]), np.max(data[:, :3])])
    ax.set_ylim([np.min(data[:, :3]), np.max(data[:, :3])])
    ax.set_zlim([np.min(data[:, :3]), np.max(data[:, :3])])

    # Plot initial points
    scatter1 = ax.scatter(data1[0, 0], data1[0, 1], data1[0, 2], color='b', label='Set 1')
    scatter2 = ax.scatter(data2[0, 0], data2[0, 1], data2[0, 2], color='r', label='Set 2')
    ax.legend()

    # Update function for animation
    def update(frame):
        scatter1._offsets3d = (data1[frame:frame+1, 0], data1[frame:frame+1, 1], data1[frame:frame+1, 2])
        scatter2._offsets3d = (data2[frame:frame+1, 0], data2[frame:frame+1, 1], data2[frame:frame+1, 2])
        ax.set_title(f"Time step: {frame + 1}")
        return scatter1, scatter2

    # Create animation
    ani = FuncAnimation(fig, update, frames=np.arange(len(data)), repeat=True)
    plt.show()
    return ani  # Keep the animation object alive

# Example usage
N = 100
data = np.random.rand(N, 6)  # Random data for illustration
anim = split_and_animate(data)  # Assign the animation to a variable
