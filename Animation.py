import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

def animate_trajectories(time, x, y, z, x1, y1, z1, trail_length=50):
    # Create a figure and a 3D axis
    fig, ax = plt.subplots(figsize=(8, 6), subplot_kw={'projection': '3d'})

    # Set the limits of the axes based on the data
    ax.set_xlim(min(min(x), min(x1)), max(max(x), max(x1)))
    ax.set_ylim(min(min(y), min(y1)), max(max(y), max(y1)))
    ax.set_zlim(min(min(z), min(z1)), max(max(z), max(z1)))

    # Initialize two line plots for the trajectories
    line1, = ax.plot([], [], [], 'r-', label='Drone', marker='o')
    line2, = ax.plot([], [], [], 'b-', label='Tip', marker='o')

    # Adding a legend
    ax.legend()

    # Animation function that updates the lines of both trajectories
    def animate(i):
        # Calculate start index for the trail
        start_idx = max(0, i - trail_length)
        
        # Update line1 for Trajectory 1
        line1.set_data(x[start_idx:i], y[start_idx:i])
        line1.set_3d_properties(z[start_idx:i])
        
        # Update line2 for Trajectory 2
        line2.set_data(x1[start_idx:i], y1[start_idx:i])
        line2.set_3d_properties(z1[start_idx:i])
        
        return line1, line2

    # Create the animation
    ani = FuncAnimation(fig, animate, frames=len(time), blit=True, interval=50)

    # Show the plot
    plt.show()
