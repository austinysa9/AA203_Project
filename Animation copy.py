import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FFMpegWriter

def animate_trajectories(time, x, y, z, x1, y1, z1, x2, y2, z2, trail_length_red=10, trail_length_blue=10):
    # Create a figure and a 3D axis
    fig, ax = plt.subplots(figsize=(8, 6), subplot_kw={'projection': '3d'})

    # Set the limits of the axes based on the data
    ax.set_xlim(min(min(x), min(x1), min(x2)), max(max(x), max(x1), max(x2)))
    ax.set_ylim(min(min(y), min(y1), min(y2)), max(max(y), max(y1), max(y2)))
    ax.set_zlim(min(min(z), min(z1), min(z2)), max(max(z), max(z1), max(z2)))

    # Initialize three line plots for the trajectories
    line1, = ax.plot([], [], [], 'r-', label='Drone', marker='o') #  (Red)  (Blue)  (Magenta)  (Green)
    line2, = ax.plot([], [], [], 'b-', label='Tip', marker='o')
    line3, = ax.plot([], [], [], 'm-', label='Control Input', marker='o')

    # Initialize the dashed line connecting the newest points of the first two trajectories
    connector, = ax.plot([], [], [], 'g--', label='Vine')  # Green dashed line

    # Adding a legend
    ax.legend()

    # Animation function that updates the lines of all trajectories
    def animate(i):
        # Update line1 for Trajectory 1 with its custom trail length
        start_idx_red = max(0, i - trail_length_red)
        line1.set_data(x[start_idx_red:i], y[start_idx_red:i])
        line1.set_3d_properties(z[start_idx_red:i])
        
        # Update line2 for Trajectory 2 with its custom trail length
        start_idx_blue = max(0, i - trail_length_blue)
        line2.set_data(x1[start_idx_blue:i], y1[start_idx_blue:i])
        line2.set_3d_properties(z1[start_idx_blue:i])
        
        # Update line3 for Trajectory 3 to only show the newest point
        if i > 0:  # Ensure there is at least one point
            line3.set_data(x2[i-1:i], y2[i-1:i])
            line3.set_3d_properties(z2[i-1:i])

        # Update connector line (only between the first two points)
        if i > 0:  # Ensure there is at least one point to connect
            connector.set_data([x[i-1], x1[i-1]], [y[i-1], y1[i-1]])
            connector.set_3d_properties([z[i-1], z1[i-1]])

        return line1, line2, line3, connector

    # Create the animation
    ani = FuncAnimation(fig, animate, frames=len(time), blit=True, interval=50)
    # writer = FFMpegWriter(fps=20, metadata=dict(artist='Me'), bitrate=1800)
    # ani.save('trajectory_animation.mp4', writer=writer)

    # Show the plot
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    plt.show()

