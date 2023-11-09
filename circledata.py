# circledata.py
# Python script used for creating animations of circles moving on screen from one position to another
# This file should be able to run with specifying different parameters
# Parameters that will vary between animation that will be in the naming convention include:
#   - Shape (in this case circle)
#   - Direction (right/left/up/down/diagonal/bouncy)
#   - Size
#   - Speed
#   - noise
#   - index for cases

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import random

def gaussianNoise():
    # Create the figure
    fig, ax = plt.subplots()

    # Create the data
    x = np.linspace(0, 1, 100)
    y = np.sin(x)

    # Add the noise
    noise = np.random.normal(0, 0.1, 100)
    y += noise

    # Plot the data
    ax.plot(x, y)

    # Show the figure
    plt.show()

def circles(direction, size, speed, noise, index=1):
    
    # Create a figure that's 256x256 pixels
    fig = plt.figure(figsize=(256/80, 256/80))  # 80 DPI is the default DPI for Matplotlib

    # Create an axis with no axis labels or ticks
    ax = fig.add_subplot(111)
    # ax.axis('off')

    # Define the initial position of the shape
    x0, y0 = 0.5, 0.5
    radius = 0.05  # Radius of the circle

    # Create the circle patch
    circle = plt.Circle((x0, y0), radius, fc='blue')

    # Add the circle to the axis
    ax.add_patch(circle)

    ax.set_xlim(0, 256)
    ax.set_ylim(0, 256)
    plt.show()

    # Define the animation function to update the position of the circle randomly
    def update(frame):
        # Calculate a random direction for movement
        direction_x = random.uniform(-0.01, 0.01)
        direction_y = random.uniform(-0.01, 0.01)

        # Calculate the new position of the circle
        x = circle.get_center()[0] + direction_x
        y = circle.get_center()[1] + direction_y

        # Update the position of the circle patch
        circle.set_center((x, y))

    # Create an animation object
    ani = animation.FuncAnimation(fig, update, frames=100, repeat=False, blit=False)

    # Display the animation
    plt.show()

if __name__ == "__main__":
    # circles()
    gaussianNoise()