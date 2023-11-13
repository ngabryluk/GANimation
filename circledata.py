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
import argparse
import random

parser = argparse.ArgumentParser(description="Specify the parameters of the data being created.")
parser.add_argument("-i", "--iterations", type=int, default=1,
                    help="Set the number of animations that will be created."                    
)
parser.add_argument("Shape", choices=["circle", "rectangle", "triangle"], default=random.choice(["circle", "rectangle", "triangle"]),
                    help="Pick the shape that will be used in creating data."
)
parser.add_argument("-r", "--radius", type=float, choices=np.arange(5, 40, 0.01), default=round(random.uniform(5, 40), 2),
                    help="Set the radius of the circle (5.00-40.00 - 2 decimal places)."
)
parser.add_argument("-rw", "--rectwidth", type=float, choices=np.arange(10, 75, 0.01), default=round(random.uniform(10, 50), 2),
                    help="Set the width of the rectangle (10.00-75.00 - 2 decimal places)."
)
parser.add_argument("-rh", "--rectheight", type=float, choices=np.arange(10, 75, 0.01), default=round(random.uniform(10, 50), 2),
                    help="Set the height of the rectangle (10.00-75.00 - 2 decimal places)."
)
parser.add_argument("-b", "--base", type=float, choices=np.arange(10, 75, 0.01), default=round(random.uniform(10, 50), 2),
                    help="Set the base of the triangle (10.00-75.00 - 2 decimal places)."
)
parser.add_argument("-th", "--triheight", type=float, choices=np.arange(10, 75, 0.01), default=round(random.uniform(10, 50), 2),
                    help="Set the height of the triangle (10.00-75.00 - 2 decimal places)."
)
parser.add_argument("-d", "--direction", choices=["right", "left", "up", "down", "diagonal", "bouncy", "random"],
                    default=random.choice(["right", "left", "up", "down", "diagonal", "bouncy", "random"]),
                    help="Set the direction that the shape will move."
)
parser.add_argument("-s", "--speed", type=int, default=200,
                    help="Set the speed of the animations. The parameter this will be plugged into is the amount of "
                    "milliseconds of delay between frames, so the lower the number, the faster the animation will be."
)

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

def circles(direction, size, speed):
    
    dpi = 142

    # Create a figure that's 256x256 pixels
    fig = plt.figure(figsize=(256/dpi, 256/dpi), dpi=dpi)

    # Create an axis with no axis labels or ticks
    ax = fig.add_subplot(111)
    # ax.axis('off')

    # Define the initial position of the shape
    x0, y0 = 50, 50
    radius = 50  # Radius of the circle

    # Create the circle patch
    circle = plt.Circle((x0, y0), radius, fc='black')

    # Add the circle to the axis
    ax.add_patch(circle)

    ax.set_xlim(0, 256)
    ax.set_ylim(0, 256)
    
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
    # ani = animation.FuncAnimation(fig, update, frames=100, repeat=False, blit=False)

    # Display the animation
    plt.show()

if __name__ == "__main__":
    if 
    circles()
    # gaussianNoise()