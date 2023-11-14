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
import os
import random
import pdb

# Path to store the results in
ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

parser = argparse.ArgumentParser(description="Specify the parameters of the data being created.")
parser.add_argument("-i", "--iterations", type=int, default=1,
                    help="Set the number of animations that will be created."                    
)
parser.add_argument("shape", choices=["circle", "rectangle", "triangle"], default=None,
                    help="Pick the shape that will be used in creating data."
)
parser.add_argument("-r", "--radius", type=int, choices=range(10, 78), default=0,
                    help="Set the radius of the circle (10-77)."
)
parser.add_argument("-rw", "--rectwidth", type=int, choices=range(10, 76), default=0,
                    help="Set the width of the rectangle (10-75)."
)
parser.add_argument("-rh", "--rectheight", type=int, choices=range(10, 76), default=0,
                    help="Set the height of the rectangle (10-75)."
)
parser.add_argument("-b", "--base", type=int, choices=range(10, 76), default=0,
                    help="Set the base of the triangle (10-75)."
)
parser.add_argument("-th", "--triheight", type=int, choices=range(10, 76), default=0,
                    help="Set the height of the triangle (10-75)."
)
parser.add_argument("-d", "--direction", choices=["right", "left", "up", "down", "diagonal", "bouncy"], default=None,
                    help="Set the direction that the shape will move."
)
parser.add_argument("-s", "--speed", type=int, choices=np.arange(0.4, 2, 0.01), default=0,
                    help="Set the speed of the animations. This is the value that x or y is increased by each frame."
)

def circle(direction, size, speed, iterations):

    # If a value wasn't specified, check a flag that will tell the program to make that value random each iteration
    randDirection, randSize, randSpeed = False, False, False

    if direction is None:
        randDirection = True
    if size == 0:
        randSize = True
    if speed == 0:
        randSpeed = True

    i = 1

    while i <= iterations:

        # Set a random value for the parameters we want randomized for each animation
        if randDirection:
            direction = random.choice(["right", "left", "up", "down", "diagonal"])
        if randSize:
            size = random.randint(10, 77)
        if randSpeed:
            speed = random.uniform(0.4, 2)
        
        # Create a figure that's 256x256 pixels
        dpi = 142
        fig = plt.figure(figsize=(256/dpi, 256/dpi), dpi=dpi)

        # Create an axis with no axis labels or ticks and a black background
        ax = fig.add_subplot(111)
        ax.axis('off')
        fig.set_facecolor("black")
        ax.set_facecolor("black")

        # Set the plot axis to by 256 x 256 to match the pixels
        ax.set_xlim(0, 256)
        ax.set_ylim(0, 256)

        radius = size  # Radius of the circle

        # Randomly pick which diagonal direction to go (up and right, down and right, down and left, up and left) for the function
        rand = random.randint(1, 4)


        # Define the initial position of the shape based on the direction (we don't want to go out of bounds!)
        x0, y0 = setpositioncircle(direction, radius, rand)

        # Create the circle patch
        circle = plt.Circle((x0, y0), radius, fc='white')

        # Add the circle to the axis
        ax.add_patch(circle)
        
        # Define the animation function to update the position of the patch
        def right(frame):
            # Calculate the new position of the patch
            x = x0 + frame * speed
            y = y0

            # Save the current frame
            # plt.savefig(f'{ROOT}\\c-r{radius}-r-{speed}-{frame:03}.jpg')

            # Update the position of the patch
            circle.set_center((x, y))

        def left(frame):
            # Calculate the new position of the circle
            x = x0 - frame * speed
            y = y0

            # Save the current frame
            # plt.savefig(f'{ROOT}\\c-r{radius}-l-{speed}-{frame:03}.jpg')

            # Update the position of the circle patch
            circle.set_center((x, y))

        def up(frame):
            # Calculate the new position of the circle
            x = x0 
            y = y0 + frame * speed

            # Save the current frame
            # plt.savefig(f'{ROOT}\\c-r{radius}-u-{speed}-{frame:03}.jpg')

            # Update the position of the circle patch
            circle.set_center((x, y))

        def down(frame):
            # Calculate the new position of the circle
            x = x0
            y = y0 - frame * speed

            # Save the current frame
            # plt.savefig(f'{ROOT}\\c-r{radius}-dwn-{speed}-{frame:03}.jpg')

            # Update the position of the circle patch
            circle.set_center((x, y))
        
        def diagonal(frame):
            # Update x and y based on the random diagonal direction picked
            if rand == 1:
                x = x0 + frame * speed
                y = y0 + frame * speed
            elif rand == 2:
                x = x0 + frame * speed
                y = y0 - frame * speed
            elif rand == 3:
                x = x0 - frame * speed
                y = y0 - frame * speed
            elif rand == 4:
                x = x0 - frame * speed
                y = y0 + frame * speed

            # Save the current frame
            # plt.savefig(f'{ROOT}\\c-r{radius}-diag-{speed}-{frame:03}.jpg')

            circle.set_center((x, y))

        # Create the animation objects and save them
        if direction == "right":
            ani = animation.FuncAnimation(fig, right, interval=75, frames=50, repeat=False, blit=False)
            ani.save(f'{ROOT}\\{i:03}-c-r{radius}-r-{int((speed / 2) * 100)}.gif', fps=25)
        elif direction == "left":
            ani = animation.FuncAnimation(fig, left, interval=75, frames=50, repeat=False, blit=False)
            ani.save(f'{ROOT}\\{i:03}-c-r{radius}-l-{int((speed / 2) * 100)}.gif', fps=25)
        elif direction == "up":
            ani = animation.FuncAnimation(fig, up, interval=75, frames=50, repeat=False, blit=False)
            ani.save(f'{ROOT}\\{i:03}-c-r{radius}-u-{int((speed / 2) * 100)}.gif', fps=25)
        elif direction == "down":
            ani = animation.FuncAnimation(fig, down, interval=75, frames=50, repeat=False, blit=False)
            # ani.save(f'{ROOT}\\{i:03}-c-r{radius}-dwn-{int((speed / 2) * 100)}.gif', fps=25)
        elif direction == "diagonal":
            ani = animation.FuncAnimation(fig, diagonal, interval=75, frames=50, repeat=False, blit=False)
            ani.save(f'{ROOT}\\{i:03}-c-r{radius}-diag-{int((speed / 2) * 100)}.gif', fps=25)

        # Display the animation using plt.show() below
        plt.show()

        plt.close()

        i += 1

def setpositioncircle(direction, size, diagonalDirection):
    maxDistOver = 155 # Farthest over circle can go (x or y axis) without going off the board from the animation
    x, y = 0, 0 # Initialize x and y

    if direction == "right":
        x, y = round(random.uniform(size, maxDistOver - size), 2), round(random.uniform(size, 256 - size), 2)
    elif direction == "left":
        x, y = round(random.uniform(256 - maxDistOver + size, 256 - size), 2), round(random.uniform(size, 256 - size), 2)
    elif direction == "up":
        x, y = round(random.uniform(size, 256 - size), 2), round(random.uniform(size, maxDistOver - size), 2)
    elif direction == "down":
        x, y = round(random.uniform(size, 256 - size), 2), round(random.uniform(256 - maxDistOver + size, 256 - size), 2)
    elif direction == "diagonal":
        if diagonalDirection == 1: # Up and right 
            x, y = round(random.uniform(size, maxDistOver - size), 2), round(random.uniform(size, maxDistOver - size), 2)
        elif diagonalDirection == 2: # Down and right
            x, y = round(random.uniform(size, maxDistOver - size), 2), round(random.uniform(256 - maxDistOver + size, 256 - size), 2)
        elif diagonalDirection == 3: # Down and left
            x, y = round(random.uniform(256 - maxDistOver + size, 256 - size), 2), round(random.uniform(256 - maxDistOver + size, 256 - size), 2)
        elif diagonalDirection == 4: # Up and left
            x, y = round(random.uniform(256 - maxDistOver + size, 256 - size), 2), round(random.uniform(size, maxDistOver - size), 2)

    return x, y

def triangle(direction, base, height, speed, iterations):
    # If a value wasn't specified, check a flag that will tell the program to make that value random each iteration
    randDirection, randBase, randHeight, randSpeed = False, False, False, False

    if direction is None:
        randDirection = True
    if base == 0:
        randBase = True
    if height == 0:
        randHeight = True
    if speed == 0:
        randSpeed = True

    i = 1

    while i <= iterations:

        # Set a random value for the parameters we want randomized for each animation
        if randDirection:
            direction = random.choice(["right", "left", "up", "down", "diagonal"])
        if randBase:
            base = random.randint(10, 77)
        if randSpeed:
            speed = random.uniform(0.4, 2)

        # Create a figure that's 256x256 pixels
        dpi = 142
        fig = plt.figure(figsize=(256/dpi, 256/dpi), dpi=dpi)

        # Create an axis with no axis labels or ticks and a black background
        ax = fig.add_subplot(111)
        ax.axis('off')
        fig.set_facecolor("black")
        ax.set_facecolor("black")


def test():

    # Create a figure and axis with no axis labels or ticks
    fig, ax = plt.subplots()
    # ax.axis('off')

    x1, y1, x2, y2, x3, y3 = 0.8, 0.8, 1.0, 0.8, 0.9, 1.0
    # Create the circle patch
    triangle = plt.Polygon([(x1, y1), (x2, y2), (x3, y3)], fc='black')

    # Add the circle to the axis
    ax.add_patch(triangle)

    # Define the animation function to update the position of the circle
    def update(frame):
        # Calculate the new position of the circle
        x1delta = x1 - frame * 0.01
        y1delta = y1

        x2delta = x2 - frame * 0.01
        y2delta = y2

        x3delta = x3 - frame * 0.01
        y3delta = y3

        # Update the position of the circle patch
        triangle.set_xy([(x1delta, y1delta), (x2delta, y2delta), (x3delta, y3delta)])

    # Create an animation object
    ani = animation.FuncAnimation(fig, update, frames=50, repeat=False, blit=False)

    # Display the animation
    plt.show()


if __name__ == "__main__":
    args = parser.parse_args()
    
    shape = None
    if args.shape is None:
        shape = random.choice(["circle", "rectangle", "triangle"])
    else:
        shape = args.shape

    if shape == "circle":
        circle(args.direction, args.radius, args.speed, args.iterations)
    elif shape == "triangle":
        triangle(args.direction, args.base, args.triheight, args.speed, args.iterations)
    else:
        test()
