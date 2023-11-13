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
parser.add_argument("shape", choices=["circle", "rectangle", "triangle"], default=random.choice(["circle", "rectangle", "triangle"]),
                    help="Pick the shape that will be used in creating data."
)
parser.add_argument("-r", "--radius", type=float, choices=np.arange(10, 77, 0.01), default=round(random.uniform(5, 40), 2),
                    help="Set the radius of the circle (10.00-77.00 - 2 decimal places)."
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
parser.add_argument("-s", "--speed", type=int, choices=range(40, 400), default=random.randint(40, 400),
                    help="Set the speed of the animations. The parameter this will be plugged into is the amount of "
                    "milliseconds of delay between frames, so the lower the number, the faster the animation will be."
)

def createPatch(direction, size, ax):
    # Define the initial position of the shape based on the direction (we don't want to go out of bounds!)
    x0, y0 = setposition(direction, size)

    # Create the circle patch
    circle = plt.Circle((x0, y0), size, fc='black')

    # Add the circle to the axis
    ax.add_patch(circle)

def circle(direction, size, speed, iterations):

        i = 0

        while i < iterations:
            
            # Create a figure that's 256x256 pixels
            dpi = 142
            fig = plt.figure(figsize=(256/dpi, 256/dpi), dpi=dpi)

            # Create an axis with no axis labels or ticks and a black background
            ax = fig.add_subplot(111)
            # ax.axis('off')
            # fig.set_facecolor("black")
            # ax.set_facecolor("black")

            # Set the plot axis to by 256 x 256 to match the pixels
            ax.set_xlim(0, 256)
            ax.set_ylim(0, 256)

            radius = size  # Radius of the circle

            # Randomly pick which diagonal direction to go (up and right, down and right, down and left, up and left) for the function
            rand = random.randint(1, 4)

            
            # Define the animation function to update the position of the patch
            def right(frame):
                # Calculate the new position of the patch
                x = x0 + frame * 2
                y = y0

                # Save the current frame
                # plt.savefig(f'{DATA}\\circle_right\\circle_right_{i+1}\\circle_right_{i+1}_{frame}.jpg')

                # Update the position of the patch
                circle.set_center((x, y))

            def left(frame):
                # Calculate the new position of the circle
                x = x0 - frame * 2
                y = y0

                # Save the current frame
                # plt.savefig(f'{DATA}\\circle_left\\circle_left_{i+1}\\circle_left_{i+1}_{frame}.jpg')

                # Update the position of the circle patch
                circle.set_center((x, y))

            def up(frame):
                # Calculate the new position of the circle
                x = x0 
                y = y0 + frame * 2

                # Save the current frame
                # plt.savefig(f'{DATA}\\circle_up\\circle_up_{i+1}\\circle_up_{i+1}_{frame}.jpg')

                # Update the position of the circle patch
                circle.set_center((x, y))

            def down(frame):
                # Calculate the new position of the circle
                x = x0
                y = y0 - frame * 2

                # Save the current frame
                # plt.savefig(f'{DATA}\\circle_down\\circle_down_{i+1}\\circle_down_{i+1}_{frame}.jpg')

                # Update the position of the circle patch
                circle.set_center((x, y))
            
            def diagonal(frame):
                
                if rand == 1:
                    x = x0 + frame * 2
                    y = y0 + frame * 2
                elif rand == 2:
                    x = x0 + frame * 2
                    y = y0 - frame * 2
                elif rand == 3:
                    x = x0 - frame * 2
                    y = y0 - frame * 2
                elif rand == 4:
                    x = x0 - frame * 2
                    y = y0 + frame * 2

                circle.set_center((x, y))

            def randomdir(frame):
                # Calculate a random direction for movement
                direction_x = random.uniform(-0.01, 0.01)
                direction_y = random.uniform(-0.01, 0.01)

                # Calculate the new position of the circle
                x = circle.get_center()[0] + direction_x
                y = circle.get_center()[1] + direction_y

                # Update the position of the circle patch
                circle.set_center((x, y))

            # Create an animation object based on the direction given in the argument
            if direction == "right":
                # Define the initial position of the shape based on the direction (we don't want to go out of bounds!)
                x0, y0 = setposition(direction, size, rand)

                # Create the circle patch
                circle = plt.Circle((x0, y0), radius, fc='black')

                # Add the circle to the axis
                ax.add_patch(circle)

                # Create the animation object
                ani = animation.FuncAnimation(fig, right, interval=speed, frames=50, repeat=False, blit=False)
            elif direction == "left":
                # Define the initial position of the shape based on the direction (we don't want to go out of bounds!)
                x0, y0 = setposition(direction, size, rand)

                # Create the circle patch
                circle = plt.Circle((x0, y0), radius, fc='black')

                # Add the circle to the axis
                ax.add_patch(circle)

                # Create the animation object
                ani = animation.FuncAnimation(fig, left, interval=speed, frames=50, repeat=False, blit=False)
            elif direction == "up":
                # Define the initial position of the shape based on the direction (we don't want to go out of bounds!)
                x0, y0 = setposition(direction, size, rand)

                # Create the circle patch
                circle = plt.Circle((x0, y0), radius, fc='black')

                # Add the circle to the axis
                ax.add_patch(circle)

                # Create the animation object
                ani = animation.FuncAnimation(fig, up, interval=speed, frames=50, repeat=False, blit=False)
            elif direction == "down":
                # Define the initial position of the shape based on the direction (we don't want to go out of bounds!)
                x0, y0 = setposition(direction, size, rand)

                # Create the circle patch
                circle = plt.Circle((x0, y0), radius, fc='black')

                # Add the circle to the axis
                ax.add_patch(circle)

                # Create the animation object
                ani = animation.FuncAnimation(fig, down, interval=speed, frames=50, repeat=False, blit=False)
            elif direction == "diagonal":
                # Define the initial position of the shape based on the direction (we don't want to go out of bounds!)
                x0, y0 = setposition(direction, size, rand)

                # Create the circle patch
                circle = plt.Circle((x0, y0), radius, fc='black')

                # Add the circle to the axis
                ax.add_patch(circle)

                # Create the animation object
                ani = animation.FuncAnimation(fig, diagonal, interval=speed, frames=50, repeat=False, blit=False)

            # Display the animation
            plt.show()
            plt.close()

            i += 1

def setposition(direction, size, diagonalDirection):
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

if __name__ == "__main__":
    args = parser.parse_args()
    
    if args.shape == "circle":
        circle(args.direction, args.radius, args.speed, args.iterations)
