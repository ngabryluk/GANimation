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
import imageio
import argparse
import os
import random
import pdb

# Path to store the results in
ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
TEST = os.path.join(os.path.dirname(os.path.dirname(__file__)), "test")

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
parser.add_argument("-s", "--speed", type=int, choices=range(20, 101), default=0,
                    help="Set the speed of the animations. This is percentage of the maximum speed of 2 pixels per frame."
)
parser.add_argument("-n", "--noise", type=int, choices=range(1, 101), default=0,
                    help="Set the percentage of noise to be added to the image.")

def circle(direction, radius, speed, ind, diagonalDirection, fig, ax, noise):
        
        i = ind # This is the ith iteration of the program

        # Define the initial position of the shape based on the direction (we don't want to go out of bounds!)
        x0, y0 = setpositioncircle(direction, radius, diagonalDirection)

        # Create the circle patch
        circle = plt.Circle((x0, y0), radius, fc='white')

        # Add the circle to the axis
        ax.add_patch(circle)

        # Need to add noise to the the whole figure
        # It seems that the noise is being added behind the patch

        # plt.savefig("fig.png")
        # arr = plt.imread("fig.png")

        # pdb.set_trace()

        # plt.imshow(arr, cmap="gray_r")
        # plt.show()

        # # pdb.set_trace()

        # if os.path.exists("fig.png"):
        #     os.remove("fig.png")
        # else:
        #     print("The file does not exist")


        # For 50 frames...
        for j in range(50):
            # Create the animation objects and save them
            if direction == "right":
                circleright(circle, speed, noise, x0, y0, radius, j + 1, i)
                # ani = animation.FuncAnimation(fig, right, interval=75, frames=50, repeat=False, blit=False)
                # ani.save(f'{ROOT}\\{i:03}-c-r{radius}-r-{speed}.gif', fps=25)
            elif direction == "left":
                circleleft(speed, noise, x0, y0, radius, j + 1, i)
                # ani = animation.FuncAnimation(fig, left, interval=75, frames=50, repeat=False, blit=False)
                # ani.save(f'{ROOT}\\{i:03}-c-r{radius}-l-{speed}.gif', fps=25)
            elif direction == "up":
                circleup(speed, noise, x0, y0, radius, j + 1, i)
                # ani = animation.FuncAnimation(fig, up, interval=75, frames=50, repeat=False, blit=False)
                # ani.save(f'{ROOT}\\{i:03}-c-r{radius}-u-{speed}.gif', fps=25)
            elif direction == "down":
                circledown(speed, noise, x0, y0, radius, j + 1, i)
                # ani = animation.FuncAnimation(fig, down, interval=75, frames=50, repeat=False, blit=False)
                # ani.save(f'{ROOT}\\{i:03}-c-r{radius}-dwn-{speed}.gif', fps=25)
            elif direction == "diagonal":
                circlediagonal(speed, noise, x0, y0, radius, j + 1, i, diagonalDirection)
                # ani = animation.FuncAnimation(fig, diagonal, interval=75, frames=50, repeat=False, blit=False)
                # ani.save(f'{ROOT}\\{i:03}-c-r{radius}-diag-{speed}.gif', fps=25)

        ####################################
        # Add noise to images if specified #
        ####################################

        if noise > 0:
            # Get a list of the files in the folder where we saved the frames of the animation
            img_list = os.listdir(TEST)

            for img in img_list:
                # Read the image as a numpy array
                img_arr = imageio.imread(img)
                img_arr = np.array(img_arr)

                # Add noise to this 
                img_arr += generate_noisy_matrix(256, noise)
                plt.imshow(img_arr, cmap="gray_r")
                plt.savefig(os.path.join(TEST, img))

        plt.close()

def setpositioncircle(direction, radius, diagonalDirection):
    maxDistOver = 155 # Farthest over shape can be (x or y axis) without going off the board from the animation
    x, y = 0, 0 # Initialize x and y

    if direction == "right":
        x, y = round(random.uniform(radius, maxDistOver - radius), 2), round(random.uniform(radius, 256 - radius), 2)
    elif direction == "left":
        x, y = round(random.uniform(256 - maxDistOver + radius, 256 - radius), 2), round(random.uniform(radius, 256 - radius), 2)
    elif direction == "up":
        x, y = round(random.uniform(radius, 256 - radius), 2), round(random.uniform(radius, maxDistOver - radius), 2)
    elif direction == "down":
        x, y = round(random.uniform(radius, 256 - radius), 2), round(random.uniform(256 - maxDistOver + radius, 256 - radius), 2)
    elif direction == "diagonal":
        if diagonalDirection == 1: # Up and right 
            x, y = round(random.uniform(radius, maxDistOver - radius), 2), round(random.uniform(radius, maxDistOver - radius), 2)
        elif diagonalDirection == 2: # Down and right
            x, y = round(random.uniform(radius, maxDistOver - radius), 2), round(random.uniform(256 - maxDistOver + radius, 256 - radius), 2)
        elif diagonalDirection == 3: # Down and left
            x, y = round(random.uniform(256 - maxDistOver + radius, 256 - radius), 2), round(random.uniform(256 - maxDistOver + radius, 256 - radius), 2)
        elif diagonalDirection == 4: # Up and left
            x, y = round(random.uniform(256 - maxDistOver + radius, 256 - radius), 2), round(random.uniform(radius, maxDistOver - radius), 2)

    return x, y

# Define the animation function to update the position of the patch
def circleright(circle, speed, noise, x0, y0, radius, frame, iteration):
    # Calculate the new position of the patch
    x = x0 + ((speed / 100) * 2.0)
    y = y0

    # Save the current frame
    plt.savefig(f'{TEST}\\{iteration+1:03}-{frame:02}-c-r{radius}-r-{speed}-{noise}.jpg')

    # Update the position of the patch
    circle.set_center((x, y))

def circleleft(speed, x0, y0, radius, frame):
    # Calculate the new position of the circle
    x = x0 - ((speed / 100) * 2.0)
    y = y0

    # Save the current frame
    # plt.savefig(f'{ROOT}\\{iteration:03}-c-r{radius}-l-{speed}-{frame:02}.jpg')

    # Update the position of the circle patch
    circle.set_center((x, y))

def circleup(speed, x0, y0, radius, frame):
    # Calculate the new position of the circle
    x = x0 
    y = y0 + ((speed / 100) * 2.0)

    # Save the current frame
    # plt.savefig(f'{ROOT}\\{iteration:03}-c-r{radius}-u-{speed}-{frame:02}.jpg')

    # Update the position of the circle patch
    circle.set_center((x, y))

def circledown(speed, x0, y0, radius, frame):
    # Calculate the new position of the circle
    x = x0
    y = y0 - ((speed / 100) * 2.0)

    # Save the current frame
    # plt.savefig(f'{ROOT}\\c-r{radius}-dwn-{speed}-{frame:03}.jpg')

    # Update the position of the circle patch
    circle.set_center((x, y))

def circlediagonal(speed, x0, y0, radius, frame, diagonalDirection):
    # Update x and y based on the random diagonal direction picked
    if diagonalDirection == 1: # Up and right
        x = x0 + ((speed / 100) * 2.0)
        y = y0 + ((speed / 100) * 2.0)
    elif diagonalDirection == 2: # Down and right
        x = x0 + ((speed / 100) * 2.0)
        y = y0 - ((speed / 100) * 2.0)
    elif diagonalDirection == 3: # Down and left
        x = x0 - ((speed / 100) * 2.0)
        y = y0 - ((speed / 100) * 2.0)
    elif diagonalDirection == 4: # Up and left
        x = x0 - ((speed / 100) * 2.0)
        y = y0 + ((speed / 100) * 2.0)

    # Save the current frame
    # plt.savefig(f'{ROOT}\\c-r{radius}-diag-{speed}-{frame:03}.jpg')

    circle.set_center((x, y))

def triangle(direction, base, height, speed, ind, diagonalDirection, fig, ax, noise):
    
    i = ind # This is the ith iteration of the program

    # Define the initial position of the shape based on the direction (we don't want to go out of bounds!)
    x1, y1, x2, y2, x3, y3 = setpositiontriangle(direction, base, height, diagonalDirection)

    # Create the triangle patch
    tri = plt.Polygon([(x1, y1), (x2, y2), (x3, y3)], fc="white")

    # Add the triangle to the axis
    ax.add_patch(tri)

    def right(frame):
        # Calculate the new position of the triangle
        x1delta = x1 + frame * ((speed / 100) * 2.0)
        y1delta = y1

        x2delta = x2 + frame * ((speed / 100) * 2.0)
        y2delta = y2

        x3delta = x3 + frame * ((speed / 100) * 2.0)
        y3delta = y3

        # Update the position of the triangle patch
        tri.set_xy([(x1delta, y1delta), (x2delta, y2delta), (x3delta, y3delta)])

    def left(frame):
        # Calculate the new position of the triangle
        x1delta = x1 - frame * ((speed / 100) * 2.0)
        y1delta = y1

        x2delta = x2 - frame * ((speed / 100) * 2.0)
        y2delta = y2

        x3delta = x3 - frame * ((speed / 100) * 2.0)
        y3delta = y3

        # Update the position of the triangle patch
        tri.set_xy([(x1delta, y1delta), (x2delta, y2delta), (x3delta, y3delta)])

    def up(frame):
        # Calculate the new position of the triangle
        x1delta = x1
        y1delta = y1 + frame * ((speed / 100) * 2.0)

        x2delta = x2
        y2delta = y2 + frame * ((speed / 100) * 2.0)

        x3delta = x3
        y3delta = y3 + frame * ((speed / 100) * 2.0)

        # Update the position of the triangle patch
        tri.set_xy([(x1delta, y1delta), (x2delta, y2delta), (x3delta, y3delta)])

    def down(frame):
        # Calculate the new position of the triangle
        x1delta = x1
        y1delta = y1 - frame * ((speed / 100) * 2.0)

        x2delta = x2
        y2delta = y2 - frame * ((speed / 100) * 2.0)

        x3delta = x3
        y3delta = y3 - frame * ((speed / 100) * 2.0)

        # Update the position of the triangle patch
        tri.set_xy([(x1delta, y1delta), (x2delta, y2delta), (x3delta, y3delta)])

    def diagonal(frame):
        # Calculate the new position of the triangle
        if diagonalDirection == 1: # Up and right
            x1delta = x1 + frame * ((speed / 100) * 2.0)
            y1delta = y1 + frame * ((speed / 100) * 2.0)

            x2delta = x2 + frame * ((speed / 100) * 2.0)
            y2delta = y2 + frame * ((speed / 100) * 2.0)

            x3delta = x3 + frame * ((speed / 100) * 2.0)
            y3delta = y3 + frame * ((speed / 100) * 2.0)
        elif diagonalDirection == 2: # Down and right
            x1delta = x1 + frame * ((speed / 100) * 2.0)
            y1delta = y1 - frame * ((speed / 100) * 2.0)

            x2delta = x2 + frame * ((speed / 100) * 2.0)
            y2delta = y2 - frame * ((speed / 100) * 2.0)

            x3delta = x3 + frame * ((speed / 100) * 2.0)
            y3delta = y3 - frame * ((speed / 100) * 2.0)
        elif diagonalDirection == 3: # Down and left
            x1delta = x1 - frame * ((speed / 100) * 2.0)
            y1delta = y1 - frame * ((speed / 100) * 2.0)

            x2delta = x2 - frame * ((speed / 100) * 2.0)
            y2delta = y2 - frame * ((speed / 100) * 2.0)

            x3delta = x3 - frame * ((speed / 100) * 2.0)
            y3delta = y3 - frame * ((speed / 100) * 2.0)
        elif diagonalDirection == 4: # Up and left
            x1delta = x1 - frame * ((speed / 100) * 2.0)
            y1delta = y1 + frame * ((speed / 100) * 2.0)

            x2delta = x2 - frame * ((speed / 100) * 2.0)
            y2delta = y2 + frame * ((speed / 100) * 2.0)

            x3delta = x3 - frame * ((speed / 100) * 2.0)
            y3delta = y3 + frame * ((speed / 100) * 2.0)

        # Update the position of the triangle patch
        tri.set_xy([(x1delta, y1delta), (x2delta, y2delta), (x3delta, y3delta)])
    
    # Create the animation objects and save them
    if direction == "right":
        ani = animation.FuncAnimation(fig, right, interval=75, frames=50, repeat=False, blit=False)
        ani.save(f'{ROOT}\\{i:03}-t-b{base}h{height}-r-{speed}.gif', fps=25)
    elif direction == "left":
        ani = animation.FuncAnimation(fig, left, interval=75, frames=50, repeat=False, blit=False)
        ani.save(f'{ROOT}\\{i:03}-t-b{base}h{height}-l-{speed}.gif', fps=25)
    elif direction == "up":
        ani = animation.FuncAnimation(fig, up, interval=75, frames=50, repeat=False, blit=False)
        ani.save(f'{ROOT}\\{i:03}-t-b{base}h{height}-u-{speed}.gif', fps=25)
    elif direction == "down":
        ani = animation.FuncAnimation(fig, down, interval=75, frames=50, repeat=False, blit=False)
        ani.save(f'{ROOT}\\{i:03}-t-b{base}h{height}-dwn-{speed}.gif', fps=25)
    elif direction == "diagonal":
        ani = animation.FuncAnimation(fig, diagonal, interval=75, frames=50, repeat=False, blit=False)
        ani.save(f'{ROOT}\\{i:03}-t-b{base}h{height}-diag-{speed}.gif', fps=25)

    # Display the animation using plt.show() below
    # plt.show()

    plt.close()

def setpositiontriangle(direction, base, height, diagonalDirection):
    maxDistOver = 155 # Farthest over shape can be (x or y axis) without going off the board from the animation
    x, y = 0, 0 # Initialize x and y

    if direction == 'right':
        x1, y1 = round(random.uniform(0, maxDistOver - base), 2), round(random.uniform(0, 256 - height), 2)
        x2, y2 = x1 + base, y1
        x3, y3 = x1 + (base / 2), y2 + height
    elif direction == 'left':
        x2, y2 = round(random.uniform(256 - maxDistOver + base, 256), 2), round(random.uniform(0, 256 - height), 2)
        x1, y1 = x2 - base, y2
        x3, y3 = x1 + (base / 2), y2 + height
    elif direction == 'up':
        x1, y1 = round(random.uniform(0, 256 - base), 2), round(random.uniform(0, maxDistOver - height), 2)
        x2, y2 = x1 + base, y1
        x3, y3 = x1 + (base / 2), y2 + height
    elif direction == 'down':
        x1, y1 = round(random.uniform(0, 256 - base), 2), round(random.uniform(256 - maxDistOver, 256 - height), 2)
        x2, y2 = x1 + base, y1
        x3, y3 = x1 + (base / 2), y2 + height
    elif direction == 'diagonal':
        if diagonalDirection == 1:
            x1, y1 = round(random.uniform(0, maxDistOver - base), 2), round(random.uniform(0, maxDistOver - height), 2)
            x2, y2 = x1 + base, y1
            x3, y3 = x1 + (base / 2), y2 + height
        elif diagonalDirection == 2:
            x1, y1 = round(random.uniform(0, maxDistOver - base), 2), round(random.uniform(256 - maxDistOver, 256 - height), 2)
            x2, y2 = x1 + base, y1
            x3, y3 = x1 + (base / 2), y2 + height
        elif diagonalDirection == 3:
            x2, y2 = round(random.uniform(256 - maxDistOver + base, 256), 2), round(random.uniform(256 - maxDistOver, 256 - height), 2)
            x1, y1 = x2 - base, y2
            x3, y3 = x1 + (base / 2), y2 + height
        elif diagonalDirection == 4:
            x2, y2 = round(random.uniform(256 - maxDistOver + base, 256), 2), round(random.uniform(0, maxDistOver - height), 2)
            x1, y1 = x2 - base, y2
            x3, y3 = x1 + (base / 2), y2 + height
    
    return x1, y1, x2, y2, x3, y3

def generate_noisy_matrix(n, true_percentage):
    # Initialize a matrix of values from [0, 255]
    random_matrix = np.random.randint(0, 256, size=(n, n), dtype=np.uint8)
    # Boolean mask where 'true_percentage' of the values are taken fromm the matrix and the rest are white
    boolean_mask = np.random.choice([False, True], size=(n, n), p=[1 - true_percentage/100, true_percentage/100])
    # Take the values from the matrix or white based on the mask
    noisy_matrix = np.where(boolean_mask, random_matrix, 255)
    return noisy_matrix

def main(args):
    for i in range(args.iterations):
        # If a value wasn't specified, check a flag that will tell the program to make that value random each iteration
        randDirection, randRadius, randBase, randTriangleHeight, randWidth, randRectangleHeight, randSpeed = False, False, False, False, False, False, False

        if args.direction is None:
            randDirection = True
        if args.radius == 0:
            randRadius = True
        if args.base == 0:
            randBase = True
        if args.triheight == 0:
            randTriangleHeight = True
        if args.rectwidth == 0:
            randWidth = True
        if args.rectheight == 0:
            randRectangleHeight = True
        if args.speed == 0:
            randSpeed = True

        # Set a random value for the parameters we want randomized for each animation, or what we specified it to be
        if randDirection:
            direction = random.choice(["right", "left", "up", "down", "diagonal"])
        else:
            direction = args.direction

        if randRadius:
            radius = random.randint(10, 77)
        else:
            radius = args.radius

        if randBase:
            base = random.randint(10, 75)
        else:
            base = args.base

        if randTriangleHeight:
            triheight = random.randint(10, 75)
        else:
            triheight = args.triheight

        if randWidth:
            rectWidth = random.randint(10, 75)
        else:
            rectWidth = args.rectWidth

        if randRectangleHeight:
            rectHeight = random.randint(10, 75)
        else:
            rectHeight = args.rectHeight

        if randSpeed:
            speed = random.randint(20, 100)
        else:
            speed = args.speed
        
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

        # Randomly pick which diagonal direction to go (up and right, down and right, down and left, up and left) for the function
        diagonalDirection = random.randint(1, 4)

        shape = None
        if args.shape is None:
            shape = random.choice(["circle", "rectangle", "triangle"])
        else:
            shape = args.shape

        if shape == "circle":
            circle(direction, radius, speed, i, diagonalDirection, fig, ax, args.noise)
        elif shape == "triangle":
            triangle(direction, base, triheight, speed, i, diagonalDirection, fig, ax, args.noise)

if __name__ == "__main__":
    main(parser.parse_args())

    
