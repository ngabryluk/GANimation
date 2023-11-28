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
import imageio.v2 as imageio
import argparse
import os
import random
import time
import pdb

# Path to store the results in
ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Data\\dataset')
TEMP = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Data\\temp')

parser = argparse.ArgumentParser(description="Specify the parameters of the data being created.")
parser.add_argument("-i", "--iterations", type=int, default=1,
                    help="Set the number of animations that will be created."                    
)
parser.add_argument("--shape", choices=["circle", "rectangle", "triangle"], default=None,
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
parser.add_argument("-n", "--noise", type=int, choices=range(0, 101), default=0,
                    help="Set the percentage of noise to be added to the image.")
parser.add_argument("--seed", type=int, default=None,
                    help="Apply a seed to initialize the random number generator.")

def circle(direction, radius, speed, ind, diagonalDirection, ax, noise, include_noise, randNoise):
        
    i = ind # This is the ith iteration of the program

    # Define the initial position of the shape based on the direction (we don't want to go out of bounds!)
    x0, y0 = setpositioncircle(direction, radius, diagonalDirection)

    # Create the circle patch
    circle = plt.Circle((x0, y0), radius, fc='white')

    # Add the circle to the axis
    ax.add_patch(circle)

    # If noise is specified, generate the noisy matrix with that noise
    if noise > 0:
        noise_matrix1 = generate_noisy_matrix(256, noise)
        noise_matrix2 = generate_noisy_matrix(256, noise)
    # Generate the noisy matrix with the random noise if the flag to include noise is True
    elif include_noise:
        noise_matrix1 = generate_noisy_matrix(256, randNoise)
        noise_matrix2 = generate_noisy_matrix(256, randNoise)

    # For 50 frames...
    for j in range(50):
        # Move the circle in the direction given
        if direction == "right":
            x0, y0 = circleright(circle, speed, noise, randNoise, include_noise, x0, y0, radius, j + 1, i)
        elif direction == "left":
            x0, y0 = circleleft(circle, speed, noise, randNoise, include_noise, x0, y0, radius, j + 1, i)
        elif direction == "up":
            x0, y0 = circleup(circle, speed, noise, randNoise, include_noise, x0, y0, radius, j + 1, i)
        elif direction == "down":
            x0, y0 = circledown(circle, speed, noise, randNoise, include_noise, x0, y0, radius, j + 1, i)
        elif direction == "diagonal":
            x0, y0 = circlediagonal(circle, speed, noise, randNoise, include_noise, x0, y0, radius, j + 1, i, diagonalDirection)

    # After the frames have been saved, go back and add noise to those frames if we have noise
    if noise > 0 or include_noise:
        add_noise(noise_matrix1, noise_matrix2, noise)

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

def circleright(circle, speed, noise, randNoise, include_noise, x0, y0, radius, frame, iteration):
    # Calculate the new position of the patch
    x = x0 + ((speed / 100) * 2.0)
    y = y0
    
    # Save the current frame
    if noise > 0:
        plt.savefig(f'{TEMP}\\{iteration+1:05}-{frame:02}-c-r{radius}-r-{speed}-{noise}.jpg')
    elif include_noise:
        plt.savefig(f'{TEMP}\\{iteration+1:05}-{frame:02}-c-r{radius}-r-{speed}-{randNoise}.jpg')
    else:
        plt.savefig(f'{ROOT}\\{iteration+1:05}-{frame:02}-c-r{radius}-r-{speed}-{noise}.jpg')
    
    # Update the position of the patch
    circle.set_center((x, y))

    return x, y

def circleleft(circle, speed, noise, randNoise, include_noise, x0, y0, radius, frame, iteration):
    # Calculate the new position of the circle
    x = x0 - ((speed / 100) * 2.0)
    y = y0

    # Save the current frame
    if noise > 0:
        plt.savefig(f'{TEMP}\\{iteration+1:05}-{frame:02}-c-r{radius}-l-{speed}-{noise}.jpg')
    elif include_noise:
        plt.savefig(f'{TEMP}\\{iteration+1:05}-{frame:02}-c-r{radius}-l-{speed}-{randNoise}.jpg')
    else:
        plt.savefig(f'{ROOT}\\{iteration+1:05}-{frame:02}-c-r{radius}-l-{speed}-{noise}.jpg')
    
    
    # Update the position of the circle patch
    circle.set_center((x, y))

    return x, y

def circleup(circle, speed, noise, randNoise, include_noise, x0, y0, radius, frame, iteration):
    # Calculate the new position of the circle
    x = x0 
    y = y0 + ((speed / 100) * 2.0)

    # Save the current frame
    if noise > 0:
        plt.savefig(f'{TEMP}\\{iteration+1:05}-{frame:02}-c-r{radius}-u-{speed}-{noise}.jpg')
    elif include_noise:
        plt.savefig(f'{TEMP}\\{iteration+1:05}-{frame:02}-c-r{radius}-u-{speed}-{randNoise}.jpg')
    else:
        plt.savefig(f'{ROOT}\\{iteration+1:05}-{frame:02}-c-r{radius}-u-{speed}-{noise}.jpg')
    

    # Update the position of the circle patch
    circle.set_center((x, y))

    return x, y

def circledown(circle, speed, noise, randNoise, include_noise, x0, y0, radius, frame, iteration):
    # Calculate the new position of the circle
    x = x0
    y = y0 - ((speed / 100) * 2.0)

    # Save the current frame
    if noise > 0:
        plt.savefig(f'{TEMP}\\{iteration+1:05}-{frame:02}-c-r{radius}-dwn-{speed}-{noise}.jpg')
    elif include_noise:
        plt.savefig(f'{TEMP}\\{iteration+1:05}-{frame:02}-c-r{radius}-dwn-{speed}-{randNoise}.jpg')
    else:
        plt.savefig(f'{ROOT}\\{iteration+1:05}-{frame:02}-c-r{radius}-dwn-{speed}-{noise}.jpg')

    # Update the position of the circle patch
    circle.set_center((x, y))

    return x, y

def circlediagonal(circle, speed, noise, randNoise, include_noise, x0, y0, radius, frame, iteration, diagonalDirection):
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
    if noise > 0:
        plt.savefig(f'{TEMP}\\{iteration+1:05}-{frame:02}-c-r{radius}-diag-{speed}-{noise}.jpg')
    elif include_noise:
        plt.savefig(f'{TEMP}\\{iteration+1:05}-{frame:02}-c-r{radius}-diag-{speed}-{randNoise}.jpg')
    else:
        plt.savefig(f'{ROOT}\\{iteration+1:05}-{frame:02}-c-r{radius}-diag-{speed}-{noise}.jpg')

    circle.set_center((x, y))
    return x, y


def triangle(direction, base, height, speed, ind, diagonalDirection, ax, noise, include_noise, randNoise):
    
    i = ind # This is the ith iteration of the program

    # Define the initial position of the shape based on the direction (we don't want to go out of bounds!)
    x1, y1, x2, y2, x3, y3 = setpositiontriangle(direction, base, height, diagonalDirection)

    # Create the triangle patch
    tri = plt.Polygon([(x1, y1), (x2, y2), (x3, y3)], fc="white")

    # Add the triangle to the axis
    ax.add_patch(tri)

    # If noise is specified, generate the noisy matrix with that noise
    if noise > 0:
        noise_matrix1 = generate_noisy_matrix(256, noise)
        noise_matrix2 = generate_noisy_matrix(256, noise)
    # Generate the noisy matrix with the random noise if the flag to include noise is True
    elif include_noise:
        noise_matrix1 = generate_noisy_matrix(256, randNoise)
        noise_matrix2 = generate_noisy_matrix(256, randNoise)

    # Create the animation objects and save them
    # For 50 frames...
    for j in range(50):
        # Move the circle in the direction given
        if direction == "right":
            x1, y1, x2, y2, x3, y3 = triangleright(tri, speed, noise, randNoise, include_noise, x1, y1, x2, y2, x3, y3, base, height, j + 1, i)
        elif direction == "left":
            x1, y1, x2, y2, x3, y3 = triangleleft(tri, speed, noise, randNoise, include_noise, x1, y1, x2, y2, x3, y3, base, height, j + 1, i)
        elif direction == "up":
            x1, y1, x2, y2, x3, y3 = triangleup(tri, speed, noise, randNoise, include_noise, x1, y1, x2, y2, x3, y3, base, height, j + 1, i)
        elif direction == "down":
            x1, y1, x2, y2, x3, y3 = triangledown(tri, speed, noise, randNoise, include_noise, x1, y1, x2, y2, x3, y3, base, height, j + 1, i)
        elif direction == "diagonal":
            x1, y1, x2, y2, x3, y3 = trianglediagonal(tri, speed, noise, randNoise, include_noise, x1, y1, x2, y2, x3, y3, base, height, j + 1, i, diagonalDirection)

    # After the frames have been saved, go back and add noise to those frames if we have noise
    if noise > 0 or include_noise:
        add_noise(noise_matrix1, noise_matrix2, noise)

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

def triangleright(tri, speed, noise, randNoise, include_noise, x1, y1, x2, y2, x3, y3, base, height, frame, iteration):
    # Calculate the new position of the triangle
    x1delta = x1 + ((speed / 100) * 2.0)
    y1delta = y1

    x2delta = x2 + ((speed / 100) * 2.0)
    y2delta = y2

    x3delta = x3 + ((speed / 100) * 2.0)
    y3delta = y3

    # Save the current frame
    if noise > 0:
        plt.savefig(f'{TEMP}\\{iteration+1:05}-{frame:02}-t-b{base}h{height}-r-{speed}-{noise}.jpg')
    elif include_noise:
        plt.savefig(f'{TEMP}\\{iteration+1:05}-{frame:02}-t-b{base}h{height}-r-{speed}-{randNoise}.jpg')
    else:
        plt.savefig(f'{ROOT}\\{iteration+1:05}-{frame:02}-t-b{base}h{height}-r-{speed}-{noise}.jpg')

    # Update the position of the triangle patch
    tri.set_xy([(x1delta, y1delta), (x2delta, y2delta), (x3delta, y3delta)])
    return x1delta, y1delta, x2delta, y2delta, x3delta, y3delta

def triangleleft(tri, speed, noise, randNoise, include_noise, x1, y1, x2, y2, x3, y3, base, height, frame, iteration):
    # Calculate the new position of the triangle
    x1delta = x1 - ((speed / 100) * 2.0)
    y1delta = y1

    x2delta = x2 - ((speed / 100) * 2.0)
    y2delta = y2

    x3delta = x3 - ((speed / 100) * 2.0)
    y3delta = y3

    # Save the current frame
    if noise > 0:
        plt.savefig(f'{TEMP}\\{iteration+1:05}-{frame:02}-t-b{base}h{height}-l-{speed}-{noise}.jpg')
    elif include_noise:
        plt.savefig(f'{TEMP}\\{iteration+1:05}-{frame:02}-t-b{base}h{height}-l-{speed}-{randNoise}.jpg')
    else:
        plt.savefig(f'{ROOT}\\{iteration+1:05}-{frame:02}-t-b{base}h{height}-l-{speed}-{noise}.jpg')

    # Update the position of the triangle patch
    tri.set_xy([(x1delta, y1delta), (x2delta, y2delta), (x3delta, y3delta)])
    return x1delta, y1delta, x2delta, y2delta, x3delta, y3delta

def triangleup(tri, speed, noise, randNoise, include_noise, x1, y1, x2, y2, x3, y3, base, height, frame, iteration):
    # Calculate the new position of the triangle
    x1delta = x1
    y1delta = y1 + ((speed / 100) * 2.0)

    x2delta = x2
    y2delta = y2 + ((speed / 100) * 2.0)

    x3delta = x3
    y3delta = y3 + ((speed / 100) * 2.0)

    # Save the current frame
    if noise > 0:
        plt.savefig(f'{TEMP}\\{iteration+1:05}-{frame:02}-t-b{base}h{height}-u-{speed}-{noise}.jpg')
    elif include_noise:
        plt.savefig(f'{TEMP}\\{iteration+1:05}-{frame:02}-t-b{base}h{height}-u-{speed}-{randNoise}.jpg')
    else:
        plt.savefig(f'{ROOT}\\{iteration+1:05}-{frame:02}-t-b{base}h{height}-u-{speed}-{noise}.jpg')

    # Update the position of the triangle patch
    tri.set_xy([(x1delta, y1delta), (x2delta, y2delta), (x3delta, y3delta)])
    return x1delta, y1delta, x2delta, y2delta, x3delta, y3delta

def triangledown(tri, speed, noise, randNoise, include_noise, x1, y1, x2, y2, x3, y3, base, height, frame, iteration):
    # Calculate the new position of the triangle
    x1delta = x1
    y1delta = y1 - ((speed / 100) * 2.0)

    x2delta = x2
    y2delta = y2 - ((speed / 100) * 2.0)

    x3delta = x3
    y3delta = y3 - ((speed / 100) * 2.0)

    # Save the current frame
    if noise > 0:
        plt.savefig(f'{TEMP}\\{iteration+1:05}-{frame:02}-t-b{base}h{height}-dwn-{speed}-{noise}.jpg')
    elif include_noise:
        plt.savefig(f'{TEMP}\\{iteration+1:05}-{frame:02}-t-b{base}h{height}-dwn-{speed}-{randNoise}.jpg')
    else:
        plt.savefig(f'{ROOT}\\{iteration+1:05}-{frame:02}-t-b{base}h{height}-dwn-{speed}-{noise}.jpg')

    # Update the position of the triangle patch
    tri.set_xy([(x1delta, y1delta), (x2delta, y2delta), (x3delta, y3delta)])
    return x1delta, y1delta, x2delta, y2delta, x3delta, y3delta

def trianglediagonal(tri, speed, noise, randNoise, include_noise, x1, y1, x2, y2, x3, y3, base, height, frame, iteration, diagonalDirection):
    # Calculate the new position of the triangle
    if diagonalDirection == 1: # Up and right
        x1delta = x1 + ((speed / 100) * 2.0)
        y1delta = y1 + ((speed / 100) * 2.0)

        x2delta = x2 + ((speed / 100) * 2.0)
        y2delta = y2 + ((speed / 100) * 2.0)

        x3delta = x3 + ((speed / 100) * 2.0)
        y3delta = y3 + ((speed / 100) * 2.0)
    elif diagonalDirection == 2: # Down and right
        x1delta = x1 + ((speed / 100) * 2.0)
        y1delta = y1 - ((speed / 100) * 2.0)

        x2delta = x2 + ((speed / 100) * 2.0)
        y2delta = y2 - ((speed / 100) * 2.0)

        x3delta = x3 + ((speed / 100) * 2.0)
        y3delta = y3 - ((speed / 100) * 2.0)
    elif diagonalDirection == 3: # Down and left
        x1delta = x1 - ((speed / 100) * 2.0)
        y1delta = y1 - ((speed / 100) * 2.0)

        x2delta = x2 - ((speed / 100) * 2.0)
        y2delta = y2 - ((speed / 100) * 2.0)

        x3delta = x3 - ((speed / 100) * 2.0)
        y3delta = y3 - ((speed / 100) * 2.0)
    elif diagonalDirection == 4: # Up and left
        x1delta = x1 - ((speed / 100) * 2.0)
        y1delta = y1 + ((speed / 100) * 2.0)

        x2delta = x2 - ((speed / 100) * 2.0)
        y2delta = y2 + ((speed / 100) * 2.0)

        x3delta = x3 - ((speed / 100) * 2.0)
        y3delta = y3 + ((speed / 100) * 2.0)

    # Save the current frame
    if noise > 0:
        plt.savefig(f'{TEMP}\\{iteration+1:05}-{frame:02}-t-b{base}h{height}-diag-{speed}-{noise}.jpg')
    elif include_noise:
        plt.savefig(f'{TEMP}\\{iteration+1:05}-{frame:02}-t-b{base}h{height}-diag-{speed}-{randNoise}.jpg')
    else:
        plt.savefig(f'{ROOT}\\{iteration+1:05}-{frame:02}-t-b{base}h{height}-diag-{speed}-{noise}.jpg')

    # Update the position of the triangle patch
    tri.set_xy([(x1delta, y1delta), (x2delta, y2delta), (x3delta, y3delta)])
    return x1delta, y1delta, x2delta, y2delta, x3delta, y3delta


def rectangle(direction, width, height, speed, ind, diagonalDirection, ax, noise, include_noise, randNoise):
    
    i = ind # This is the ith iteration of the program

    # Define the initial position of the shape based on the direction (we don't want to go out of bounds!)
    x0, y0 = setpositionrectangle(direction, width, height, diagonalDirection)

    # Create the circle patch
    rect = plt.Rectangle((x0, y0), width, height, fc='white')

    # Add the circle to the axis
    ax.add_patch(rect)

    # If noise is specified, generate the noisy matrix with that noise
    if noise > 0:
        noise_matrix1 = generate_noisy_matrix(256, noise)
        noise_matrix2 = generate_noisy_matrix(256, noise)
    # Generate the noisy matrix with the random noise if the flag to include noise is True
    elif include_noise:
        noise_matrix1 = generate_noisy_matrix(256, randNoise)
        noise_matrix2 = generate_noisy_matrix(256, randNoise)

    # For 50 frames...
    for j in range(50):
        # Move the circle in the direction given
        if direction == "right":
            x0, y0 = rectangleright(rect, speed, noise, randNoise, include_noise, x0, y0, width, height, j + 1, i)
        elif direction == "left":
            x0, y0 = rectangleleft(rect, speed, noise, randNoise, include_noise, x0, y0, width, height, j + 1, i)
        elif direction == "up":
            x0, y0 = rectangleup(rect, speed, noise, randNoise, include_noise, x0, y0, width, height, j + 1, i)
        elif direction == "down":
            x0, y0 = rectangledown(rect, speed, noise, randNoise, include_noise, x0, y0, width, height, j + 1, i)
        elif direction == "diagonal":
            x0, y0 = rectanglediagonal(rect, speed, noise, randNoise, include_noise, x0, y0, width, height, j + 1, i, diagonalDirection)

    # After the frames have been saved, go back and add noise to those frames if we have noise
    if noise > 0 or include_noise:
        add_noise(noise_matrix1, noise_matrix2, noise)

def setpositionrectangle(direction, width, height, diagonalDirection):
    maxDistOver = 155 # Farthest over shape can be (x or y axis) without going off the board from the animation
    x, y = 0, 0 # Initialize x and y

    if direction == "right":
        x, y = round(random.uniform(0, maxDistOver - width), 2), round(random.uniform(0, 256 - height), 2)
    elif direction == 'left':
        x, y = round(random.uniform(maxDistOver, 256 - width), 2), round(random.uniform(0, 256 - height), 2)
    elif direction == 'up':
        x, y = round(random.uniform(0, 256 - width), 2), round(random.uniform(0, maxDistOver - height), 2)
    elif direction == 'down':
        x, y = round(random.uniform(0, 256 - width), 2), round(random.uniform(maxDistOver, 256 - height), 2)
    elif direction == 'diagonal':
        if diagonalDirection == 1:
            x, y = round(random.uniform(0, maxDistOver - width), 2), round(random.uniform(0, maxDistOver - height), 2)
        elif diagonalDirection == 2:
            x, y = round(random.uniform(0, maxDistOver - width), 2), round(random.uniform(maxDistOver, 256 - height), 2)
        elif diagonalDirection == 3:
            x, y = round(random.uniform(maxDistOver, 256 - width), 2), round(random.uniform(maxDistOver, 256 - height), 2)
        elif diagonalDirection == 4:
            x, y = round(random.uniform(maxDistOver, 256 - width), 2), round(random.uniform(0, maxDistOver - height), 2)

    return x, y

def rectangleright(rect, speed, noise, randNoise, include_noise, x0, y0, width, height, frame, iteration):
    # Calculate the new position of the patch
    x = x0 + ((speed / 100) * 2.0)
    y = y0
    
    # Save the current frame
    if noise > 0:
        plt.savefig(f'{TEMP}\\{iteration+1:05}-{frame:02}-r-w{width}h{height}-r-{speed}-{noise}.jpg')
    elif include_noise:
        plt.savefig(f'{TEMP}\\{iteration+1:05}-{frame:02}-r-w{width}h{height}-r-{speed}-{randNoise}.jpg')
    else:
        plt.savefig(f'{ROOT}\\{iteration+1:05}-{frame:02}-r-w{width}h{height}-r-{speed}-{noise}.jpg')
    
    # Update the position of the patch
    rect.set_xy((x, y))

    return x, y

def rectangleleft(rect, speed, noise, randNoise, include_noise, x0, y0, width, height, frame, iteration):
    # Calculate the new position of the patch
    x = x0 - ((speed / 100) * 2.0)
    y = y0

    # Save the current frame
    if noise > 0:
        plt.savefig(f'{TEMP}\\{iteration+1:05}-{frame:02}-r-w{width}h{height}-l-{speed}-{noise}.jpg')
    elif include_noise:
        plt.savefig(f'{TEMP}\\{iteration+1:05}-{frame:02}-r-w{width}h{height}-l-{speed}-{randNoise}.jpg')
    else:
        plt.savefig(f'{ROOT}\\{iteration+1:05}-{frame:02}-r-w{width}h{height}-l-{speed}-{noise}.jpg')
    
    # Update the position of the patch
    rect.set_xy((x, y))

    return x, y

def rectangleup(rect, speed, noise, randNoise, include_noise, x0, y0, width, height, frame, iteration):
    # Calculate the new position of the circle
    x = x0 
    y = y0 + ((speed / 100) * 2.0)

    # Save the current frame
    if noise > 0:
        plt.savefig(f'{TEMP}\\{iteration+1:05}-{frame:02}-r-w{width}h{height}-u-{speed}-{noise}.jpg')
    elif include_noise:
        plt.savefig(f'{TEMP}\\{iteration+1:05}-{frame:02}-r-w{width}h{height}-u-{speed}-{randNoise}.jpg')
    else:
        plt.savefig(f'{ROOT}\\{iteration+1:05}-{frame:02}-r-w{width}h{height}-u-{speed}-{noise}.jpg')
    
    # Update the position of the patch
    rect.set_xy((x, y))

    return x, y

def rectangledown(rect, speed, noise, randNoise, include_noise, x0, y0, width, height, frame, iteration):
    # Calculate the new position of the circle
    x = x0
    y = y0 - ((speed / 100) * 2.0)

    # Save the current frame
    if noise > 0:
        plt.savefig(f'{TEMP}\\{iteration+1:05}-{frame:02}-r-w{width}h{height}-dwn-{speed}-{noise}.jpg')
    elif include_noise:
        plt.savefig(f'{TEMP}\\{iteration+1:05}-{frame:02}-r-w{width}h{height}-dwn-{speed}-{randNoise}.jpg')
    else:
        plt.savefig(f'{ROOT}\\{iteration+1:05}-{frame:02}-r-w{width}h{height}-dwn-{speed}-{noise}.jpg')
    
    # Update the position of the patch
    rect.set_xy((x, y))

    return x, y

def rectanglediagonal(rect, speed, noise, randNoise, include_noise, x0, y0, width, height, frame, iteration, diagonalDirection):
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
    if noise > 0:
        plt.savefig(f'{TEMP}\\{iteration+1:05}-{frame:02}-r-w{width}h{height}-diag-{speed}-{noise}.jpg')
    elif include_noise:
        plt.savefig(f'{TEMP}\\{iteration+1:05}-{frame:02}-r-w{width}h{height}-diag-{speed}-{randNoise}.jpg')
    else:
        plt.savefig(f'{ROOT}\\{iteration+1:05}-{frame:02}-r-w{width}h{height}-diag-{speed}-{noise}.jpg')
    
    # Update the position of the patch
    rect.set_xy((x, y))

    return x, y

def generate_noisy_matrix(n, true_percentage):
    # Initialize a matrix of values from [0, 255]
    random_matrix = np.random.randint(0, 256, size=(n, n), dtype=np.uint8)
    # Boolean mask where 'true_percentage' of the values are taken fromm the matrix and the rest are white
    boolean_mask = np.random.choice([False, True], size=(n, n), p=[1 - true_percentage/100, true_percentage/100])
    # Take the values from the matrix or white based on the mask
    noisy_matrix = np.where(boolean_mask, random_matrix, 0)
    return noisy_matrix

def add_noise(noise_matrix1, noise_matrix2, noise):
    
    # Get a list of the files in the folder where we saved the frames of the animation
    img_list = os.listdir(TEMP)

    for img in img_list:
        # Read the image as a numpy array
        filepath = os.path.join(TEMP, img)
        img_arr = np.array(imageio.imread(filepath))
        
        # Put the image in grayscale to make the shape (256, 256)
        img_arr = np.dot(img_arr[..., :3], [0.2989, 0.5870, 0.1140])
        img_arr = np.round(img_arr, 0)

        shape_mask = np.where(img_arr != 0) # Get a mask of the indexes where the shape is
        
        # Add noise matrix to the img numpy array 
        img_arr += noise_matrix1
        
        # Adjust the matrix so the noise adjustment looks better
        img_arr[shape_mask] -= noise_matrix1[shape_mask]

        img_arr[shape_mask] -= noise_matrix2[shape_mask] # For the shape area
        img_arr = np.clip(img_arr, 0, 255).astype(np.uint8) # Get everything between 0-255

        # Make the figure
        dpi = 142
        fig2 = plt.figure(2, figsize=(256/dpi, 256/dpi), dpi=dpi)
        fig2.set_facecolor('black')
        plt.imshow(img_arr, cmap='gray')
        plt.axis('off')

        fig2.savefig(os.path.join(ROOT, img))

        os.remove(os.path.join(TEMP, img))

        plt.close()

def main(args):
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
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


        # Coin flip to decide if to include noise or not when specified
        include_noise = np.random.choice([True, False])
        
        # The noise to add if include_noise is True
        randNoise = np.random.randint(1, 61)
        
        # Create a figure that's 256x256 pixels
        dpi = 142
        fig = plt.figure(1, figsize=(256/dpi, 256/dpi), dpi=dpi)

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
            shape = random.choice(["circle", "triangle", "rectangle"])
        else:
            shape = args.shape

        # Make a temp directory to store the files before adding noise
        os.makedirs(TEMP)

        if shape == "circle":
            circle(direction, radius, speed, i, diagonalDirection, ax, args.noise, include_noise, randNoise)
        elif shape == "triangle":
            triangle(direction, base, triheight, speed, i, diagonalDirection, ax, args.noise, include_noise, randNoise)
        elif shape == "rectangle":
            rectangle(direction, rectWidth, rectHeight, speed, i, diagonalDirection, ax, args.noise, include_noise, randNoise)

        # Delete that temp directory
        os.rmdir(TEMP)

        plt.close()

if __name__ == "__main__":
    main(parser.parse_args())
