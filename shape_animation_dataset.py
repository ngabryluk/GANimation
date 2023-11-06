# shape_animation_dataset.py
# Used to create a dataset of shapes moving on a simple background

# Checklist
#   - Directions
#       - left to right
#       - right to left
#       - top to bottom
#       - bottom to top
#   - Shapes
#       - Circle
#       - Rectangle
#       - Triangle (A little more difficult to setup with randomizing)

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random

# Create a figure and axis with no axis labels or ticks and a black background
fig, ax = plt.subplots()
# ax.axis('off')
# fig.set_facecolor("black")
# ax.set_facecolor("black")

# List of shapes to randomize
shapes = ['circle', 'rectangle', 'triangle']

# Pick a random shape
shape = shapes[random.randint(0, len(shapes) - 2)]

radius = 0.1  # Radius of the circle
width, height = 0.3, 0.2 # Height and width of the rectangle

# Define the initial position of the shapes randomly based on shape and so that it stays in bounds
x0, y0 = 0, 0
if shape == 'circle':
    x0, y0 = random.random() * (1 - (radius * 2)) + 1, random.random() * (1 - (radius * 2))
elif shape == 'rectangle':
    x0, y0 = random.random() * (1 - width), random.random() * (1 - height)
# elif shape == 'triangle':
#     maxBase = 0.3
#     pt1 = 

# Create the circle patch
circle = plt.Circle((x0, y0), radius, fc='black')
rect = plt.Rectangle((x0, y0), width, height, fc='black')
poly = plt.Polygon(([(0.1, 0.1), (0.9, 0.1), (0.5, 0.8)]), facecolor='red', edgecolor='black')

# Add the shape to the axis
if shape == 'circle':
    ax.add_patch(circle)
elif shape == 'rectangle':
    ax.add_patch(rect)
plt.show()

# Define the animation function to update the position of the circle
def update(frame):
    # Calculate the new position of the circle
    x = x0 + frame * 0.01
    y = y0

    # Update the position of the circle patch
    poly.set_xy((x, y))

# Create an animation object
ani = animation.FuncAnimation(fig, update, frames=50, interval=50, repeat=False, blit=False)

# Display the animation
plt.show()
