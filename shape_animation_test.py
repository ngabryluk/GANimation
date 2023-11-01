import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Create a figure and axis with no axis labels or ticks
fig, ax = plt.subplots()
ax.axis('off')

# Define the initial position of the shape
x0, y0 = 0.1, 0.5
radius = 0.05  # Radius of the circle
height, width = 0.3, 0.2

# Create the circle patch
circle = plt.Circle((x0, y0), radius, fc='black')
rect = plt.Rectangle((x0, y0), height, width, fc='black')

# Add the circle to the axis
ax.add_patch(rect)

# Define the animation function to update the position of the circle
def update(frame):
    # Calculate the new position of the circle
    x = x0 + frame * 0.01
    y = y0

    # Update the position of the circle patch
    rect.set_xy((x, y))

# Create an animation object
ani = animation.FuncAnimation(fig, update, frames=70, repeat=False, blit=False)

# Display the animation
plt.show()
