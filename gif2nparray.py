import os
import imageio
import numpy as np
import matplotlib.pyplot as plt
import pdb

ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
img_list = os.listdir(ROOT)

def main():
    img = imageio.imread(os.path.join(ROOT, img_list[5]))
    arr = np.array(img)

    arr += generate_noisy_matrix(256, 10)

    plt.imshow(arr, cmap="gray")
    plt.show()
    pdb.set_trace()

def generate_noisy_matrix(n, true_percentage):
    # Initialize a matrix of values from [0, 255]
    random_matrix = np.random.randint(0, 256, size=(n, n), dtype=np.uint8)
    # Boolean mask where 'true_percentage' of the values are taken fromm the matrix and the rest are white
    boolean_mask = np.random.choice([False, True], size=(n, n), p=[1 - true_percentage/100, true_percentage/100])
    # Take the values from the matrix or white based on the mask
    noisy_matrix = np.where(boolean_mask, random_matrix, 0)
    return noisy_matrix

if __name__ == "__main__":
    main()