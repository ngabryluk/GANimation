import numpy as np
import os
import imageio.v2
import pdb
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt

NUM_VIDEOS = 500 # Total number of "videos" or sets of frames that were generated
NUM_FRAMES = 50 # Number of frames per video

# Idea to try: Use half of the frames and interpolate the frames in between to get back to 50, but save those in between frames to be the ground truth
def load():
    DATA_PATH = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "Data\\dataset"
    )
    SAVE_PATH = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "Data\\save_folder"
        )
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)
        # Reads in the images from our synthetic dataset into a numpy array
        images = os.listdir(DATA_PATH)
        # Iterate through every other frame and save it to 
        frame_pairs = np.empty((((NUM_FRAMES // 2) - 1) * NUM_VIDEOS, 256, 512)) # (12000, 256, 512) 24 pairs per 500 samples
        real_middles = np.empty((((NUM_FRAMES // 2) - 1) * NUM_VIDEOS, 256, 256)) # (12000, 256, 512) 24 middle frames per 500 samples
        pairs_index, middles_index = 0, 0
        for i in range(0, len(images) - 2, 2):
            # If we are at the second to last frame, continue because the frame 2 spots ahead is the next sample
            if (i + 2) % NUM_FRAMES == 0:
                continue
            
            filepath1 = os.path.join(DATA_PATH, images[i])
            filepath2 = os.path.join(DATA_PATH, images[i + 2])
            filepath3 = os.path.join(DATA_PATH, images[i + 1])
            
            arr1 = np.array(imageio.imread(filepath1))
            arr2 = np.array(imageio.imread(filepath2))
            arr3 = np.array(imageio.imread(filepath3))

            arr1 = np.dot(arr1[..., :3], [0.2989, 0.5870, 0.1140])
            arr2 = np.dot(arr2[..., :3], [0.2989, 0.5870, 0.1140])
            arr3 = np.dot(arr3[..., :3], [0.2989, 0.5870, 0.1140])

            arr1 = np.round(arr1, 0)
            arr2 = np.round(arr2, 0)
            arr3 = np.round(arr3, 0)

            frame_pairs[pairs_index] = np.hstack((arr1, arr2))
            pairs_index += 1
            
            real_middles[middles_index] = arr3
            middles_index += 1

        np.save(os.path.join(SAVE_PATH, "frame_pairs.npy"), frame_pairs)
        np.save(os.path.join(SAVE_PATH, "real_middles.npy"), real_middles)
    frame_pairs = np.load(os.path.join(SAVE_PATH, "frame_pairs.npy"))
    real_middles = np.load(os.path.join(SAVE_PATH, "real_middles.npy"))
    return frame_pairs, real_middles


def to_gif(images):
    converted_images = np.clip(images, 0, 255).astype(np.uint8)
    imageio.mimsave("./animation.gif", converted_images, fps=25)
    # return embed.embed_file("./animation.gif")


frame_pairs, real_middles = load()

pdb.set_trace()

# INP = input_images[0].reshape(256, 512, 1)

def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(8*8*256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((8, 8, 256)))
    assert model.output_shape == (None, 8, 8, 256)  # Note: None is the batch size

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 16, 16, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 32, 32, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 64, 64, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 128, 128, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 256, 256, 1)

    return model

generator = make_generator_model()
# noise = tf.random.normal([1, 100])
# generated_image = generator(noise, training=False)
# pdb.set_trace()