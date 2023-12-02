import numpy as np
import os
import imageio.v2
import pdb
import tensorflow as tf
import matplotlib.pyplot as plt


def load():
    DATA_PATH = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "Data\\dataset"
    )
    SAVE_PATH = os.path.join(os.getcwd(), "save_folder")
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)
        # Reads in the images from our synthetic dataset into a numpy array
        images = os.listdir(DATA_PATH)
        input_images = np.empty((len(images) - 2, 256, 512))
        real_images = np.empty((len(images) - 2, 256, 256))
        for i in range(len(images) - 2):
            filepath1 = os.path.join(DATA_PATH, images[i])
            filepath2 = os.path.join(DATA_PATH, images[i + 2])
            
            arr1 = np.array(imageio.imread(filepath1))
            arr2 = np.array(imageio.imread(filepath2))

            arr1 = np.dot(arr1[..., :3], [0.2989, 0.5870, 0.1140])
            arr2 = np.dot(arr2[..., :3], [0.2989, 0.5870, 0.1140])

            arr1 = np.round(arr1, 0)
            arr2 = np.round(arr2, 0)

            input_images[i] = np.hstack((arr1, arr2))
            if i != 0:
                real_images[i - 1] = arr1
        np.save(os.path.join(SAVE_PATH, "input_images.npy"), input_images)
        np.save(os.path.join(SAVE_PATH, "real_images.npy"), real_images)
    input_images = np.load(os.path.join(SAVE_PATH, "input_images.npy"))
    real_images = np.load(os.path.join(SAVE_PATH, "real_images.npy"))
    return input_images, real_images


def to_gif(images):
    converted_images = np.clip(images, 0, 255).astype(np.uint8)
    imageio.mimsave("./animation.gif", converted_images, fps=25)
    # return embed.embed_file("./animation.gif")


input_images, real_images = load()

INP = input_images[0].reshape(256, 512, 1)

OUTPUT_CHANNELS = 3

def downsample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0.0, 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2D(
            filters,
            size,
            strides=2,
            padding="same",
            kernel_initializer=initializer,
            use_bias=False,
        )
    )

    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())

    result.add(tf.keras.layers.LeakyReLU())

    return result


down_model = downsample(3, 4)
down_result = down_model(tf.expand_dims(INP, 0))
print(down_result.shape)

def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0.0, 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2DTranspose(
            filters,
            size,
            strides=2,
            padding="same",
            kernel_initializer=initializer,
            use_bias=False,
        )
    )

    result.add(tf.keras.layers.BatchNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))
        
    result.add(tf.keras.layers.ReLU())

    return result

up_model = upsample(3, 4)
up_result = up_model(down_result)
print(up_result.shape)

def Generator():
    inputs = tf.keras.layers.Input(shape=[256, 512, 3])

    down_stack = [
        downsample(64, 4, apply_batchnorm=False),  # (batch_size, 128, 128, 64)
        downsample(128, 4),  # (batch_size, 64, 64, 128)
        downsample(256, 4),  # (batch_size, 32, 32, 256)
        downsample(512, 4),  # (batch_size, 16, 16, 512)
        downsample(512, 4),  # (batch_size, 8, 8, 512)
        downsample(512, 4),  # (batch_size, 4, 4, 512)
        downsample(512, 4),  # (batch_size, 2, 2, 512)
        downsample(512, 4),  # (batch_size, 1, 1, 512)
    ]

    up_stack = [
        upsample(512, 4, apply_dropout=True),  # (batch_size, 2, 2, 1024)
        upsample(512, 4, apply_dropout=True),  # (batch_size, 4, 4, 1024)
        upsample(512, 4, apply_dropout=True),  # (batch_size, 8, 8, 1024)
        upsample(512, 4),  # (batch_size, 16, 16, 1024)
        upsample(256, 4),  # (batch_size, 32, 32, 512)
        upsample(128, 4),  # (batch_size, 64, 64, 256)
        upsample(64, 4),  # (batch_size, 128, 128, 128)
    ]

    initializer = tf.random_normal_initializer(0.0, 0.02)
    last = tf.keras.layers.Conv2DTranspose(
        OUTPUT_CHANNELS,
        4,
        strides=2,
        padding="same",
        kernel_initializer=initializer,
        activation="tanh",
    )  # (batch_size, 256, 256, 3)

    x = inputs

    # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])

    x = last(x)
    return tf.keras.Model(inputs=inputs, outputs=x)

generator = Generator()
tf.keras.utils.plot_model(generator, show_shapes=True, dpi=64)

# gen_output = generator(INP[tf.newaxis, ...], training=False)
# plt.imshow(gen_output[0, ...])