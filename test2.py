import numpy as np
import os
import imageio.v2 as imageio
import pdb
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import time

NUM_VIDEOS = 500 # Total number of "videos" or sets of frames that were generated
NUM_FRAMES = 50 # Number of frames per video
BATCH_SIZE = 24

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
        
        # Split into batches (each sample)
        frame_pairs = np.array(np.array_split(frame_pairs, len(frame_pairs) // BATCH_SIZE))
        real_middles = np.array(np.array_split(real_middles, len(real_middles) // BATCH_SIZE))
        
        np.save(os.path.join(SAVE_PATH, "frame_pairs.npy"), frame_pairs)
        np.save(os.path.join(SAVE_PATH, "real_middles.npy"), real_middles)
    frame_pairs = np.load(os.path.join(SAVE_PATH, "frame_pairs.npy"))
    real_middles = np.load(os.path.join(SAVE_PATH, "real_middles.npy"))
    return frame_pairs, real_middles


def to_gif(images, num, fps):
    converted_images = np.clip(images, 0, 255).astype(np.uint8)
    imageio.mimsave(f"./animation-{num: 04}.gif", converted_images, fps=fps)
    # return embed.embed_file("./animation.gif")


frame_pairs, real_middles = load()

# pdb.set_trace()

def downsample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                                kernel_initializer=initializer, use_bias=False))

    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())

    result.add(tf.keras.layers.LeakyReLU())

    return result

def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                        padding='same',
                                        kernel_initializer=initializer,
                                        use_bias=False))

    result.add(tf.keras.layers.BatchNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.ReLU())

    return result

def Generator():
    inputs = tf.keras.layers.Input(shape=[256, 256, 1])

    down_stack = [
        downsample(64, 4, apply_batchnorm=False),  # (batch_size, 128, 128, 64)
        downsample(128, 4),  # (batch_size, 64, 64, 128)
        downsample(256, 4),  # (batch_size, 32, 32, 256)
        downsample(512, 4),  # (batch_size, 16, 16, 512)
    ]

    up_stack = [
        upsample(256, 4),  # (batch_size, 32, 32, 512)
        upsample(128, 4),  # (batch_size, 64, 64, 256)
        upsample(64, 4),  # (batch_size, 128, 128, 128)
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(1, 4,
                                            strides=2,
                                            padding='same',
                                            kernel_initializer=initializer,
                                            activation='tanh')  # (batch_size, 256, 256, 1)

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
generated_image = generator(frame_pairs[0, :, :, :256], training=False)
# pdb.set_trace()

def make_discriminator_model():
    model = tf.keras.Sequential()
    
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[256, 256, 1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model

discriminator = make_discriminator_model()
decision = discriminator(generated_image)
# print (decision)

# This method returns a helper function to compute cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output, real_middles):
    real_middles = tf.cast(real_middles, dtype=tf.float32)

    # pdb.set_trace()
    
    # Calculate the mean absolute error (MAE) between generated frames and ground truth middles
    mae = tf.reduce_mean(tf.abs(fake_output[:, :, :, 0] - real_middles))
    return mae

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

EPOCHS = 100
noise_dim = 100

# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_step(frame_pairs, real_middles):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        current_frames = frame_pairs[:, :, :256]
        generated_images = generator(current_frames, training=True)
        
        real_output = discriminator(real_middles, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(generated_images, real_middles)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

checkpoint_dir = 'Data\\training_checkpoints'
checkpoint_prefix = os.path.join(os.path.dirname(os.path.dirname(__file__)), checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

def train(frame_pairs, real_middles, epochs):
    for epoch in range(epochs):
        start = time.time()

        for i in range(len(frame_pairs)):
            current_frames = frame_pairs[i]
            ground_truth_middles = real_middles[i]

            # Perform a training step for the generator and discriminator
            train_step(current_frames, ground_truth_middles)

        # Produce images for the GIF as you go
        generate_and_save_images(generator, epoch + 1, frame_pairs[0, :, :, :256])

        # Save the model every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)

        print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

    # Generate after the final epoch
    generate_and_save_images(generator, epochs, frame_pairs[0, :, :, :256])

TEST_IMG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Data\\test_images")

def generate_and_save_images(model, epoch, test_input):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    predictions = model(test_input, training=False)
    predictions = np.array(predictions).reshape((25, 256, 256))
    # pdb.set_trace()
    
    dpi = 142
    for i in range(predictions.shape[0]):
        # Create a figure that's 256x256 pixels
        fig = plt.figure(figsize=(256/dpi, 256/dpi), dpi=dpi)
        ax = fig.add_subplot(111)
        ax.axis('off')
        fig.set_facecolor("black")
        ax.set_facecolor("black")
        plt.imshow(predictions[i, :, :], cmap='gray')

        plt.savefig(os.path.join(DVD_SAVE_PATH, f'img{i:04}.png'))
        plt.close()

    # to_gif(np.asarray([test_input, predictions[0, :, :, 0]]), epoch)
    # plt.show()


# train(frame_pairs, real_middles, EPOCHS)

checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
# pdb.set_trace()

DVD_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Data\\test_images")
DVD_SAVE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Data\\test_images_dvd")

from PIL import Image
# num_key_frames = 50
# with Image.open(os.path.join(os.path.dirname(TEST_IMG_PATH), 'dvd.gif')) as im:
#     for i in range(num_key_frames):
#         im.seek(im.n_frames // num_key_frames * i)
#         im.save(os.path.join(TEST_IMG_PATH, f'img{i:04}.png'))

img_list = os.listdir(DVD_PATH)
frameDVD = np.empty((25, 256, 256))
middleDVD = np.empty((25, 256, 256))
allDVD = np.empty((50, 256, 256))
frame_index, middles_index = 0, 0
for i in range(0, len(img_list), 2):
    # Read the image as a numpy array
    filepath1 = os.path.join(DVD_PATH, img_list[i])
    filepath2 = os.path.join(DVD_PATH, img_list[i + 1])
    
    img1 = Image.open(filepath1).convert("L")  # Open the image and convert to grayscale
    img2 = Image.open(filepath2).convert("L")

    # pdb.set_trace()

    arr1 = np.array(img1)
    arr2 = np.array(img2)

    # pdb.set_trace()

    frameDVD[frame_index] = arr1
    middleDVD[middles_index] = arr2
    allDVD[i] = arr1
    allDVD[i+1] = arr2

    frame_index += 1
    middles_index += 1

generate_and_save_images(generator, 1, frameDVD)
pdb.set_trace()

######################################################
# THE ARRAY IS ALL 0s, FIGURE OUT WHY THAT IS THE CASE
######################################################
