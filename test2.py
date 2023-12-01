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


def to_gif(images):
    converted_images = np.clip(images, 0, 255).astype(np.uint8)
    imageio.mimsave("./animation.gif", converted_images, fps=25)
    # return embed.embed_file("./animation.gif")


frame_pairs, real_middles = load()

pdb.set_trace()

def make_generator_model():
    model = tf.keras.Sequential()
    
    # Downsample a little then upsample
    model.add(layers.Reshape((256, 256, 1), input_shape=(256, 256)))

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
generated_image = generator(frame_pairs[0, :, :, :256], training=False)
pdb.set_trace()

def make_discriminator_model():
    model = tf.keras.Sequential()
    
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[256, 256, 1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation='sigmoid'))

    return model

discriminator = make_discriminator_model()
decision = discriminator(generated_image)
print (decision)

# This method returns a helper function to compute cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

EPOCHS = 10
noise_dim = 100
num_examples_to_generate = 16

# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_step(frame_pairs, real_middles):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        current_frames = frame_pairs[:, :, :, :, :256]
        generated_images = generator(current_frames, training=True)
        
        real_output = discriminator(real_middles, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output, real_middles)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

def train(frame_pairs, real_middles, epochs):
    for epoch in range(epochs):
        start = time.time()
        # pdb.set_trace()
        for i in range(len(frame_pairs)):
            current_frames = frame_pairs[i]
            ground_truth_middles = real_middles[i]

            # Perform a training step for the generator and discriminator
            train_step(current_frames, ground_truth_middles)

        # Produce images for the GIF as you go
        generate_and_save_images(generator, epoch + 1, frame_pairs[:, 0, :, :, :256])

        print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

    # Generate after the final epoch
    generate_and_save_images(generator, epochs, frame_pairs[:, 0, :, :, :256])
    
def generate_and_save_images(model, epoch, test_input):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    predictions = model(test_input, training=False)

    # Create a figure that's 256x256 pixels
    dpi = 142
    fig = plt.figure(1, figsize=(256/dpi, 256/dpi), dpi=dpi)

    # Create an axis with no axis labels or ticks and a black background
    ax = fig.add_subplot(111)
    ax.axis('off')
    ax.set_facecolor("black")

    # Set the plot axis to by 256 x 256 to match the pixels
    ax.set_xlim(0, 256)
    ax.set_ylim(0, 256)

    for i in range(predictions.shape[0]):
        plt.imshow(predictions[i, :, :, 0], cmap='gray')
        plt.axis('off')

    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    plt.close()
    # plt.show()

train(frame_pairs, real_middles, EPOCHS)