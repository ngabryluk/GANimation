import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from PIL import Image
from keras import layers
import tensorflow as tf
import time
from IPython import display
import pdb

ROOT = os.path.dirname(__file__)
DATA_ROOT = f'{ROOT}/cats'
# TODO: Probably need to change these two constants
BUFFER_SIZE = 5500
BATCH_SIZE = 4
HEIGHT = 512
WIDTH = 512
TRAINING_DIST = 0.3
LIGHT_MODE = True
# This method returns a helper function to compute cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
EPOCHS = 50
noise_dim = 100
num_examples_to_generate = 16
# You will reuse this seed overtime (so it's easier)
# to visualize progress in the animated GIF)
seed = tf.random.normal([num_examples_to_generate, noise_dim])
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")


class Model:
    def __init__(self, k=(5, 5), s=(2, 2)):
        self.model = tf.keras.Sequential()
        self.optimizer = tf.keras.optimizers.Adam(1e-4)
        self.kernel = k
        self.stride = s

    def get_model(self):
        return self.model

    def get_optimizer(self):
        return self.optimizer

    def get_kernel(self):
        return self.kernel

    def get_strides(self):
        return self.stride

    def set_optimizer(self, new_optimizer):
        self.optimizer = new_optimizer
        return self

    def set_kernel(self, new_kernel):
        self.kernel = new_kernel
        return self

    def set_strides(self, new_strides):
        self.stride = new_strides
        return self


class Generator(Model):
    def __init__(self, k=(5, 5), s=(2, 2)):
        super().__init__(k, s)

    # TODO: Probably extract most of these numbers to the class, so they can be specified upon creation
    # TODO: Work through this, figure out what everything does, and comment it
    def make_model(self):
        # WTF is up with 100!?!?
        self.model.add(layers.Dense((HEIGHT // 4) * (WIDTH // 4)
                       * BATCH_SIZE, use_bias=True, input_shape=(100,)))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.ReLU())
        self.model.add(layers.Reshape((HEIGHT // 4, WIDTH // 4, BATCH_SIZE)))
        # Note: None is the batch size
        assert self.model.output_shape == (
            None, HEIGHT // 4, WIDTH // 4, BATCH_SIZE)
        self.model.add(layers.Conv2DTranspose(BATCH_SIZE // 2,
                       self.kernel, self.stride, padding='same', use_bias=False))
        assert self.model.output_shape == (
            None, HEIGHT // 4, WIDTH // 4, BATCH_SIZE // 2)
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.ReLU())
        self.model.add(layers.Conv2DTranspose(BATCH_SIZE // 4,
                       self.kernel, self.stride, padding='same', use_bias=False))
        assert self.model.output_shape == (
            None, HEIGHT // 2, WIDTH // 2, BATCH_SIZE // 4)
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.ReLU())
        self.model.add(layers.Conv2DTranspose(
            3, self.kernel, self.stride, padding='same', use_bias=False, activation='tanh'))
        assert self.model.output_shape == (None, HEIGHT, WIDTH, 3)
        return self.model

    def loss(self, fake_output):
        return cross_entropy(tf.ones_like(fake_output), fake_output)


class Discriminator(Model):
    def __init__(self, k=(5, 5), s=(2, 2)):
        super().__init__(k, s)

    def make_model(self):
        self.model.add(layers.Conv2D(BATCH_SIZE // 4, self.kernel,
                       strides=self.stride, padding='same', input_shape=[HEIGHT, WIDTH, 3]))
        self.model.add(layers.LeakyReLU())
        self.model.add(layers.Dropout(0.3))
        self.model.add(layers.Conv2D(BATCH_SIZE // 2, self.kernel,
                       strides=self.stride, padding='same'))
        self.model.add(layers.LeakyReLU())
        self.model.add(layers.Dropout(0.3))
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(1))
        return self.model

    def loss(self, real_output, fake_output):
        real_loss = cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss


def generate_and_save_images(model, epoch, test_input):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    predictions = model(test_input, training=False)
    
    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i, :, :, :] / 255) #  * 127.5 + 127.5
        plt.axis('off')
        # pdb.set_trace()

    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()


# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_step(images, generator: Generator, discriminator: Discriminator):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])
    gen_model = generator.get_model()
    disc_model = discriminator.get_model()
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = gen_model(noise, training=True)
        real_output = disc_model(images, training=True)
        fake_output = disc_model(generated_images, training=True)
        # pdb.set_trace()
        gen_loss = generator.loss(fake_output)
        disc_loss = discriminator.loss(real_output, fake_output)
    gradients_of_generator = gen_tape.gradient(
        gen_loss, gen_model.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(
        disc_loss, disc_model.trainable_variables)
    generator.get_optimizer().apply_gradients(
        zip(gradients_of_generator, gen_model.trainable_variables))
    discriminator.get_optimizer().apply_gradients(
        zip(gradients_of_discriminator, disc_model.trainable_variables))


def train(dataset, epochs, generator: Generator, discriminator: Discriminator):
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator.get_optimizer(
    ), discriminator_optimizer=discriminator.get_optimizer(), generator=generator.get_model(), discriminator=discriminator.get_model())
    for epoch in range(epochs):
        start = time.time()
        for image_batch in dataset:
            train_step(image_batch, generator, discriminator)
        # Produce images for the GIF as you go
        display.clear_output(wait=True)
        generate_and_save_images(generator.get_model(), epoch + 1, seed)
        # Save the model every 15 epochs
        if (epoch + 1) % 15 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)
        print('Time for epoch {} is {} sec'.format(
            epoch + 1, time.time()-start))
        # Generate after the final epoch
        display.clear_output(wait=True)
        generate_and_save_images(generator.get_model(), epochs, seed)


def main():
    # Load the images
    start = time.time()
    print("Loading training images...")
    train_images = []
    if LIGHT_MODE:
        for file in glob.glob(f'{DATA_ROOT}/train/*.jpg'):
            if (np.random.random() < TRAINING_DIST):
                train_images.append(
                    tf.keras.preprocessing.image.img_to_array(Image.open(file)))
        train_images = np.array(train_images)
    else:
        train_images = np.array([tf.keras.preprocessing.image.img_to_array(
            Image.open(file)) for file in glob.glob(f'{DATA_ROOT}/train/*.jpg')])
    print(f'Training images loaded in {time.time() - start} seconds.\n')
    start = time.time()
    print("Loading testing images...")
    test_images = np.array([tf.keras.preprocessing.image.img_to_array(
        Image.open(file)) for file in glob.glob(f'{DATA_ROOT}/test/*.jpg')])
    print(f'Testing images loaded in {time.time() - start} seconds.\n')

    # Normalize the images to [0, 1]
    # TODO: See if I need to do this
    # print("Normalizing images...")
    # train_images = train_images / 255
    # test_images = test_images / 255
    # print("Images normalized.")

    # Batch and shuffle the data
    # TODO: See if I need to do this for the test images too
    print("Batching and shuffling data...")
    train_dataset = tf.data.Dataset.from_tensor_slices(
        train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    test_dataset = tf.data.Dataset.from_tensor_slices(
        test_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    print("Data successfully batched and shuffled.")

    # Testing the generator
    generator = Generator()
    generator_model = generator.make_model()
    noise = tf.random.uniform([1, 100], 0, 1)
    generated_image = generator_model(noise, training=False)
    plt.imshow(generated_image[0, :, :, :])
    plt.show()
    pdb.set_trace()
    discriminator = Discriminator()
    discriminator_mdoel = discriminator.make_model()
    decision = discriminator_mdoel(generated_image)
    print(decision)

    train(train_dataset, EPOCHS, generator, discriminator)
    pdb.set_trace()


if __name__ == '__main__':
    main()
