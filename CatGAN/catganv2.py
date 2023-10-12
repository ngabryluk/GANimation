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
BATCH_SIZE = 1
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
        self.model_made = False
        self.optimizer = tf.keras.optimizers.Adam(1e-4)
        self.kernel = k
        self.stride = s

    def get_model(self):
        return self.model
    
    def check_model_status(self):
        return self.model_made

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
    def __init__(self, k=(4, 4), s=(2, 2)):
        super().__init__(k, s)

    # TODO: Probably extract most of these numbers to the class, so they can be specified upon creation
    # TODO: Work through this, figure out what everything does, and comment it
    def make_model(self):
        # Reshape in 1D space
        self.model.add(layers.Dense((HEIGHT // 64) * (WIDTH // 64)
                       * 32768, use_bias=False, input_shape=(100,)))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.ReLU())
        self.model.add(layers.Reshape((HEIGHT // 64, WIDTH // 64, 32768)))
        assert self.model.output_shape == (
            None, HEIGHT // 64, WIDTH // 64, 32768)
        # Reshape into 3D space
        self.model.add(layers.Conv2DTranspose(16384,
                       self.kernel, strides=(1, 1), padding='same', use_bias=False))
        assert self.model.output_shape == (
            None, HEIGHT // 32, WIDTH // 32, BATCH_SIZE // 16384)
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.ReLU())
        # Upsample
        self.model.add(layers.Conv2DTranspose(8192,
                       self.kernel, self.stride, padding='same', use_bias=False))
        assert self.model.output_shape == (
            None, HEIGHT // 16, WIDTH // 16, 8192)
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.ReLU())
        # Upsample
        self.model.add(layers.Conv2DTranspose(4096,
                       self.kernel, self.stride, padding='same', use_bias=False))
        assert self.model.output_shape == (
            None, HEIGHT // 8, WIDTH // 8, 4096)
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.ReLU())
        # Upsample
        self.model.add(layers.Conv2DTranspose(2048,
                       self.kernel, self.stride, padding='same', use_bias=False))
        assert self.model.output_shape == (
            None, HEIGHT // 4, WIDTH // 4, 2048)
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.ReLU())
        # Upsample
        self.model.add(layers.Conv2DTranspose(1024,
                       self.kernel, self.stride, padding='same', use_bias=False))
        assert self.model.output_shape == (
            None, HEIGHT // 2, WIDTH // 2, 1024)
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.ReLU())
        # Upsample
        self.model.add(layers.Conv2DTranspose(512,
                       self.kernel, self.stride, padding='same', use_bias=False))
        assert self.model.output_shape == (
            None, HEIGHT, WIDTH, 512)
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.ReLU())
        # Upsample
        self.model.add(layers.Conv2D(3, self.kernel, padding="same", activation="tanh"))
        assert self.model.output_shape == (None, HEIGHT, WIDTH, 3)
        self.model_made = True
        return self.model

    def loss(self, fake_output):
        return cross_entropy(tf.ones_like(fake_output), fake_output)


class Discriminator(Model):
    def __init__(self, k=(4, 4), s=(2, 2)):
        super().__init__(k, s)

    def make_model(self):
        # Downsample
        self.model.add(layers.Conv2D(16384, self.kernel,
                       strides=self.stride, padding='same', input_shape=[HEIGHT, WIDTH, 3]))
        self.model.add(layers.LeakyReLU())
        self.model.add(layers.BatchNormalization())
        # Downsample
        self.model.add(layers.Conv2D(8192, self.kernel,
                       strides=self.stride, padding='same'))
        self.model.add(layers.LeakyReLU())
        self.model.add(layers.BatchNormalization())
        # Downsample
        self.model.add(layers.Conv2D(4096, self.kernel,
                       strides=self.stride, padding='same'))
        self.model.add(layers.LeakyReLU())
        self.model.add(layers.BatchNormalization())
        # Downsample
        self.model.add(layers.Conv2D(2048, self.kernel,
                       strides=self.stride, padding='same'))
        self.model.add(layers.LeakyReLU())
        self.model.add(layers.BatchNormalization())
        # Downsample
        self.model.add(layers.Conv2D(1024, self.kernel,
                       strides=self.stride, padding='same'))
        self.model.add(layers.LeakyReLU())
        self.model.add(layers.BatchNormalization())
        # Downsample
        self.model.add(layers.Conv2D(512, self.kernel,
                       strides=self.stride, padding='same'))
        self.model.add(layers.LeakyReLU())
        # Flatten and dropout
        self.model.add(layers.Flatten())
        self.model.add(layers.Dropout(0.3))
        # Binary classification layer
        self.model.add(layers.Dense(1, activation="sigmoid"))
        self.model_made = True
        return self.model

    def loss(self, real_output, fake_output):
        real_loss = cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss


class GAN:
    def __init__(self, generator = Generator(), discriminator = Discriminator()):
        self.generator = generator
        self.generator_model = generator.make_model() if not generator.check_model_status() else generator.get_model()
        self.discriminator = discriminator
        self.discriminator = discriminator.make_model() if not discriminator.check_model_status() else discriminator.get_model()
    
    def get_generator(self):
        return self.generator
    
    def get_discriminator(self):
        return self.discriminator
    
    def get_generator_model(self):
        return self.generator.get_model()
    
    def get_discriminator_model(self):
        return self.discriminator.get_model()

    def generate_and_save_images(self, epoch, test_input):
        # Notice `training` is set to False.
        # This is so all layers run in inference mode (batchnorm).
        predictions = self.generator_model(test_input, training=False)
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
    def train_step(self, images):
        noise = tf.random.normal([BATCH_SIZE, noise_dim])
        gen_model = self.generator.get_model()
        disc_model = self.discriminator.get_model()
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = gen_model(noise, training=True)
            real_output = disc_model(images, training=True)
            fake_output = disc_model(generated_images, training=True)
            # pdb.set_trace()
            gen_loss = self.generator.loss(fake_output)
            disc_loss = self.discriminator.loss(real_output, fake_output)
        gradients_of_generator = gen_tape.gradient(
            gen_loss, gen_model.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(
            disc_loss, disc_model.trainable_variables)
        self.generator.get_optimizer().apply_gradients(
            zip(gradients_of_generator, gen_model.trainable_variables))
        self.discriminator.get_optimizer().apply_gradients(
            zip(gradients_of_discriminator, disc_model.trainable_variables))

    def train(self, dataset, epochs):
        checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator.get_optimizer(
        ), discriminator_optimizer=self.discriminator.get_optimizer(), generator=self.generator.get_model(), discriminator=self.discriminator.get_model())
        for epoch in range(epochs):
            start = time.time()
            for image_batch in dataset:
                self.train_step(image_batch)
            # Produce images for the GIF as you go
            display.clear_output(wait=True)
            self.generate_and_save_images(epoch + 1, seed)
            # Save the model every 15 epochs
            if (epoch + 1) % 15 == 0:
                checkpoint.save(file_prefix=checkpoint_prefix)
            print('Time for epoch {} is {} sec'.format(
                epoch + 1, time.time()-start))
            # Generate after the final epoch
            display.clear_output(wait=True)
            self.generate_and_save_images(epochs, seed)


def main():
    # Load the images
    start = time.time()
    print("Loading training images...")
    # train_images = []
    # if LIGHT_MODE:
    #     for file in glob.glob(f'{DATA_ROOT}/train/*.jpg'):
    #         if (np.random.random() < TRAINING_DIST):
    #             train_images.append(
    #                 tf.keras.preprocessing.image.img_to_array(Image.open(file)))
    #     train_images = np.array(train_images)
    # else:
    #     train_images = np.array([tf.keras.preprocessing.image.img_to_array(
    #         Image.open(file)) for file in glob.glob(f'{DATA_ROOT}/train/*.jpg')])
    train_images = tf.keras.utils.image_dataset_from_directory(f'{DATA_ROOT}/train', label_mode=None, image_size=(HEIGHT, WIDTH), batch_size=BATCH_SIZE, shuffle=True)
    print(f'Training images loaded in {time.time() - start} seconds.\n')
    start = time.time()
    print("Loading testing images...")
    # test_images = np.array([tf.keras.preprocessing.image.img_to_array(
    #     Image.open(file)) for file in glob.glob(f'{DATA_ROOT}/test/*.jpg')])
    test_images = tf.keras.utils.image_dataset_from_directory(f'{DATA_ROOT}/test', label_mode=None, image_size=(HEIGHT, WIDTH), batch_size=BATCH_SIZE, shuffle=True)
    print(f'Testing images loaded in {time.time() - start} seconds.\n')

    # Normalize the images to [0, 1]
    # TODO: See if I need to do this
    print("Normalizing images...")
    train_dataset = train_images.map(lambda x: (x - 127.5) / 127.5)
    test_dataset = test_images.map(lambda x: (x - 127.5) / 127.5)
    print("Images normalized.")

    # Batch and shuffle the data
    # TODO: See if I need to do this for the test images too
    # print("Batching and shuffling data...")
    # train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    # test_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    # print("Data successfully batched and shuffled.")

    # Testing the generator
    gan = GAN()
    noise = tf.random.uniform([1, 100], 0, 1)
    generated_image = gan.get_generator_model()(noise, training=False)
    plt.imshow(generated_image[0, :, :, :])
    plt.show()
    pdb.set_trace()
    decision = gan.get_discriminator_model()(generated_image)
    print(decision)

    gan.train(train_dataset, EPOCHS)
    pdb.set_trace()


if __name__ == '__main__':
    main()
