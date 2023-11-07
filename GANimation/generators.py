import sys
from os import path

sys.path.append(path.abspath(path.dirname(__file__)))

from typing import Tuple
import keras
from keras import layers
from model import Model
from math import floor
import pdb


class PrimaryGenerator(Model):
    # Don't know what type hinting to put for the optimizer
    def __init__(
        self,
        model: keras.Model | keras.Sequential = keras.Sequential(),
        optimizer=keras.optimizers.Adam(1e-4),
        k: Tuple[int, int] = (5, 5),
        s: Tuple[int, int] = (2, 2),
    ):
        super().__init__(model, optimizer, k, s)

    def make_model(
        self, input_size: Tuple[int, int, int] = (28, 28, 1), num_upsamples: int = 2
    ):
        # ((n - k) / s) + 1
        height, width, depth = input_size
        scale = pow(2, num_upsamples)
        self.model.add(
            layers.Dense(
                int(height / scale)
                * int(width / scale)
                * (64 * scale),  # last one won't really work for generalization
                use_bias=False,
                input_shape=(100,),
            )
        )
        print(self.model.output_shape)
        # self.model.add(
        #     layers.Dense(
        #         int(height / scale)
        #         * int(width / scale)
        #         * (64 * scale / 2),  # last one won't really work for generalization
        #         use_bias=False,
        #         input_shape=(100,),
        #     )
        # )
        # print(self.model.output_shape)
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.LeakyReLU())

        self.model.add(
            layers.Reshape((int(height / scale), int(width / scale), (64 * scale)))
        )
        print(self.model.output_shape)
        # Note: None is the batch size
        # 7, 7, 256
        assert self.model.output_shape == (
            None,
            int(height / scale),
            int(width / scale),
            (64 * scale),
        )

        self.model.add(
            layers.Conv2DTranspose(
                64 * scale / 2,
                self.kernel,
                strides=(1, 1),
                padding="same",
                use_bias=False,
            )
        )
        print(self.model.output_shape)
        # 7, 7, 128
        assert self.model.output_shape == (
            None,
            int(height / scale),
            int(width / scale),
            (64 * int(scale / 2)),
        )
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.LeakyReLU())

        for num in range(num_upsamples - 1, 0, -1):
            print(self.model.output_shape)
            self.model.add(
                layers.Conv2DTranspose(
                    64 * pow(2, num - 1),
                    self.kernel,
                    strides=self.stride,
                    padding="same",
                    use_bias=False,
                )
            )
            print(self.model.output_shape)
            # 14, 14, 64
            assert self.model.output_shape == (
                None,
                int(height / pow(2, num)),
                int(width / pow(2, num)),
                64 * pow(2, num - 1),
            )
            self.model.add(layers.BatchNormalization())
            self.model.add(layers.LeakyReLU())

        self.model.add(
            layers.Conv2DTranspose(
                depth,
                self.kernel,
                strides=self.stride,
                padding="same",
                use_bias=False,
                activation="tanh",
            )
        )
        # 28, 28, 1
        assert self.model.output_shape == (None, height, width, depth)
        self.model_made = True

    def train(self):
        pass

    def predict(self):
        pass


class RefinementGenerator(Model):
    # Don't know what type hinting to put for the optimizer
    def __init__(
        self,
        model: keras.Model | keras.Sequential = keras.Sequential(),
        optimizer=keras.optimizers.Adam(1e-4),
        k: Tuple[int, int] = (5, 5),
        s: Tuple[int, int] = (2, 2),
    ):
        super().__init__(model, optimizer, k, s)

    def make_model(self):
        self.model_made = True

    def train(self):
        pass

    def predict(self):
        pass


if __name__ == "__main__":
    test = PrimaryGenerator()
    print("Test model assigned")
    test.make_model((64, 64, 1))
    print("Test model made")
    test.summary()
