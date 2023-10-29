from typing import Tuple
import keras
from exceptions import *


class Model:
    # Don't know what type hinting to put for the optimizer
    def __init__(self, model: keras.Model | keras.Sequential = keras.Sequential(), optimizer=keras.optimizers.Adam(1e-4), k: Tuple[int, int] = (5, 5), s: Tuple[int, int] = (2, 2)):
        self.model = model
        self.optimizer = optimizer
        self.kernel = k
        self.stride = s

    def summary(self):
        try:
            print(self.model.summary(expand_nested=True, show_trainable=True))
        except ValueError:
            raise ModelException("print a summary for")

    def is_frozen(self) -> bool:
        try:
            assert self.check_model_status()
        except AssertionError:
            raise ModelException("check frozen status of")

        for layer in self.model.layers:
            if layer.trainable:
                return False
        return True

    def is_unfrozen(self) -> bool:
        try:
            assert self.check_model_status()
        except AssertionError:
            raise ModelException("check frozen status of")

        for layer in self.model.layers:
            if not layer.trainable:
                return False
        return True

    def freeze(self):
        try:
            assert self.check_model_status()
        except AssertionError:
            raise ModelException("freeze")

        try:
            assert self.is_unfrozen()
        except AssertionError:
            raise FreezeException('frozen', 'freeze')

        for layer in self.model.layers:
            layer.trainable = False

    def unfreeze(self):
        try:
            assert self.check_model_status()
        except AssertionError:
            raise ModelException("unfreeze")

        try:
            assert self.is_frozen()
        except AssertionError:
            raise FreezeException('unfrozen', 'unfreeze')

        for layer in self.model.layers:
            layer.trainable = True

    def get_model(self):
        return self.model

    def check_model_status(self) -> bool:
        return self.model.built

    # Don't know what type hinting to put for the optimizer
    def get_optimizer(self):
        return self.optimizer

    def get_kernel(self) -> Tuple[int, int]:
        return self.kernel

    def get_strides(self) -> Tuple[int, int]:
        return self.stride

    # Don't know what type hinting to put for the optimizer
    def set_optimizer(self, new_optimizer):
        self.optimizer = new_optimizer

    def set_kernel(self, new_kernel: Tuple[int, int]):
        self.kernel = new_kernel

    def set_strides(self, new_strides: Tuple[int, int]):
        self.stride = new_strides


if __name__ == "__main__":
    test = Model()
    test.is_frozen()
