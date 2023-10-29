from typing import Tuple
import keras
from model import Model


class PrimaryGenerator(Model):
    # Don't know what type hinting to put for the optimizer
    def __init__(self, model: keras.Model | keras.Sequential = keras.Sequential(), optimizer=keras.optimizers.Adam(1e-4), k: Tuple[int, int] = (5, 5), s: Tuple[int, int] = (2, 2)):
        super().__init__(model, optimizer, k, s)

    def make_model(self):
        self.model_made = True

    def train(self):
        pass

    def predict(self):
        pass


if __name__ == "__main__":
    test = PrimaryGenerator()
