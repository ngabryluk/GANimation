import sys
from os import path

sys.path.append(path.abspath(path.dirname(__file__)))

from discriminators import *
from generators import *


class RefinementGAN:
    def __init__(
        self,
        gen: RefinementGenerator = RefinementGenerator(),
        disc: RefinementDiscriminator = RefinementDiscriminator(),
    ):
        self.generator = gen
        self.discriminator = disc

    def train(self):
        while True:
            if self.discriminator.is_frozen():
                self.discriminator.unfreeze()
            self.discriminator.train()
            self.discriminator.freeze()
            if self.generator.is_frozen():
                self.generator.unfreeze()
            self.generator.train()
            self.generator.freeze()
            break

    def predict(self):
        return self.generator.predict()


class PrimaryGAN:
    def __init__(
        self,
        gen: PrimaryGenerator = PrimaryGenerator(),
        disc: PrimaryDiscriminator = PrimaryDiscriminator(),
    ):
        self.generator = gen
        self.discriminator = disc

    def train(self):
        # while True:
        #     if self.discriminator.is_frozen():
        #         self.discriminator.unfreeze()
        #     self.discriminator.train()
        #     self.discriminator.freeze()
        #     if self.generator.is_frozen():
        #         self.generator.unfreeze()
        #     self.generator.train()
        #     self.generator.freeze()
        #     break
        pass

    def predict(self):
        return self.generator.predict()


class GANimation:
    def __init__(
        self,
        primary_gan: PrimaryGAN = PrimaryGAN(),
        refinement_gan: RefinementGAN = RefinementGAN(),
    ):
        self.primary = primary_gan
        self.refinement = refinement_gan

    def train(self):
        self.primary.train()
        self.refinement.train()

    def predict(self):
        pass
