from .discriminator import PrimaryDiscriminator
from .generator import PrimaryGenerator


class PrimaryGAN:
    def __init__(self, gen: PrimaryGenerator = PrimaryGenerator(), disc: PrimaryDiscriminator = PrimaryDiscriminator()):
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


if __name__ == "__main__":
    test = PrimaryGAN()
