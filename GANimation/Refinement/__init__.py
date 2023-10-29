from .discriminator import RefinementDiscriminator
from .generator import RefinementGenerator


class RefinementGAN:
    def __init__(self, gen: RefinementGenerator = RefinementGenerator(), disc: RefinementDiscriminator = RefinementDiscriminator()):
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


if __name__ == "__main__":
    test = RefinementGAN()
