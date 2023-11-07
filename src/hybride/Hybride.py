from src.geneticAlgorithm.Genetics import Genetics


class Hybride(Genetics):

    def __init__(self, options: dict):
        super().__init__(options)
        self.genomes = []

        

    def step(self):
        for genome in self.genomes:
            genome.step(self.counter)
        self.counter += 1
        if self.counter >= self.maxActionCount:
            self.end_generation()
            self.reset()



