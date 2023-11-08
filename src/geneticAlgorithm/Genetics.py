import os
import sys

current_dir = os.path.dirname(__file__)
sys.path.insert( 0, current_dir)
from Genome import Genome


class Genetics:
  def __init__(self,options:dict):
    self.genomes = []
    self.counter = 0
    self.generations = 0
    self.mutation_rate = options.get("mutation_rate",0.05)
    self.clients = options["clients"]
    for client in self.clients :
      self.genomes.append(Genome(client, options["maxActionCount"], options["goal"]))
      
    self.maxActionCount = options["maxActionCount"]
    self.goal = options["goal"]
    
    
  def step(self):
    for genome in self.genomes:
      genome.step(self.counter)
    self.counter += 1
    if self.counter >= self.maxActionCount:
      self.end_generation()
      self.reset()
      
  def end_generation(self):
    # sort genomes by fitness
    self.genomes.sort(key=lambda g: g.fitness())
    for i in range(len(self.genomes)):
      print(f"{self.genomes[i].client.username}: {self.genomes[i].fitness()}")
    # kill bottom half
    self.genomes = self.genomes[:len(self.genomes)//2]
    # breed top half
    for i in range(len(self.genomes)):
      self.genomes.append(self.genomes[i] + self.genomes[i+1])
    # mutate
    for genome in self.genomes:
      genome.mutate(self.mutation_rate)
    # We may have duplicated some clients, we fix that here
    for g, c in zip(self.genomes, self.clients):
      g.client = c
      
    
    self.counter = 0
    self.generations += 1
    print(f"================= Generation {self.generations} =================")
      
  def reset(self):
    for genome in self.genomes:
      genome.reset()
      
  def __str__(self):
    arr = []
    for genome in self.genomes:
      arr.append(str(genome))
    return "\n".join(arr)
  