import random


class Genome:
  def __init__(self,client,maxActions:int,goal:dict):
    self.client = client
    self.actions = []
    self.goal = goal
    self.finished = -1
    self.maxActions = maxActions
    self.fitness_tracker = 0
    self.actions = []
    for _ in range(self.maxActions):
        self.actions.append(random.randint(0, self.client.actionCount-1))

    
  def has_completed(self):
    pos = self.client.get_position()
    return abs(pos["x"] - self.goal["x"]) < 0.5 and abs(pos["y"] - self.goal["y"]) < 0.5 and abs(pos["z"] - self.goal["z"]) < 0.5
  
  def distance_to_goal(self):
    pos = self.client.get_position()
    return (pos["x"] - self.goal["x"])**2 + (pos["y"] - self.goal["y"])**2 + (pos["z"] - self.goal["z"])**2
  
  def fitness(self):
    return self.fitness_tracker * 1/max(1,self.finished)
    
  ### Perform action ###
  def step(self, count:int):
    if self.finished > 0:
      return
    
    if self.has_completed():
      self.finished = count
      self.client.idle()
      self.print("Completed!")
      return
    self.fitness_tracker += self.distance_to_goal()
    self.client.do_action( self.actions[count])
    
  def reset(self):
    self.finished = -1
    self.fitness_tracker = 0
    self.client.reset()
    
  def mutate(self, chance:float):
    for i in range(len(self.actions)):
      if random.random() < chance:
        self.actions[i] = random.randint(0, self.client.actionCount-1)
    
  def print(self,string:str):
    print(f"{self.client.username}: {string}")
    
  def __str__(self):
    return f"{self.client.username}: {self.client.get_position()}"
  
  def __add__(self, other):
    newGenome = Genome(self.client, self.maxActions,self.goal)
    f1 = self.fitness()
    f2 = other.fitness()
    r = f1/(f1+f2)
    
    for i in range(len(self.actions)):
      if random.random() < r:
        newGenome.actions[i] = self.actions[i]
      else:
        newGenome.actions[i] = other.actions[i]
    return newGenome
  