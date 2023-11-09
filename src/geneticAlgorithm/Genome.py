import random


class Genome:
    prev_positions = list()

    def __init__(self, client, maxActions: int, goal: dict):
        self.client = client
        self.actions = []
        self.goal = goal
        self.finished = -1
        self.maxActions = maxActions
        self.fitness_tracker = 0
        self.actions = []
        for _ in range(self.maxActions):
            self.actions.append(random.randint(0, self.client.actionCount - 1))

    def has_completed(self):
        return self.client.has_reached_goal(self.goal)
        #pos = self.client.has_reached_goal()
        #return abs(pos["x"] - self.goal["x"]) < 0.5 and abs(pos["y"] - self.goal["y"]) < 0.5 and abs(
        #    pos["z"] - self.goal["z"]) < 0.5

    def distance_to_goal(self):
        pos = self.client.get_position()
        return (pos["x"] - self.goal["x"]) ** 2 + (pos["y"] - self.goal["y"]) ** 2 + (pos["z"] - self.goal["z"]) ** 2

    def fitness(self):
        return self.fitness_tracker / max(1, self.finished)

    ### Perform action ###
    def step(self, count: int):
        if self.finished > 0:
            return

        if self.has_completed():
            self.finished = count - 100
            self.client.idle()
            #self.print("Completed!")
            return
        self.update_fitness()
        self.client.do_action(self.actions[count])

    def update_fitness(self):
        # Idea: fitness based on:
        # - distance to goal
        # - actions taken (if goal is reached)
        # - movements in the past 5 moves
        # Append current position to prev_positions
        self.prev_positions.append(self.client.get_position())
        # Remove oldest position if prev_positions is too long
        l = len(self.prev_positions)
        if l > 5:
            self.prev_positions.pop(0)
            l -= 1
        # calculate the difference between the current position and the oldest position
        diff = max(self.prev_positions[l - 1].xzDistanceTo(self.prev_positions[0]), 0.1) if l > 0 else 1
        self.fitness_tracker += self.distance_to_goal() #/ diff

    def reset(self):
        self.finished = -1
        self.fitness_tracker = 0
        self.client.reset()

    def mutate(self, chance: float):
        for i in range(len(self.actions)):
            if random.random() < chance:
                self.actions[i] = random.randint(0, self.client.actionCount - 1)

    def print(self, string: str):
        print(f"{self.client.username}: {string}")

    def __str__(self):
        return f"{self.client.username}: {self.client.get_position()}"

    def __add__(self, other):
        newGenome = Genome(self.client, self.maxActions, self.goal)
        f1 = self.fitness()
        f2 = other.fitness()
        r = f1 / (f1 + f2)

        for i in range(len(self.actions)):
            if random.random() < r:
                newGenome.actions[i] = self.actions[i]
            else:
                newGenome.actions[i] = other.actions[i]
        return newGenome
