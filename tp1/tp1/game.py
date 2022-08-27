import numpy as np


class Game:
    def __init__(self, size: int, num_colors: int):
        if size < 0 or num_colors < 0:
            raise ValueError("Game must have positive size and amount of colors")
        self.size = size
        self.num_colors = num_colors
        self.matrix = np.random.randint(1, self.num_colors + 1, size=(self.size, self.size))

    def shuffle(self):
        self.matrix = np.random.randint(1, self.num_colors + 1, size=(self.size, self.size))


print(Game(3, 3).matrix)
