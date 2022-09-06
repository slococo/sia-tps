import time
from math import ceil, floor

import numpy as np

from tp1.ladoB.fitness import euclidean_distance_fitness


class Color:
    def __init__(self, rgb):
        self.genes = np.zeros(24)
        self.rgb = np.array(rgb)
        self.fitness = None
        self.genes = np.array(
            np.concatenate(
                [[np.uint8(i) for i in "{0:08b}".format(num)] for num in self.rgb]
            )
        )

    def is_goal(self, target):
        for i in range(0, 3):
            if self.rgb[i] < target.rgb[i] - 5 or self.rgb[i] > target.rgb[i] + 5:
                return False
        return True

    def calculate_fitness(self, target):
        if self.fitness is None:
            self.fitness = euclidean_distance_fitness(target.rgb, self.rgb)
        return self.fitness

    def calc_rgb(self):
        self.rgb = np.zeros(3)
        for i in range(0, 8):
            self.rgb[0] = np.left_shift(np.uint8(self.rgb[0]), 1)
            self.rgb[0] += self.genes[i]
            self.rgb[1] = np.left_shift(np.uint8(self.rgb[1]), 1)
            self.rgb[1] += self.genes[i + 8]
            self.rgb[2] = np.left_shift(np.uint8(self.rgb[2]), 1)
            self.rgb[2] += self.genes[i + 16]

    def __str__(self):
        return self.rgb.__str__()

    def __lt__(self, other):
        return self.fitness - other.fitness


class GeneticExecutor:
    def __init__(
        self,
        colors,
        target,
        cross_method,
        selection_method,
        mutation_method,
        end_method=None,
        max_gen=1000,
        gen_size=50
    ):
        self.target = target
        self.gen_n = 0
        self.generation = colors
        self.cross_method = cross_method
        self.selection_method = selection_method
        self.end_method = end_method
        self.new_gen = None
        self.max_gen = max_gen
        self.found_sol = False
        self.mutation_method = mutation_method
        self.gen_size = gen_size

    def start(self):
        start_time = time.time()
        while not self.end():
            self.select()
            self.generate()
        print("{:.2f}".format((time.time() - start_time) * 1000) + "ms")
        return self.gen_n

    def generate(self):
        self.new_gen = []
        if len(self.generation) <= 1:
            return
        for i in range(0, floor(len(self.generation) / 2)):
            child1, child2 = self.cross_method(
                self.generation[i], self.generation[-(i + 1)], self.mutation_method
            )

            self.new_gen.append(child1)
            self.new_gen.append(child2)
        if len(self.generation) % 2 == 1:
            child1, child2 = self.cross_method(
                self.generation[ceil(len(self.generation) / 2)],
                self.generation[floor(np.random.uniform(0, len(self.generation)))],
                self.mutation_method,
            )
            self.new_gen.append(child1)
            self.new_gen.append(child2)

    def select(self):
        if self.new_gen:
            for new in self.new_gen:
                self.generation.append(new)

        self.generation = self.selection_method(
            # self.generation, floor(len(self.generation) / 2), self.target
            self.generation, self.gen_size, self.target
        )
        self.gen_n += 1

    def end(self):
        if self.end_method:
            return self.end_method(self.generation, self.gen_n, self.target)
        return self.default_stop()

    def success(self):
        print(self.gen_n)
        for color in self.generation:
            print(color)
        return True

    def default_stop(self):
        for color in self.generation:
            if color.is_goal(self.target):
                # print("Goal reached")
                # print("Generation: " + self.gen_n.__str__())
                # print("Color: " + color.__str__())
                self.found_sol = True
                return True
        if self.gen_n >= self.max_gen:
            return True
        return False
