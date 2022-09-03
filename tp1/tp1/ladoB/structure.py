import time
from math import ceil, floor

import fitness
import numpy as np


class Color:
    def __init__(self, rgb):
        self.genes = np.zeros(24)
        self.rgb = np.array(rgb)
        self.fitness = None
        # for i in range(0, 8):
        #     self.genes[i] = (rgb[0] >> (7 - i)) & 1
        #     self.genes[i + 8] = (rgb[1] >> (7 - i)) & 1
        #     self.genes[i + 16] = (rgb[2] >> (7 - i)) & 1
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
            self.fitness = fitness.euclidean_distance_fitness(target.rgb, self.rgb)
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
    def __init__(self, colors, target, cross_method, selection_method, end_method):
        self.target = target
        self.gen_n = 0
        self.generation = colors
        self.cross_method = cross_method
        self.selection_method = selection_method
        self.end_method = end_method
        self.new_gen = None

    def start(self):
        start_time = time.time()
        while not self.end():
            self.select()
            self.generate()
        # print(self.gen_n)
        print("{:.2f}".format((time.time() - start_time) * 1000) + "ms")
        return self.gen_n
        # return self.success()

    def generate(self):
        self.new_gen = []
        if len(self.generation) <= 1:
            return
        for i in range(0, floor(len(self.generation) / 2)):
            child1, child2 = self.cross_method(
                self.generation[i], self.generation[-(i + 1)]
            )

            self.new_gen.append(child1)
            self.new_gen.append(child2)
        if len(self.generation) % 2 == 1:
            child1, child2 = self.cross_method(
                self.generation[ceil(len(self.generation) / 2)],
                self.generation[floor(np.random.uniform(0, len(self.generation)))],
            )
            self.new_gen.append(child1)
            self.new_gen.append(child2)

    def select(self):
        if self.new_gen:
            for new in self.new_gen:
                self.generation.append(new)

        self.generation = self.selection_method(
            self.generation, floor(len(self.generation) / 2), self.target
        )
        self.gen_n += 1

    def end(self):
        return self.end_method(self.generation, self.gen_n, self.target)

    def success(self):
        print(self.gen_n)
        for color in self.generation:
            print(color)
        return True
