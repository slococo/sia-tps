from math import floor, ceil

import numpy as np
import fitness


class Color:
    def __init__(self, rgb):
        self.genes = np.zeros(24)
        self.rgb = rgb
        self.fitness = None
        for i in range(0, 8):
            self.genes[i] = (rgb[0] >> (7 - i) & 1)
            self.genes[i + 8] = (rgb[1] >> (7 - i) & 1)
            self.genes[i + 16] = (rgb[2] >> (7 - i) & 1)

    def is_goal(self, target):
        for i in range(0, 3):
            if self.rgb[i] < target.rgb[i] - 10 or self.rgb[i] > target.rgb[i] + 10:
                return False
        return True

    def calculate_fitness(self, target):
        if self.fitness is None:
            self.fitness = fitness.euclidean_distance_fitness(target.rgb, self.rgb)
        return self.fitness

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
        while not self.end():
            self.select()
            self.generate()
        return self.success()

    def generate(self):
        if not self.new_gen:
            self.new_gen = []
        for i in range(0, floor(len(self.generation) / 2)):
            child1, child2 = self.cross_method(self.generation[i], self.generation[-(i + 1)])
            self.new_gen.append(child1)
            self.new_gen.append(child2)
        if len(self.generation) % 2 == 1:
            child1, child2 = self.cross_method(self.generation[ceil(len(self.generation) / 2)],
                                               self.generation[floor(np.random.uniform(0, len(self.generation)))])
            self.new_gen.append(child1)
            self.new_gen.append(child2)
            self.generation = self.new_gen

    def select(self):
        self.generation = self.selection_method(self.generation, len(self.generation), self.target)
        self.gen_n += 1

    def end(self):
        if self.end_method(self.generation, self.gen_n, self.target):
            return True
        return False

    def success(self):
        print(self.gen_n)
        for color in self.generation:
            print(color)
        return True
