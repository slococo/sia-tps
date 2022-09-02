import random

import numpy as np

from tp1.ladoB.structure import Color

import copy

def uniform_mutate(color: Color):
    for i in range(0, 24):
        mut = random.uniform(0, 1)
        if mut < 0.1:
            color.genes[i] = 1 - color.genes[i]


def uniform_cross(par1: Color, par2: Color):
    child1 = copy.copy(par1)
    child2 = copy.copy(par2)
    for i in range(0, 24):
        prob = round(random.uniform(0, 1))
        if prob == 1:
            aux = child1.genes[i]
            child1.genes[i] = child2.genes[i]
            child2.genes[i] = aux

    uniform_mutate(child1)
    uniform_mutate(child2)

    child1.rgb = np.zeros(3)
    child2.rgb = np.zeros(3)
    for i in range(0, 8):
        child1.rgb[0] = np.left_shift(np.uint8(child1.rgb[0]), 1)
        child1.rgb[0] += child1.genes[i]

        child1.rgb[1] = np.left_shift(np.uint8(child1.rgb[1]), 1)
        child1.rgb[1] += child1.genes[i + 8]

        child1.rgb[2] = np.left_shift(np.uint8(child1.rgb[2]), 1)
        child1.rgb[2] += child1.genes[i + 16]

        child2.rgb[0] = np.left_shift(np.uint8(child2.rgb[0]), 1)
        child2.rgb[0] += child2.genes[i]

        child2.rgb[1] = np.left_shift(np.uint8(child2.rgb[1]), 1)
        child2.rgb[1] += child2.genes[i + 8]

        child2.rgb[2] = np.left_shift(np.uint8(child2.rgb[2]), 1)
        child2.rgb[2] += child2.genes[i + 16]

    return child1, child2

