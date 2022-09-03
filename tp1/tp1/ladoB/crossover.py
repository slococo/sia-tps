import copy
import random

from tp1.ladoB.mutation import uniform_mutate
from tp1.ladoB.structure import Color


def uniform_cross(par1: Color, par2: Color):
    child1 = copy.deepcopy(par1)
    child2 = copy.deepcopy(par2)
    for i in range(0, 24):
        prob = round(random.uniform(0, 1))
        if prob == 1:
            aux = child1.genes[i]
            child1.genes[i] = child2.genes[i]
            child2.genes[i] = aux

    uniform_mutate(child1)
    uniform_mutate(child2)

    return child1, child2


def two_point_cross(par1: Color, par2: Color):
    p1 = random.uniform(0, 24)
    p2 = random.uniform(0, 24)
    aux = max(p2, p1)
    p1 = min(p1, p2)
    p2 = round(aux)
    i = round(p1)

    child1 = copy.deepcopy(par1)
    child2 = copy.deepcopy(par2)

    while i < p2:
        aux = child1.genes[i]
        child1.genes[i] = child2.genes[i]
        child2.genes[i] = aux
        i += 1

    return child1, child2


def avg_cross(par1: Color, par2: Color):
    rgb1 = par1.rgb
    rgb2 = par2.rgb
    for i in range(0, 3):
        prob = random.uniform(0, 1)
        rgb1[i] = prob * par1.rgb[i] + (1 - prob) * par2.rgb[i]
        prob = random.uniform(0, 1)
        rgb2[i] = (1 - prob) * par1.rgb[i] + prob * par2.rgb[i]

    for i in range(0, 3):
        prob = random.uniform(0, 1)
        mov = random.normalvariate(0, 1)
        if prob < 0.00:
            if rgb1[i] + mov > 0:
                rgb1[i] += mov
        prob = random.uniform(0, 1)
        mov = random.normalvariate(0, 1)
        if prob < 0.00:
            if rgb2[i] + mov > 0:
                rgb2[i] += mov

    return Color(rgb1), Color(rgb2)
