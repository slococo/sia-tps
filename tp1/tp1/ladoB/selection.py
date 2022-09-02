import math
import random
from typing import List
from structure import Color
import numpy as np


def elite_selection(pop: List[Color], num_to_select, target):
    for i in pop:
        i.calculate_fitness(target)
    sorted_pop = np.sort(pop)
    pop_len = len(sorted_pop)
    solution = []
    for i in range(0, pop_len):
        times = int(np.ceil((num_to_select - i) / pop_len))
        for j in range(0, times):
            solution = [*solution, sorted_pop[i]]
    return solution


def roulette_selection(pop: List[Color], num_to_select, target):
    fitness = 0
    for i in pop:
        fitness += i.calculate_fitness(target)

    return roulette_aux(pop, num_to_select, fitness, target)


def roulette_aux(pop: List[Color], k, accum_fit, target):
    rs = []
    for i in range(0, k):
        rs.append(random.uniform(0, 1))

    res = []
    sorted_pop = np.sort(pop)

    for r in rs:
        fit = 0
        for x in sorted_pop:
            fit += x.calculate_fitness(target) / accum_fit
            if r <= fit:
                res.append(x)
                break

    return res


def boltzmann_selection(pop: List[Color], num_to_select, target):
    # t = 20
    t = 2 + (20 - 4) * math.exp(-1)

    avg = 0
    for i in pop:
        avg += math.exp(i.calculate_fitness(target) / t)

    avg /= len(pop)
    tot = 0
    for i in pop:
        i.fitness = math.exp(i.calculate_fitness(target) / t) / avg
        tot += i.fitness

    return roulette_aux(pop, num_to_select, tot, target)
