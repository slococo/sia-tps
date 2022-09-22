import math
import random
from typing import List

import numpy as np
from tp1.ladoB.structure import Color


def elite_selection(pop: List[Color], num_to_select, target):
    for i in pop:
        i.calculate_fitness(target)
    sorted_pop = np.sort(pop)[::-1]
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


it = 0


def boltzmann_selection(pop: List[Color], num_to_select, target):
    global it
    it += 1
    t = 20 + (20 - 4) * math.exp(-it)

    avg = 0
    for i in pop:
        avg += math.exp(i.calculate_fitness(target) / t)

    avg /= len(pop)
    tot = 0
    for i in pop:
        i.fitness = math.exp(i.calculate_fitness(target) / t) / avg
        tot += i.fitness

    return roulette_aux(pop, num_to_select, tot, target)


def tournament_selection_det(pop: List[Color], num_to_select, target):
    pool = pop.copy()
    sol = []

    for col in pop:
        col.calculate_fitness(target)

    for i in range(0, num_to_select):
        random.shuffle(pool)
        group = pool[0 : round(len(pool) / 3)]
        if not group:
            break
        group.sort(reverse=True)
        aux = group.pop()
        sol.append(aux)
        pool.remove(aux)

    return sol


def tournament_selection_prob(pop: List[Color], num_to_select, target):
    threshold = 0.8
    pool = pop.copy()
    sol = []

    for col in pop:
        col.calculate_fitness(target)

    for i in range(0, num_to_select):
        random.shuffle(pool)
        group = pool[0:2]
        if not group:
            break
        group.sort(reverse=True)
        if random.uniform(0, 1) < threshold:
            aux = group.pop()
        else:
            aux = group[-1]
        sol.append(aux)
        pool.remove(aux)

    return sol
