from typing import List
from structure import Color
import numpy as np


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
    return np.array(solution)


