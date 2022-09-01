import fitness
import numpy as np


def elite_selection(pop, num_to_select):
    pop_size = len(pop)
    sorted_pop = np.sort(pop)

