import math

import numpy as np

max_dist = 442


# def euclidean_distance_fitness(target, actual):
#     return max_dist - np.linalg.norm(target - actual)


def euclidean_distance_fitness(target, actual):
    return 30 / (np.linalg.norm(target - actual) + 1)
