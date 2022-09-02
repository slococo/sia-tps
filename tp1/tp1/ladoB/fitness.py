import numpy as np


def euclidean_distance_fitness(target, actual):
    return np.linalg.norm(target - actual)

