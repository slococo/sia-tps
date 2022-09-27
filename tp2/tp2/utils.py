import math

import numpy as np


def denormalise(values, a, b, min, max):
    return (np.subtract(values, a) / (b - a)) * (max - min) + min


def normalise(values, a, b):
    min = np.min(values)
    max = np.max(values)
    if min == max:
        raise "Values are not normalisable"
    else:
        return np.add((b - a) * (np.subtract(values, min) / (max - min)), a)


def identity(x):
    res = []
    for i in x:
        res.append(i)
    return res


def ident_diff(x):
    res = []
    for _ in x:
        res.append(1)
    return res


b = 1


def tanh_diff(x):
    res = []
    for i in x:
        res.append(b * (1 - math.tanh(b * i)))
    return res


def tanh_arr(x):
    res = []
    for i in x:
        res.append(math.tanh(b * i))
    return res


def logistic_arr(x):
    res = []
    for i in x:
        res.append(1 / (1 + math.exp(-2 * b * i)))
    return res


def logistic_diff(x):
    res = []
    for i in x:
        res.append(
            2 * b / (1 + math.exp(-2 * b * i)) * (1 - 1 / (1 + math.exp(-2 * b * i)))
        )
    return res
