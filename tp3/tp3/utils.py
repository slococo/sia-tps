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
        return np.add((b - a) * (np.subtract(values, min) / (max - min)), a), min, max


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


relu_alfa = 0.25


def relu(x):
    res = []
    for i in x:
        if i >= 0:
            res.append(i)
        else:
            res.append(relu_alfa * i)
    return res


def relu_diff(x):
    res = []
    for i in x:
        if i >= 0:
            res.append(1)
        else:
            res.append(relu_alfa)
    return res


b = 0.05


def set_b(b_val):
    global b
    b = b_val


def get_b():
    return b


def tanh_diff(x):
    res = []
    x = np.atleast_1d(x)
    for i in x:
        res.append(b * (1 - ((math.tanh(b * i)) ** 2)))
    return res


def tanh_arr(x):
    res = []
    x = np.atleast_1d(x)
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
        res.append(2 * b * np.exp(-2 * b * i) / np.power((np.exp(-2 * b * i) + 1), 2))
    return res


def step(x):
    return np.sign(x)


def quadratic_error(exp, res):
    return np.average(np.power((np.subtract(exp, res) / 2), 2))


delta = 1e-6


def log_error(exp, res):
    return np.average(np.add(
        np.multiply(
            np.multiply(0.5, (1 + exp)), np.log((1 + exp + delta) / (1 + res + delta))
        ),
        np.multiply(
            np.multiply(0.5, (1 - exp)), np.log((1 - exp + delta) / (1 - res + delta))
        ),
    ))


def res_sign(res, _):
    return np.sign(res)
