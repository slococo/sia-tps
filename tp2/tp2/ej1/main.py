import json

import matplotlib.pyplot as plt
import numpy as np
import math
from tp2.perceptron import Perceptron


def main(config_path=None):
    if config_path is None:
        config_path = "ej1/config.json"
    with open(config_path) as f:
        data = json.load(f)["xor"]
        print(data)

        # matrix = np.random.rand(1, 3)
        matrix = np.zeros((1, 3))
        perceptron = Perceptron(matrix, None, tanh_arr, tanh_diff, len(matrix) + 1, 1)
        perceptron.train(data, 0, 10000, "batch")
        print(perceptron.matrix_arr)
        print(perceptron.predict([1, 1, 1]))
        print(perceptron.predict([1, -1, -1]))
        print(perceptron.predict([1, -1, 1]))
        print(perceptron.predict([1, 1, -1]))


def zeroes(x):
    res = []
    for i in x:
        res.append(0)
    return res


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


def tanh_diff(x):
    res = []
    for i in x:
        res.append(1 - math.tanh(i))
    return res


def tanh_arr(x):
    res = []
    for i in x:
        res.append(math.tanh(i))
    return res


if __name__ == "__main__":
    main("config.json")
