import json

import matplotlib

matplotlib.use("TkAgg")
import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

from tp2.perceptron import Perceptron


def main(config_path=None, data_path=None):
    if config_path is None:
        config_path = "tp2/ej1/config.json"
    if data_path is None:
        data_path = "tp2/ej1/data.json"

    with open(config_path) as f:
        data = json.load(f)
        dataset = data["dataset"]
        learning = data["learning"]
        max_iter = data["max_iter"]
        error = data["error"]
        eta = data["eta"]

    with open(data_path) as f:
        data = json.load(f)[dataset]
        # matrix = np.random.rand(1, 3)
        matrix = np.zeros((1, 3))
        perceptron = Perceptron(matrix, None, tanh_arr, tanh_diff, len(matrix) + 1, eta)
        perceptron.train(data, error, max_iter, learning)
        print(perceptron.matrix_arr)
        print("x: 1 ~ y: 1")
        print(perceptron.predict([1, 1, 1]))
        print("x: -1 ~ y: -1")
        print(perceptron.predict([1, -1, -1]))
        print("x: -1 ~ y: 1")
        print(perceptron.predict([1, -1, 1]))
        print("x: 1 ~ y: -1")
        print(perceptron.predict([1, 1, -1]))

        x = np.outer(np.linspace(-3, 3, 32), np.ones(32))
        y = x.copy().T
        coefs = perceptron.matrix_arr[0]
        z = np.tanh(coefs[0] + x * coefs[1] + y * coefs[2])
        fig = plt.figure(figsize=(14, 9))
        ax = plt.axes(projection="3d")
        ax.plot_surface(x, y, z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        X = np.array([1, 1, -1, -1])
        Y = np.array([1, -1, -1, 1])
        ax.scatter(X, Y, color="black")
        plt.show()

        # perceptron.save()
        # del perceptron
        # perceptron = Perceptron.load()
        # print(perceptron.matrix_arr)


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


if __name__ == "__main__":
    main("config.json", "data.json")
