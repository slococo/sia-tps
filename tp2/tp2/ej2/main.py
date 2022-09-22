import json

import matplotlib

matplotlib.use("TkAgg")
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm

from tp2.perceptron import Perceptron


def main(config_path=None):
    if config_path is None:
        config_path = "tp2/ej2/config.json"

    path = "dataset.csv"
    data_column_names = ["x1", "x2", "x3", "y"]

    df = pd.read_csv(path)

    data_matrix = df[data_column_names].to_numpy()
    data_matrix = np.insert(data_matrix, 0, 1, axis=1)

    with open(config_path) as f:
        data = json.load(f)
        learning = data["learning"]
        max_iter = data["max_iter"]
        error = data["error"]
        eta = data["eta"]

    # matrix = np.random.rand(1, 4)
    matrix = np.zeros((1, 4))
    perceptron = Perceptron(matrix, None, tanh_arr, tanh_diff, len(matrix) + 1, eta)
    training_data = data_matrix[: round(len(data_matrix) / 3)]
    print(data_matrix)

    perceptron.train(training_data / 89, error, max_iter, learning)

    for data in data_matrix:
        print("expected: ", data[-1], "\tout: ", perceptron.predict(data[:-1])[0])


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
    main("config.json")
