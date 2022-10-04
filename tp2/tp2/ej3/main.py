import json

import numpy as np
import pandas as pd

from tp2 import utils
from tp2.ej3 import graph
from tp2.perceptron import Perceptron


def res_index(x):
    res = np.full_like(x, fill_value=-1)
    # res = np.zeros_like(x)
    res[np.argmax(x)] = 1
    return res


def main(config_path=None):
    if config_path is None:
        config_path = "tp2/ej3/config.json"

    path = "digitsformat.csv"

    df = pd.read_csv(path, sep=' ')

    data_column_names = ["x" + str(i) for i in range(1, 36)]
    expected_column_names = ["y"]

    df = pd.read_csv(path)

    data_matrix = df[data_column_names].to_numpy()
    res_matrix = df[expected_column_names].to_numpy()
    data_matrix = np.insert(data_matrix, 0, 1, axis=1)

    try:
        with open(config_path) as f:
            data = json.load(f)
            learning = data["learning"]
            max_iter = data["max_iter"]
            error = data["error"]
            eta = data["eta"]
    except FileNotFoundError:
        print("Couldn't find config path")
        exit(1)

    # matrix = np.zeros((21, 36))
    matrix = np.random.rand(28, 36)
    matrix1 = np.random.rand(24, 28)
    # matrix2 = np.zeros((14, 21))
    matrix2 = np.random.rand(18, 24)
    matrix3 = np.random.rand(14, 18)
    matrix4 = np.random.rand(12, 14)
    # matrix3 = np.zeros((10, 14))
    matrix5 = np.random.rand(10, 12)

    matr = [matrix, matrix1, matrix2, matrix3, matrix4, matrix5]
    perceptron = Perceptron(
        matr, None, utils.tanh_arr, utils.tanh_diff, len(matrix) + 1, eta
    )

    data_min = np.min(data_matrix)
    data_normalised = np.subtract(
        2 * (np.subtract(data_matrix, data_min) / (np.max(data_matrix) - data_min)), 1
    )

    data_normalised = np.concatenate((data_normalised, res_matrix), axis=1)

    training_data = data_normalised

    perceptron.train(training_data, error, max_iter, learning, res_index)

    for data in data_normalised:
        print(
            "expected: ",
            data[-1],
            "\tout: ",
            np.argmax(perceptron.predict(data[:-1]))
        )

    # graph.plot(df)

def main_xor(config_path=None, data_path=None):
    if config_path is None:
        config_path = "tp2/ej3/config.json"
    if data_path is None:
        data_path = "tp2/ej3/data.json"

    with open(config_path) as f:
        data = json.load(f)
        learning = data["learning"]
        max_iter = data["max_iter"]
        error = data["error"]
        eta = data["eta"]

    with open(data_path) as f:
        data = json.load(f)["xor"]
        matrix1 = np.random.rand(6, 3)
        matrix3 = np.random.rand(3, 6)
        matrix2 = np.atleast_2d(np.random.rand(1, 3))
        perceptron = Perceptron(
            [matrix1, matrix3, matrix2], None, utils.tanh_arr, utils.tanh_diff, -1, eta, input_keep_prob=0.8, hidden_keep_prob=0.5
        )

        perceptron.train(data, error, max_iter, learning)
        # print(perceptron.matrix_arr)

        print("x1: 1 ~ x2: 1 ~ exp = -1 ~ res = ", perceptron.predict([1, 1, 1]))
        print("x1: -1 ~ x2: -1 ~ exp = -1 ~ res = ", perceptron.predict([1, -1, -1]))
        print("x1: -1 ~ x2: 1 ~ exp = 1 ~ res = ", perceptron.predict([1, -1, 1]))
        print("x1: 1 ~ x2: -1 ~ exp = 1 ~ res = ", perceptron.predict([1, 1, -1]))


if __name__ == "__main__":
    main_xor("config.json", "data.json")

