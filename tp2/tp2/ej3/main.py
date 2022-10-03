import json

import numpy as np
import pandas as pd

from tp2 import utils
from tp2.ej2 import graph
from tp2.perceptron import Perceptron


def res_index(x):
    res = np.zeros_like(x)
    res[np.argmax(x)] = 1
    return res


def main(config_path=None):
    if config_path is None:
        config_path = "tp2/ej3/config.json"

    path = "digitsformat.csv"

    df = pd.read_csv(path, sep=' ', header=None)

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

    matrix = np.zeros((21, 36))
    matrix2 = np.zeros((14, 21))
    matrix3 = np.zeros((10, 14))
    matr = [matrix, matrix2, matrix3]
    perceptron = Perceptron(
        matr, None, utils.tanh_arr, utils.tanh_diff, len(matrix) + 1, eta
    )

    data_min = np.min(data_matrix)
    data_normalised = np.subtract(
        2 * (np.subtract(data_matrix, data_min) / (np.max(data_matrix) - data_min)), 1
    )

    data_normalised = np.concatenate((data_normalised, res_matrix), axis=1)

    training_data = data_normalised[: round(len(data_normalised) / 2)]

    perceptron.train(training_data, error, max_iter, learning, res_index)

    for data in data_normalised:
        print(
            "expected: ",
            data,
            "\tout: ",
            np.argmax(perceptron.predict(data[:-1]))
        )

    # graph.plot(df)

if __name__ == "__main__":
    main("config.json")

