import json

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("TkAgg")

from matplotlib import pyplot as plt

from tp2 import utils
from tp2.ej3.c.wrapper import Wrapper
from tp2.perceptron import Perceptron


def res_index(x):
    res = np.full_like(x, fill_value=-1)
    res[np.argmax(x)] = 1
    return res


def main(config_path=None):
    if config_path is None:
        config_path = "tp2/ej3/c/config.json"

    path = "digitsformat.csv"

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

    matr_dims = [21, 10]
    perceptron = Perceptron(
        len(data_matrix[0]), matr_dims, None, utils.tanh_arr, utils.tanh_diff, eta
    )

    # data_min = np.min(data_matrix)
    # data_normalised = np.subtract(
    #     2 * (np.subtract(data_matrix, data_min) / (np.max(data_matrix) - data_min)), 1
    # )
    data_normalised = data_matrix
    data_normalised = np.concatenate((data_normalised, res_matrix), axis=1)

    training_data = data_normalised

    # perceptron.train(training_data, error, max_iter, learning, res_index)
    historic, errors = perceptron.train(training_data, error, max_iter, learning, res_index)

    for data in data_normalised:
        print(
            "expected: ",
            data[-1],
            "\tout: ",
            np.argmax(perceptron.predict(data[:-1]))
        )

    wrapper = Wrapper(perceptron, data, historic, errors)
    wrapper.save()

    if wrapper.historic:
        fig = plt.figure(figsize=(14, 9))
        plt.plot(range(1, len(wrapper.errors) + 1), wrapper.errors)
        plt.show()


if __name__ == "__main__":
    main("config.json")
    # wrapper = Wrapper.load()
    # print(wrapper.perceptron.matrix_arr)

