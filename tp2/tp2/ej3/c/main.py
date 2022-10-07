import json
import time

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("TkAgg")

from matplotlib import pyplot as plt
from tp2 import utils
from tp2.ej3.c.wrapper import Wrapper
from tp2.optimizer import adaptative_eta, momentum
from tp2.perceptron import Perceptron


def res_index(x, n):
    res = np.full_like(x, fill_value=-1)
    res[round(n)] = 1
    return res


def main(config_path=None, data_path=None):
    if config_path is None:
        config_path = "tp2/ej3/c/config.json"

    if data_path is None:
        path = "tp2/ej3/c/digitsformat.csv"
    else:
        path = data_path

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
            optimizer = data["optimizer"]
            g_function = data["g_function"]
            eta_adapt = data["eta_adapt"]
            beta = data["beta"]
            max_iter = data["max_iter"]
            error = data["error"]
            eta = data["eta"]
    except FileNotFoundError:
        raise "Couldn't find config path"

    optimizer = globals()[optimizer]
    if eta_adapt is not None:
        eta_adapt = globals()[eta_adapt]
    match g_function:
        case "tanh":
            g_function = utils.tanh_arr
            g_diff = utils.tanh_diff
        case "identity":
            g_function = utils.identity
            g_diff = utils.ident_diff
        case "logistic":
            g_function = utils.logistic_arr
            g_diff = utils.logistic_diff
        case _:
            g_function = np.sign
            g_diff = utils.ident_diff
    utils.set_b(beta)

    # matr_dims = [30, 22, 16, 10]
    # perceptron = Perceptron(
    #     len(data_matrix[0]), matr_dims, optimizer, g_function, g_diff, eta, eta_adapt, 0.8, 0.5
    # )
    matr_dims = [21, 10]
    perceptron = Perceptron(
        len(data_matrix[0]), matr_dims, optimizer, g_function, g_diff, eta, eta_adapt
    )

    data_min = np.min(data_matrix)
    data_normalised = np.subtract(
        2 * (np.subtract(data_matrix, data_min) / (np.max(data_matrix) - data_min)), 1
    )

    data_normalised = np.concatenate((data_normalised, res_matrix), axis=1)

    training_data = data_normalised

    start_time = time.time()
    historic, errors, _ = perceptron.train(
        training_data, error, max_iter, learning, res_index
    )
    print("Zeit: {:.2f}s".format((time.time() - start_time)))

    for data in data_normalised:
        print(
            "expected: ", data[-1], "\tout: ", np.argmax(perceptron.predict(data[:-1]))
        )

    wrapper = Wrapper(perceptron, data, historic, errors)
    # wrapper.save()

    if wrapper.historic:
        fig = plt.figure(figsize=(14, 9))
        plt.plot(range(1, len(wrapper.errors) + 1), wrapper.errors)
        plt.ylim(0, 1)
        plt.show()


if __name__ == "__main__":
    main("config.json", "digitsformat.csv")
