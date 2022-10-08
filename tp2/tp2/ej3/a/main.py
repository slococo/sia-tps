import json
import time

import matplotlib
import numpy as np

matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
from tp2 import utils
from tp2.ej3.c.wrapper import Wrapper
from tp2.optimizer import *
from tp2.perceptron import Perceptron


def main(config_path=None, data_path=None):
    if config_path is None:
        config_path = "tp2/ej3/a/config.json"
    if data_path is None:
        data_path = "tp2/ej3/a/data.json"

    try:
        with open(config_path) as f:
            data = json.load(f)
            dataset = data["dataset"]
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

    try:
        with open(data_path) as f:
            data = json.load(f)[dataset]
    except FileNotFoundError:
        raise "Couldn't find config path"

    matr_dims = [6, 3, 1]
    perceptron = Perceptron(
        len(data[0]) - 1, matr_dims, optimizer, g_function, g_diff, eta, eta_adapt
    )

    start_time = time.time()
    historic, errors, _ = perceptron.train(data, error, max_iter, learning)
    print("Zeit: {:.2f}s".format((time.time() - start_time)))

    print("x1: 1 ~ x2: 1 ~ exp = -1 ~ res = ", perceptron.predict([1, 1, 1]))
    print("x1: -1 ~ x2: -1 ~ exp = -1 ~ res = ", perceptron.predict([1, -1, -1]))
    print("x1: -1 ~ x2: 1 ~ exp = 1 ~ res = ", perceptron.predict([1, -1, 1]))
    print("x1: 1 ~ x2: -1 ~ exp = 1 ~ res = ", perceptron.predict([1, 1, -1]))

    wrapper = Wrapper(perceptron, data, historic, errors)
    wrapper.save()

    if wrapper.historic:
        fig = plt.figure(figsize=(14, 9))
        plt.plot(range(1, len(wrapper.errors) + 1), wrapper.errors)
        plt.show()


if __name__ == "__main__":
    main("config.json", "data.json")
