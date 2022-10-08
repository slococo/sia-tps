import json
import time

import matplotlib
import numpy as np
from tp2.ej1.animation import create_animation

matplotlib.use("TkAgg")

from matplotlib import pyplot as plt
from tp2 import utils
from tp2.ej1.wrapper import Wrapper
from tp2.optimizer import *
from tp2.perceptron import Perceptron


def main(config_path=None, data_path=None):
    if config_path is None:
        config_path = "tp2/ej1/config.json"
    if data_path is None:
        data_path = "tp2/ej1/data.json"

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
            beta = 1
    except FileNotFoundError:
        raise "Couldn't find config path"

    optimizer = globals()[optimizer]
    if eta_adapt is not None:
        eta_adapt = globals()[eta_adapt]
    match g_function:
        case "tanh":
            g = utils.tanh_arr
            g_diff = utils.tanh_diff
        case "identity":
            g = utils.identity
            g_diff = utils.ident_diff
        case "logistic":
            g = utils.logistic_arr
            g_diff = utils.logistic_diff
        case _:
            g = np.sign
            g_diff = utils.ident_diff
    utils.set_b(beta)

    try:
        with open(data_path) as f:
            data = json.load(f)[dataset]
    except FileNotFoundError:
        raise "Couldn't find config path"

    matr_dims = [1]
    perceptron = Perceptron(
        len(data[0]) - 1, matr_dims, optimizer, g, g_diff, eta, eta_adapt
    )

    start_time = time.time()
    historic, errors, layer_historic = perceptron.train(data, error, max_iter, learning)
    print("Zeit: {:.8f}ms".format((time.time() - start_time) / 1000))

    for i in data:
        print(i[:-1], i[-1], perceptron.predict(i[:-1]))

    wrapper = Wrapper(
        perceptron, data, historic, layer_historic, errors, learning, g_function
    )
    wrapper.save()

    if wrapper.errors:
        fig = plt.figure(figsize=(14, 9))
        plt.plot(range(1, len(wrapper.errors) + 1), wrapper.errors)
        plt.ylim(0, 1)
        fig.savefig(
            "anim-"
            + g_function
            + "-"
            + learning
            + "-"
            + perceptron.optimizer.__name__
            + "-error.png",
            dpi=fig.dpi,
        )
        plt.show()

    # if wrapper.historic:
    #     create_animation(
    #         wrapper.data,
    #         wrapper.historic,
    #         wrapper.layer_historic,
    #         wrapper.perceptron,
    #         wrapper.learning,
    #         wrapper.g_function,
    #     )


if __name__ == "__main__":
    main("config.json", "data.json")
