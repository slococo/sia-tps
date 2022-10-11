import json
import time

import matplotlib
import numpy as np
import pandas as pd
from tp2.ej3.c import plot_probability
from tp2.error_graph import ErrorGraph
from tp2.grapher import Grapher
from tp2.initializer import Initializer
from tp2.loader import CSVLoader
from tp2.tester import Tester

matplotlib.use("TkAgg")

from matplotlib import pyplot as plt
from tp2 import utils
from tp2.ej3.b.wrapper import Wrapper
from tp2.optimizer import *


def main(config_path=None, data_path=None, noisy_data_path=None):
    if config_path is None:
        config_path = "tp2/ej3/b/config.json"
    if data_path is None:
        data_path = "tp2/ej3/b/digitsformat.csv"
    if noisy_data_path is None:
        noisy_data_path = "tp2/ej3/b/noisy_digitsformat.csv"

    data_column = ["x" + str(i) for i in range(1, 36)]
    expected_column = ["y"]

    fig = plt.figure(figsize=(14, 9))

    data, exp = CSVLoader.load(data_path, False, data_column, expected_column, True)
    data = np.subtract(1, data)
    noisy_data_path = "noisy_digitsformat.csv"
    noisy_data, noisy_exp = CSVLoader.load(
        noisy_data_path, False, data_column, expected_column, True
    )

    # numbers = np.concatenate((data, noisy_data), axis=0)[:, 1:]
    # Grapher.graph_numbers(numbers)
    # exit(1)

    # optimizers = [rms_prop, adam, adamax, nadam, adadelta]
    # colors = ['green', 'yellow', 'purple', 'orange', 'brown', 'pink']
    optimizers = [rms_prop, adam, adamax, nadam]
    colors = ["green", "yellow", "purple", "orange"]
    # optimizers = [adadelta]
    # colors = ['brown']
    # optimizers = [gradient, momentum]
    # colors = ['red', 'blue']
    matr_dims = [15, 2]
    noisy_acums = []
    for j in range(0, len(optimizers)):
        print(optimizers[j].__name__)
        errors = []
        noisy_errors = []
        for i in range(0, 3):
            perceptron, max_iter, error, learning, eta, _ = Initializer.initialize(
                config_path, matr_dims, 35
            )
            perceptron.optimizer = optimizers[j]

            start_time = time.time()
            historic, aux, layer_historic, epoch = perceptron.train(
                np.concatenate((data, np.atleast_2d(exp)), 1),
                error,
                max_iter,
                learning,
                utils.res_index,
            )
            print("Error: ", aux[-1])
            print("Epochs: ", epoch)
            print("Zeit: {:.8f}s".format((time.time() - start_time)))
            errors.append(aux)

            noisy_error = Tester.test(
                perceptron,
                noisy_data,
                noisy_exp,
                utils.quadratic_error,
                utils.res_index,
            )
            noisy_errors.append(noisy_error)
            print(f"Noisy error: {noisy_error}")

            # wrapper = Wrapper(perceptron, data, historic, learning)
            # wrapper.save()
            #
            # exit(1)

            # plot_probability.plot_probabilities(perceptron, data, exp, "temp.png")

        noisy_acums.append(noisy_errors)

        ErrorGraph.make_plt(errors, colors[j])

    plt.legend([i.__name__.__str__() for i in optimizers])
    plt.show()
    fig.savefig("error" + round(time.time()).__str__() + ".png")
    plt.close()

    # fig = plt.figure(figsize=(14, 9))
    # ErrorGraph.bar_predict_graph(noisy_acums, colors, [i.__name__.__str__() for i in optimizers])
    # fig.savefig("predict" + round(time.time()).__str__() + ".png")
    # plt.show()


if __name__ == "__main__":
    main("config.json", "digitsformat.csv", "noisy_digitsformat.csv")
