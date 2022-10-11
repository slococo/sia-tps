import json
import time

import matplotlib
import numpy as np
import pandas as pd
from tp2.error_graph import ErrorGraph
from tp2.initializer import Initializer
from tp2.loader import CSVLoader, JSONLoader
from tp2.tester import Tester

matplotlib.use("TkAgg")

from matplotlib import pyplot as plt
from tp2 import utils
from tp2.ej3.c.wrapper import Wrapper
from tp2.optimizer import *


def main(config_path=None, data_path=None):
    if config_path is None:
        config_path = "tp2/ej1/config.json"
    if data_path is None:
        data_path = "tp2/ej1/data.json"

    fig = plt.figure(figsize=(14, 9))

    # optimizers = [gradient, momentum, rms_prop, adam, adamax, nadam]
    # colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'brown']
    optimizers = [adadelta]
    colors = ["olive"]
    matr_dims = [1]
    for j in range(0, len(optimizers)):
        print(optimizers[j].__name__)
        errors = []
        for i in range(0, 5):
            (
                perceptron,
                max_iter,
                error,
                learning,
                eta,
                dataset,
            ) = Initializer.initialize(config_path, matr_dims, 2)
            data, exp = JSONLoader.load(data_path, True, dataset, True)
            perceptron.optimizer = optimizers[j]

            start_time = time.time()
            historic, aux, layer_historic, epoch = perceptron.train(
                np.concatenate((data, np.atleast_2d(exp).T), 1),
                error,
                max_iter,
                learning,
            )
            print("Error: ", aux[-1])
            print("Epochs: ", epoch)
            print("Zeit: {:.8f}s".format((time.time() - start_time)))
            errors.append(aux)

            predict_error = Tester.test(perceptron, data, exp, utils.quadratic_error)
            print(f"Predict error: {predict_error}")

        max_len = 0
        for error in errors:
            max_len = max(len(error), max_len)
        for i in range(0, len(errors)):
            aux = np.atleast_1d(errors[i])
            aux.resize(max_len)
            errors[i] = aux
            # errors[i] = np.array(list(map(help, aux)))

        ErrorGraph.make_plt(errors, colors[j])

    plt.legend([i.__name__.__str__() for i in optimizers])
    # plt.show()
    fig.savefig("fig" + round(time.time()).__str__() + ".png")
    plt.close()


def help(x):
    if x == 0:
        return 0.25
    return x


if __name__ == "__main__":
    main("config.json", "data.json")
