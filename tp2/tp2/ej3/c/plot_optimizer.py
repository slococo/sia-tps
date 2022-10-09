import json
import time

import matplotlib
import numpy as np
import pandas as pd

from tp2.error_graph import ErrorGraph
from tp2.initializer import Initializer
from tp2.loader import CSVLoader
from tp2.tester import Tester

matplotlib.use("TkAgg")

from matplotlib import pyplot as plt
from tp2 import utils
from tp2.ej3.c.wrapper import Wrapper
from tp2.optimizer import *


def main(config_path=None, data_path=None):
    if config_path is None:
        config_path = "tp2/ej3/c/config.json"
    if data_path is None:
        data_path = "tp2/ej3/c/digitsformat.csv"

    data_column = ["x" + str(i) for i in range(1, 36)]
    expected_column = ["y"]

    data, exp = CSVLoader.load(data_path, False, data_column, expected_column, True)
    data = np.subtract(1, data)
    optimizers = [gradient, momentum, rms_prop, adam, adamax, nadam]
    colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'brown']
    matr_dims = [20, 10]
    for j in range(0, len(optimizers)):
        print(optimizers[j].__name__)
        errors = []
        for i in range(0, 1):
            perceptron, max_iter, error, learning, eta, _ = Initializer.initialize(config_path, matr_dims, 35)
            perceptron.optimizer = optimizers[j]

            start_time = time.time()
            historic, aux, layer_historic = perceptron.train(np.concatenate((data, np.atleast_2d(exp)), 1), error, max_iter, learning, utils.res_index)
            print("Zeit: {:.8f}s".format((time.time() - start_time)))
            errors.append(aux)

            predict_error = Tester.test(perceptron, data, exp, utils.quadratic_error, utils.res_index)
            print(f"Predict error: {predict_error}")

        ErrorGraph.make_plt(errors, colors[j])

    plt.legend([i.__name__.__str__() for i in optimizers])
    plt.show()


if __name__ == "__main__":
    main("config.json", "digitsformat.csv")

