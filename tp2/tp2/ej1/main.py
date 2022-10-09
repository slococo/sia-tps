import json
import time

import matplotlib
import numpy as np
from tp2.error_graph import ErrorGraph
from tp2.initializer import Initializer
from tp2.loader import JSONLoader
from tp2.tester import Tester

matplotlib.use("TkAgg")

from tp2 import utils
from tp2.optimizer import *


def main(config_path=None, data_path=None):
    if config_path is None:
        config_path = "tp2/ej1/config.json"
    if data_path is None:
        data_path = "tp2/ej1/data.json"

    errors = []
    matr_dims = [1]
    for i in range(0, 5):
        perceptron, max_iter, error, learning, eta, dataset = Initializer.initialize(config_path, matr_dims, 2)
        data, exp = JSONLoader.load(data_path, True, dataset, True)

        start_time = time.time()
        historic, aux, layer_historic = perceptron.train(np.concatenate((data, np.atleast_2d(exp).T), 1), error, max_iter, learning)
        print("Zeit: {:.8f}ms".format((time.time() - start_time) / 1000))
        errors.append(aux)

        predict_error = Tester.test(perceptron, data, exp, utils.quadratic_error)

        print(f"Predict error: {predict_error}")

    ErrorGraph.plot_error(errors)

    # create_animation(
    #     data,
    #     historic,
    #     layer_historic,
    #     perceptron,
    #     learning,
    #     perceptron.g,
    # )


if __name__ == "__main__":
    main("config.json", "data.json")
