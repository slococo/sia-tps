import json
import time

import matplotlib
import numpy as np

from tp2.ej1.animation import create_animation
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
    for i in range(0, 1):
        perceptron, max_iter, error, learning, eta, dataset = Initializer.initialize(config_path, matr_dims, 2)
        data, exp = JSONLoader.load(data_path, True, dataset, True)

        start_time = time.time()
        historic, aux, layer_historic, epoch = perceptron.train(np.concatenate((data, np.atleast_2d(exp).T), 1), error, max_iter, learning)
        print("Epochs: ", epoch)
        print("Error: ", aux[-1])
        print("Zeit: {:.8f}ms".format((time.time() - start_time) * 1000))
        errors.append(aux)

        predict_error = Tester.test(perceptron, data, exp, utils.quadratic_error)

        print(f"Predict error: {predict_error}")
        print()
        create_animation(
            data,
            historic,
            layer_historic,
            perceptron,
            learning,
            perceptron.g,
        )

    ErrorGraph.plot_error(errors)




if __name__ == "__main__":
    main("config.json", "data.json")
