import time

import matplotlib
import numpy as np

from tp3 import utils
from tp3.ej1.grapher import Grapher
from tp3.initializer import Initializer
from tp3.loader import JSONLoader
from tp3.tester import Tester

matplotlib.use("TkAgg")


def main(config_path=None, data_path=None):
    if config_path is None:
        config_path = "tp3/ej1/config.json"
    if data_path is None:
        data_path = "tp3/ej1/data.json"

    errors = []
    matr_dims = [1]
    perceptron, max_iter, error, learning, eta, dataset = Initializer.initialize(
        config_path, matr_dims, 2
    )

    data, exp = JSONLoader.load(data_path, True, dataset, True)

    start_time = time.time()
    historic, aux, layer_historic, epoch = perceptron.train(
        np.concatenate((data, np.atleast_2d(exp).T), 1), error, max_iter, learning
    )
    print("Epochs: ", epoch)
    print("Error: ", aux[-1])
    print("Zeit: {:.8f}ms".format((time.time() - start_time) * 1000))
    errors.append(aux)

    predict_error = Tester.test(perceptron, data, exp, utils.quadratic_error)
    
    print(f"Predict error: {predict_error}")

    # Grapher.graph_data(data)


if __name__ == "__main__":
    main("config.json", "data.json")
