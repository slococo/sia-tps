import json
import time

from tp2 import utils
from tp2.error_graph import ErrorGraph
from tp2.initializer import Initializer
from tp2.loader import JSONLoader
from tp2.optimizer import *
from tp2.tester import Tester


def main(config_path=None, data_path=None):
    if config_path is None:
        config_path = "tp2/ej3/a/config.json"
    if data_path is None:
        data_path = "tp2/ej3/a/data.json"

    errors = []
    matr_dims = [5, 1]
    for i in range(0, 1):
        perceptron, max_iter, error, learning, eta, dataset = Initializer.initialize(
            config_path, matr_dims, 2
        )
        data, exp = JSONLoader.load(data_path, True, dataset, True)

        start_time = time.time()
        historic, aux, layer_historic, epochs = perceptron.train(
            np.concatenate((data, np.atleast_2d(exp).T), 1), error, max_iter, learning
        )
        print("Epochs: ", epochs)
        print("Error: ", aux[-1])
        print("Zeit: {:.8f}s".format((time.time() - start_time)))
        errors.append(aux)

        predict_error = Tester.test(perceptron, data, exp, utils.quadratic_error)

        print(f"Predict error: {predict_error}")

    ErrorGraph.plot_error(errors)


if __name__ == "__main__":
    main("config.json", "data.json")
