import time

import matplotlib.pyplot as plt
import numpy as np

from tp3 import utils
from tp3.ej1.a.wrapper import Wrapper
from tp3.ej1.grapher import Grapher
from tp3.error_graph import ErrorGraph
from tp3.initializer import Initializer
from tp3.loader import CSVLoader
from tp3.tester import Tester


def main(config_path=None, data_path=None):
    if config_path is None:
        config_path = "tp3/ej1/a/config.json"
    if data_path is None:
        data_path = "tp3/ej1/fonts.csv"

    errors = []
    # matr_dims = [15, 2, 15, 35]
    # matr_dims = [25, 12, 2, 12, 25, 35]
    # matr_dims = [20, 10, 2, 10, 20, 35]
    matr_dims = [25, 17, 2, 17, 25, 35]
    perceptron, max_iter, error, learning, eta, dataset = Initializer.initialize(
        config_path, matr_dims, 35, "ae"
    )

    data_column = ["x" + str(i) for i in range(1, 36)]
    data, exp = CSVLoader.load(data_path, False, data_column, None, False)

    start_time = time.time()
    historic, aux, layer_historic, epoch = perceptron.train(
        data, np.atleast_2d(exp), error, max_iter, learning, res_fun=None
    )
    print("Epochs: ", epoch)
    print("Error: ", aux[-1])
    print("Zeit: {:.8f}ms".format((time.time() - start_time) * 1000))
    errors.append(aux)

    predict_error = Tester.test(
        perceptron, data, exp, utils.quadratic_error, res_fun=None
    )

    print(f"Predict error: {predict_error}")

    ErrorGraph.make_plt(errors, "red")
    plt.show()

    perceptron.save()
    # wrapper = Wrapper(perceptron, data, historic, errors)
    # wrapper.save()

    # for i in range(0, len(data)):
    #     Grapher.graph_in_out(data[i], perceptron.predict(data[i]))


if __name__ == "__main__":
    main("config.json", "../fonts.csv")
