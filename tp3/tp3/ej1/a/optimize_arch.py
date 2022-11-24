import time

import numpy as np
from matplotlib import pyplot as plt

from tp3.ej1.grapher import Grapher
from tp3.error_graph import ErrorGraph
from tp3.initializer import Initializer
from tp3.loader import CSVLoader
from tp3.perceptron import Perceptron
from tp3 import optimizer, perceptron


def main(config_path=None, data_path=None):
    if config_path is None:
        config_path = "tp3/ej1/a/config.json"
    if data_path is None:
        data_path = "tp3/ej1/fonts.csv"

    data_column = ["x" + str(i) for i in range(1, 36)]
    data, exp = CSVLoader.load(data_path, False, data_column, None, False)

    architectures = [
        # [2, 35],
        # [10, 2, 10, 35],
        # [17, 2, 17, 35],
        # [25, 2, 25, 35],
        [25, 12, 2, 12, 25, 35],
        [25, 17, 2, 17, 25, 35]
    ]

    # iters = [40000, 40000, 40000, 40000, 40000, 40000]
    # iters = [50000, 50000, 50000]
    iters = [20000, 20000]
    # iters = [1000, 1000, 1000, 1000, 1000, 1000]

    fig = plt.figure(figsize=(14, 9))

    # colors = ["yellow", "green", "red", "brown", "indigo", "blue"]
    # colors = ["brown", "indigo", "blue"]
    colors = ["indigo", "blue"]
    for i in range(0, len(architectures)):
        errors = []
        for _ in range(0, 2):
            optimizer.reset_state()
            perceptron, max_iter, error, learning, eta, dataset = Initializer.initialize(
                config_path, architectures[i], 35, "ae"
            )
            start_time = time.time()
            historic, aux, layer_historic, epoch = perceptron.train(
                data, np.atleast_2d(exp), error, iters[i], learning
            )
            print("Epochs: ", epoch)
            print("Error: ", aux[-1])
            print("Zeit: {:.8f}ms".format((time.time() - start_time) * 1000))
            errors.append(aux)

        ErrorGraph.make_plt(errors, colors[i])

    plt.xlim([0, 40000])
    plt.ylim([-0.01, 0.3])
    architectures_print = [
        # [35, 2, 35],
        # [35, 10, 2, 10, 35],
        # [35, 17, 2, 17, 35],
        # [35, 25, 2, 25, 35],
        [35, 25, 12, 2, 12, 25, 35],
        [35, 25, 17, 2, 17, 25, 35]
    ]
    plt.legend([i.__str__() for i in architectures_print])
    plt.show()
    fig.savefig("error" + round(time.time()).__str__() + ".png")
    plt.close()


if __name__ == "__main__":
    main("config.json", "../fonts.csv")
