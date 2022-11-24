import time

import numpy as np
from matplotlib import pyplot as plt

from tp3 import optimizer, perceptron
from tp3.ej1.grapher import Grapher
from tp3.error_graph import ErrorGraph
from tp3.initializer import Initializer
from tp3.loader import CSVLoader
from tp3.perceptron import Perceptron


def main(config_path=None, data_path=None):
    if config_path is None:
        config_path = "tp3/ej1/a/config.json"
    if data_path is None:
        data_path = "tp3/ej1/fonts.csv"

    data_column = ["x" + str(i) for i in range(1, 36)]
    data, exp = CSVLoader.load(data_path, False, data_column, None, False)

    bs = [0.25, 0.15, 0.05]

    etas = [0.001, 0.005]

    fig = plt.figure(figsize=(14, 9))

    colors = ["yellow", "green", "red", "brown", "indigo", "blue"]
    for i in range(0, len(etas)):
        for j in range(0, len(bs)):
            errors = []
            for _ in range(0, 2):
                optimizer.reset_state()
                (
                    perceptron,
                    max_iter,
                    error,
                    learning,
                    eta,
                    dataset,
                ) = Initializer.initialize(
                    config_path, [25, 17, 2, 17, 25, 35], 35, "ae"
                )
                start_time = time.time()
                historic, aux, layer_historic, epoch = perceptron.train(
                    data, np.atleast_2d(exp), error, max_iter, learning
                )
                print("Epochs: ", epoch)
                print("Error: ", aux[-1])
                print("Zeit: {:.8f}ms".format((time.time() - start_time) * 1000))
                errors.append(aux)

            ErrorGraph.make_plt(errors, colors[i * len(bs) + j])

    plt.xlim([0, 20000])
    plt.ylim([-0.01, 0.3])
    labels = [
        "b = 0.25 ~ eta = 0.001",
        "b = 0.15 ~ eta = 0.001",
        "b = 0.05 ~ eta = 0.001",
        "b = 0.25 ~ eta = 0.005",
        "b = 0.15 ~ eta = 0.005",
        "b = 0.05 ~ eta = 0.005",
    ]
    plt.legend(labels)
    plt.show()
    fig.savefig("params" + round(time.time()).__str__() + ".png")
    plt.close()


if __name__ == "__main__":
    main("../tp3/ej1/a/config.json", "../fonts.csv")
