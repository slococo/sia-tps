import time

import numpy as np

from tp3.ej1.grapher import Grapher
from tp3.error_graph import ErrorGraph
from tp3.initializer import Initializer
from tp3.loader import CSVLoader
from tp3.perceptron import Perceptron
from tp3 import optimizer, perceptron

import matplotlib

matplotlib.use("TkAgg")

import matplotlib.pyplot as plt


def main(config_path=None, data_path=None):
    if config_path is None:
        config_path = "tp3/ej1/a/config.json"
    if data_path is None:
        data_path = "tp3/ej1/fonts.csv"

    # perceptron = Perceptron.load()

    data_column = ["x" + str(i) for i in range(1, 36)]
    data, exp = CSVLoader.load(data_path, False, data_column, None, False)

    matr_dims = [25, 2, 25, 35]
    perceptron, max_iter, error, learning, eta, dataset = Initializer.initialize(
        config_path, matr_dims, 35, "ae"
    )

    fig = plt.figure(figsize=(14, 9))

    optimizers = [optimizer.momentum, optimizer.adam, optimizer.rms_prop, optimizer.nadam]
    colors = ["yellow", "green", "red", "purple"]
    for j in range(0, len(optimizers)):
        print(optimizers[j].__name__)
        errors = []
        for i in range(0, 2):
            start_time = time.time()
            historic, aux, layer_historic, epoch = perceptron.train(
                data, np.atleast_2d(exp), error, max_iter, learning
            )
            print("Epochs: ", epoch)
            print("Error: ", aux[-1])
            print("Zeit: {:.8f}ms".format((time.time() - start_time) * 1000))
            errors.append(aux)

        ErrorGraph.make_plt(errors, colors[j])

    plt.legend([i.__name__.__str__() for i in optimizers])
    plt.show()
    fig.savefig("error" + round(time.time()).__str__() + ".png")
    plt.close()


if __name__ == "__main__":
    main("config.json", "../fonts.csv")
