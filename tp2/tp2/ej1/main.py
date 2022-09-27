import json

import numpy as np

from tp2 import utils
from tp2.ej1 import graph
from tp2.perceptron import Perceptron


def main(config_path=None, data_path=None):
    if config_path is None:
        config_path = "tp2/ej1/config.json"
    if data_path is None:
        data_path = "tp2/ej1/data.json"

    with open(config_path) as f:
        data = json.load(f)
        dataset = data["dataset"]
        learning = data["learning"]
        max_iter = data["max_iter"]
        error = data["error"]
        eta = data["eta"]

    with open(data_path) as f:
        data = json.load(f)[dataset]
        matrix = np.zeros((1, 3))
        perceptron = Perceptron(
            matrix, None, utils.tanh_arr, utils.tanh_diff, len(matrix) + 1, eta
        )
        perceptron.train(data, error, max_iter, learning)
        print(perceptron.matrix_arr)
        print("x: 1 ~ y: 1")
        print(perceptron.predict([1, 1, 1]))
        print("x: -1 ~ y: -1")
        print(perceptron.predict([1, -1, -1]))
        print("x: -1 ~ y: 1")
        print(perceptron.predict([1, -1, 1]))
        print("x: 1 ~ y: -1")
        print(perceptron.predict([1, 1, -1]))

        perceptron.save()

        graph.plot(perceptron)


if __name__ == "__main__":
    main("config.json", "data.json")
