import json

import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

from tp2 import utils
from tp2.ej3.c.wrapper import Wrapper
from tp2.perceptron import Perceptron


def main(config_path=None, data_path=None):
    if config_path is None:
        config_path = "tp2/ej3/a/config.json"
    if data_path is None:
        data_path = "tp2/ej3/a/data.json"

    with open(config_path) as f:
        data = json.load(f)
        learning = data["learning"]
        max_iter = data["max_iter"]
        error = data["error"]
        eta = data["eta"]

    with open(data_path) as f:
        data = json.load(f)["xor"]
        matr_dims = [6, 3, 1]
        perceptron = Perceptron(
            len(data[0]) - 1, matr_dims, None, utils.tanh_arr, utils.tanh_diff, eta
        )

        historic, errors = perceptron.train(data, error, max_iter, learning)

        print("x1: 1 ~ x2: 1 ~ exp = -1 ~ res = ", perceptron.predict([1, 1, 1]))
        print("x1: -1 ~ x2: -1 ~ exp = -1 ~ res = ", perceptron.predict([1, -1, -1]))
        print("x1: -1 ~ x2: 1 ~ exp = 1 ~ res = ", perceptron.predict([1, -1, 1]))
        print("x1: 1 ~ x2: -1 ~ exp = 1 ~ res = ", perceptron.predict([1, 1, -1]))

        wrapper = Wrapper(perceptron, data, historic, errors)
        wrapper.save()

        if wrapper.historic:
            fig = plt.figure(figsize=(14, 9))
            plt.plot(range(1, len(wrapper.errors) + 1), wrapper.errors)
            plt.show()


if __name__ == "__main__":
    main("config.json", "data.json")
