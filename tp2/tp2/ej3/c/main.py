import json
import time

import matplotlib
import numpy as np
import pandas as pd

from tp2.error_graph import ErrorGraph
from tp2.initializer import Initializer
from tp2.loader import CSVLoader
from tp2.tester import Tester

matplotlib.use("TkAgg")

from matplotlib import pyplot as plt
from tp2 import utils
from tp2.ej3.c.wrapper import Wrapper
from tp2.optimizer import *


def main(config_path=None, data_path=None, noisy_data_path=None):
    if config_path is None:
        config_path = "tp2/ej3/c/config.json"
    if data_path is None:
        data_path = "tp2/ej3/c/digitsformat.csv"
    if noisy_data_path is None:
        noisy_data_path = "tp2/ej3/c/noisy_digitsformat.csv"

    data_column = ["x" + str(i) for i in range(1, 36)]
    expected_column = ["y"]

    data, exp = CSVLoader.load(data_path, False, data_column, expected_column, True)
    data = np.subtract(1, data)
    noisy_data, noisy_exp = CSVLoader.load(noisy_data_path, False, data_column, expected_column, True)
    errors = []
    matr_dims = [20, 10]
    for i in range(0, 2):
        perceptron, max_iter, error, learning, eta, _ = Initializer.initialize(config_path, matr_dims, 35)

        start_time = time.time()
        historic, aux, layer_historic, epoch = perceptron.train(np.concatenate((data, np.atleast_2d(exp)), 1), error, max_iter, learning, utils.res_index)
        print("Error: ", aux[-1])
        print("Epochs: ", epoch)
        print("Zeit: {:.8f}s".format((time.time() - start_time)))
        errors.append(aux)

        predict_error = Tester.test(perceptron, data, exp, utils.quadratic_error, utils.res_index)
        print(f"Predict error: {predict_error}")

        noisy_error = Tester.test(perceptron, noisy_data, noisy_exp, utils.quadratic_error, utils.res_index)
        print(f"Noisy error: {noisy_error}")

    ErrorGraph.plot_error(errors)



    # data_column_names = ["x" + str(i) for i in range(1, 36)]
    # expected_column_names = ["y"]
    #
    # df = pd.read_csv(path)
    #
    # data_matrix = df[data_column_names].to_numpy()
    # res_matrix = df[expected_column_names].to_numpy()
    # data_matrix = np.insert(data_matrix, 0, 1, axis=1)
    #
    # noisy_df = pd.read_csv("noisy_digitsformat.csv")
    #
    # noisy_data_matrix = noisy_df[data_column_names].to_numpy()
    # noisy_res_matrix = noisy_df[expected_column_names].to_numpy()
    # noisy_data_matrix = np.insert(noisy_data_matrix, 0, 1, axis=1)
    #
    # try:
    #     with open(config_path) as f:
    #         data = json.load(f)
    #         learning = data["learning"]
    #         optimizer = data["optimizer"]
    #         g_function = data["g_function"]
    #         eta_adapt = data["eta_adapt"]
    #         beta = data["beta"]
    #         max_iter = data["max_iter"]
    #         error = data["error"]
    #         eta = data["eta"]
    # except FileNotFoundError:
    #     raise "Couldn't find config path"
    #
    # optimizer = globals()[optimizer]
    # if eta_adapt is not None:
    #     eta_adapt = globals()[eta_adapt]
    # match g_function:
    #     case "tanh":
    #         g_function = utils.tanh_arr
    #         g_diff = utils.tanh_diff
    #     case "identity":
    #         g_function = utils.identity
    #         g_diff = utils.ident_diff
    #     case "logistic":
    #         g_function = utils.logistic_arr
    #         g_diff = utils.logistic_diff
    #     case _:
    #         g_function = np.sign
    #         g_diff = utils.ident_diff
    # utils.set_b(beta)
    #
    # matr_dims = [21, 10]
    # perceptron = Perceptron(
    #     len(data_matrix[0]), matr_dims, optimizer, g_function, g_diff, eta, eta_adapt
    # )
    # # perceptron = Wrapper.load().perceptron
    #
    # data_min = np.min(data_matrix)
    # data_normalised = np.subtract(
    #     2 * (np.subtract(data_matrix, data_min) / (np.max(data_matrix) - data_min)), 1
    # )
    #
    # data_normalised = np.concatenate((data_normalised, res_matrix), axis=1)
    #
    # training_data = data_normalised
    #
    # start_time = time.time()
    # historic, errors, _ = perceptron.train(
    #     training_data, error, max_iter, learning, res_index
    # )
    # print("Zeit: {:.2f}s".format((time.time() - start_time)))
    #
    # for data in data_normalised:
    #     print(
    #         "expected: ", data[-1], "\tout: ", np.argmax(perceptron.predict(data[:-1]))
    #     )
    #
    # noisy_data_min = np.min(noisy_data_matrix)
    # noisy_data_normalised = np.subtract(
    #     2 * (np.subtract(noisy_data_matrix, noisy_data_min) / (np.max(noisy_data_matrix) - noisy_data_min)), 1
    # )
    #
    # noisy_data_normalised = np.concatenate((noisy_data_normalised, noisy_res_matrix), axis=1)
    #
    # print("\nNoisy: ")
    # for noisy_data in noisy_data_normalised:
    #     # aux = noisy_data[1:-1]
    #     # plt.imshow(utils.denormalise(aux, -1, 1, 0, 1).reshape(7, 5) * [1, 1, 1], interpolation="nearest")
    #     # plt.show()
    #     print(
    #         "expected: ", noisy_data[-1], "\tout: ", np.argmax(perceptron.predict(noisy_data[:-1]))
    #     )
    #
    # wrapper = Wrapper(perceptron, data, historic, errors)
    # wrapper.save()
    #
    # if wrapper.historic:
    #     fig = plt.figure(figsize=(14, 9))
    #     plt.plot(range(1, len(wrapper.errors) + 1), wrapper.errors)
    #     plt.ylim(0, 1)
    #     plt.show()


if __name__ == "__main__":
    main("config.json", "digitsformat.csv", "noisy_digitsformat.csv")
