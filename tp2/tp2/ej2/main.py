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
from tp2.ej2 import animation, graph
from tp2.ej2.wrapper import Wrapper
from tp2.optimizer import *
from tp2.perceptron import Perceptron
from tp2.datapartitioning import *


def main(config_path=None, data_path=None):
    if config_path is None:
        config_path = "tp2/ej2/config.json"
    if data_path is None:
        data_path = "tp2/ej2/dataset.csv"

    data_column = ["x1", "x2", "x3"]
    expected_column = ["y"]

    data, exp = CSVLoader.load(data_path, False, data_column, expected_column, False)
    errors = []
    matr_dims = [5, 1]
    for i in range(0, 2):
        perceptron, max_iter, error, learning, eta, _ = Initializer.initialize(config_path, matr_dims, 3)

        start_time = time.time()
        historic, aux, layer_historic, epoch = perceptron.train(np.concatenate((data, np.atleast_2d(exp)), 1), error, max_iter, learning)
        print("Epochs: ", epoch)
        print("Error: ", aux[-1])
        print("Zeit: {:.8f}s".format((time.time() - start_time)))
        errors.append(aux)

        predict_error = Tester.test(perceptron, data, exp, utils.quadratic_error)

        print(f"Predict error: {predict_error}")

    ErrorGraph.plot_error(errors)

    # path = "dataset.csv"
    #
    #
    # df = pd.read_csv(path)
    #
    # data_matrix = df[data_column_names].to_numpy()
    # res_matrix = df[expected_column_names].to_numpy()
    # data_matrix = np.insert(data_matrix, 0, 1, axis=1)
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
    #         partitioning_method = data["partitioning_method"]
    #         partitioning_parameter = data["partitioning_parameter"]
    # except FileNotFoundError:
    #     raise "Couldn't find config path"
    #
    # if partitioning_parameter is None or partitioning_method is None:
    #     exit(1)
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
    # matr_dims = [1]
    # # perceptron = Perceptron(
    # #     len(data_matrix[0]), matr_dims, optimizer, g_function, g_diff, eta, eta_adapt
    # # )
    #
    # res_min = np.min(res_matrix)
    # res_normalised = np.subtract(
    #     2 * (np.subtract(res_matrix, res_min) / (np.max(res_matrix) - res_min)), 1
    # )
    # data_min = np.min(data_matrix)
    # data_normalised = np.subtract(
    #     2 * (np.subtract(data_matrix, data_min) / (np.max(data_matrix) - data_min)), 1
    # )
    #
    # data_normalised = np.concatenate((data_normalised, res_normalised), axis=1)
    #
    # #training_data = data_normalised[: round(len(data_normalised) / 2)]
    # #training_data = data_normalised[: round(len(data_normalised))]
    #
    # start_time = time.time()
    #
    # match partitioning_method:
    #     case "holdout":
    #         f = holdout
    #     case "k_fold":
    #         f = k_fold
    #
    # historic, errors, returned_result = f(
    #     len(data_matrix[0]),
    #     matr_dims,
    #     optimizer,
    #     g_function,
    #     g_diff,
    #     eta,
    #     eta_adapt,
    #     partitioning_parameter,
    #     data_normalised,
    #     error,
    #     max_iter,
    #     learning
    # )
    #
    # testing_result: TestingResult = returned_result
    #
    # # historic, errors, _ = perceptron.train(training_data, error, max_iter, learning)
    # # print("Zeit: {:.2f}s".format((time.time() - start_time)))
    #
    # a, b = np.min(res_matrix), np.max(res_matrix)
    #
    # match partitioning_method:
    #     case "holdout":
    #         print("Training probability: " + str(partitioning_parameter))
    #     case "k-fold":
    #         print("Fold number: " + str(partitioning_parameter))
    #
    # global_error = 0
    # for i in range(len(testing_result.expected)):
    #     local_error = 0
    #     print("Perceptron number: " + str(i))
    #     for j in range(len(testing_result.expected[i])):
    #         print(
    #                 "expected: ",
    #                 utils.denormalise(testing_result.expected[i][j], -1, 1, a, b),
    #                 "\tout: ",
    #                 utils.denormalise(testing_result.results[i][j], -1, 1, a, b),
    #         )
    #         local_error += np.average((np.subtract(testing_result.expected[i][j], testing_result.results[i][j]) / 2) ** 2)
    #
    #     print("Perceptron error: " + str(local_error))
    #     global_error += local_error
    #
    # global_error /= len(testing_result.expected)
    # print("Algorithm error: " + str(global_error))
    #
    # # predict_error = 0
    # # for data in data_normalised:
    # #    pred = perceptron.predict(data[:-1])
    # #    predict_error += np.average((np.subtract(data[-1], pred) / 2) ** 2)
    # #    print(
    # #        "expected: ",
    # #        utils.denormalise(data[-1:], -1, 1, a, b),
    # #        "\tout: ",
    # #        utils.denormalise(pred, -1, 1, a, b),
    # #    )
    #
    # # predict_error /= len(data_normalised)
    # # print("Error with full data: ", predict_error)
    #
    # if historic:
    #     fig = plt.figure(figsize=(14, 9))
    #     plt.plot(range(1, len(errors) + 1), errors)
    #     plt.show()


if __name__ == "__main__":
    main("config.json", "dataset.csv")
