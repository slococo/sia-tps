import random
import time

import numpy
import numpy as np

from tp3 import utils
from tp3.ej1.grapher import Grapher
from tp3.initializer import Initializer
from tp3.loader import CSVLoader
from tp3.noise import gaussian_noise, uniform_noise
from tp3.tester import Tester


def main(config_path=None, data_path=None):
    if config_path is None:
        config_path = "tp3/ej1/b/config.json"
    if data_path is None:
        data_path = "tp3/ej1/fonts.csv"

    errors = []
    # matr_dims = [35, 35, 35, 35]
    matr_dims = [25, 17, 4, 17, 25, 35]
    perceptron, max_iter, error, learning, eta, dataset = Initializer.initialize(
        config_path, matr_dims, 35, "ae"
    )

    data_column = ["x" + str(i) for i in range(1, 36)]
    data, exp = CSVLoader.load(data_path, False, data_column, None, False)
    # data_low = data[0:24]
    exp_low = exp[0:24]

    # data_high = data[24:]
    # data_aux = gaussian_noise(0, 0.1, data_high)
    data_aux = []
    exp_aux = []
    # for i in range(0, 3):
        # if not i:
    exp_aux = exp_low
    # else:
    #     exp_aux = np.concatenate((exp_aux, exp_low))
    for char in data[:24, 1:]:
        # v = np.random.exponential(0.2)
        # data_aux.append(gaussian_noise(0, v, char))
        data_aux.append(char)

    # for i in range(0, len(data_aux)):
    #     for j in range(0, len(data_aux[i])):
    #         if data_aux[i][j] > 1:
    #             data_aux[i][j] = 1
    #         elif data_aux[i][j] < -1:
    #             data_aux[i][j] = -1

    Grapher.graph_chars(data_aux, 5, 7, 8)

    data_aux = np.insert(data_aux, 0, 1, axis=1)

    start_time = time.time()
    historic, aux, layer_historic, epoch = perceptron.train(
        data_aux, np.atleast_2d(exp_aux), error, max_iter, learning
    )
    print("Epochs: ", epoch)
    print("Error: ", aux[-1])
    print("Zeit: {:.8f}ms".format((time.time() - start_time) * 1000))
    errors.append(aux)

    exp_aux = data[:, 1:]
    data_aux = []
    for char in data[:, 1:]:
        # v = np.random.exponential(0.2)
        # data_aux.append(gaussian_noise(0, v, char))
        data_aux.append(char)

    # for i in range(0, len(data_aux)):
    #     for j in range(0, len(data_aux[i])):
    #         if data_aux[i][j] > 1:
    #             data_aux[i][j] = 1
    #         elif data_aux[i][j] < -1:
    #             data_aux[i][j] = -1

    res = []
    data_aux2 = np.insert(data_aux, 0, 1, axis=1)
    for i in range(0, len(data_aux2)):
        res.append(perceptron.predict(data_aux2[i]))

    Grapher.graph_chars(data_aux, 5, 7, 8)
    Grapher.graph_chars(res, 5, 7, 8)

    # Grapher.graph_chars(data_aux, 5, 7, 4)

    # predict_error = Tester.test(perceptron, data_aux, exp_aux, utils.quadratic_error)
    # print(f"Predict error: {predict_error}")

    # for i in range(0, len(data_high)):
    #     Grapher.graph_in_out(data_aux[i], perceptron.predict(data_aux[i]))


if __name__ == "__main__":
    main("config.json", "../fonts.csv")
