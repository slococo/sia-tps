import json

import numpy as np
import pandas as pd

from tp2 import utils
from tp2.ej2 import graph
from tp2.ej2 import animation
from tp2.perceptron import Perceptron


def main(config_path=None):
    if config_path is None:
        config_path = "tp2/ej2/config.json"

    path = "dataset.csv"
    data_column_names = ["x1", "x2", "x3"]
    expected_column_names = ["y"]

    df = pd.read_csv(path)

    data_matrix = df[data_column_names].to_numpy()
    res_matrix = df[expected_column_names].to_numpy()
    data_matrix = np.insert(data_matrix, 0, 1, axis=1)

    try:
        with open(config_path) as f:
            data = json.load(f)
            learning = data["learning"]
            max_iter = data["max_iter"]
            error = data["error"]
            eta = data["eta"]
    except FileNotFoundError:
        print("Couldn't find config path")
        exit(1)

    matrix = [np.atleast_2d(np.zeros((1, 4)))]
    perceptron = Perceptron(
        matrix, None, utils.tanh_arr, utils.tanh_diff, eta
    )

    res_min = np.min(res_matrix)
    res_normalised = np.subtract(
        2 * (np.subtract(res_matrix, res_min) / (np.max(res_matrix) - res_min)), 1
    )
    data_min = np.min(data_matrix)
    data_normalised = np.subtract(
        2 * (np.subtract(data_matrix, data_min) / (np.max(data_matrix) - data_min)), 1
    )

    data_normalised = np.concatenate((data_normalised, res_normalised), axis=1)

    # training_data = data_normalised[: round(len(data_normalised) / 2)]
    training_data = data_normalised[: round(len(data_normalised))]

    historic = perceptron.train(training_data, error, max_iter, learning)

    a, b = np.min(res_matrix), np.max(res_matrix)
    for data in data_normalised:
        print(
            "expected: ",
            utils.denormalise(data[-1], -1, 1, a, b),
            "\tout: ",
            utils.denormalise(perceptron.predict(data[:-1])[0], -1, 1, a, b),
        )

    perceptron.data = data_matrix
    perceptron.save()
    # graph.plot(df)
    animation.create_animation(data_matrix, historic)


if __name__ == "__main__":
    main("config.json")
