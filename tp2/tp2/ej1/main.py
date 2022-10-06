import json

from tp2 import utils, optimizer
from tp2.ej1 import graph
from tp2.perceptron import Perceptron


def main(config_path=None, data_path=None):
    if config_path is None:
        config_path = "tp2/ej1/config.json"
    if data_path is None:
        data_path = "tp2/ej1/data.json"

    try:
        with open(config_path) as f:
            data = json.load(f)
            dataset = data["dataset"]
            learning = data["learning"]
            max_iter = data["max_iter"]
            error = data["error"]
            eta = data["eta"]
    except FileNotFoundError:
        print("Couldn't find config path")
        exit(1)

    try:
        with open(data_path) as f:
            data = json.load(f)[dataset]
    except FileNotFoundError:
        print("Couldn't find config path")
        exit(1)

    matr_dims = [1]
    perceptron = Perceptron(
        len(data[0]) - 1, matr_dims, None, utils.tanh_arr, utils.tanh_diff, eta, optimizer.adaptative_eta
    )

    perceptron.train(data, error, max_iter, learning)

    print("x: 1 ~ y: 1")
    print(perceptron.predict([1, 1, 1]))
    print("x: -1 ~ y: -1")
    print(perceptron.predict([1, -1, -1]))
    print("x: -1 ~ y: 1")
    print(perceptron.predict([1, -1, 1]))
    print("x: 1 ~ y: -1")
    print(perceptron.predict([1, 1, -1]))

    perceptron.save()


if __name__ == "__main__":
    main("config.json", "data.json")
