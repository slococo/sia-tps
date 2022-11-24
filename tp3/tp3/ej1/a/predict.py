import time

import numpy as np

from tp3 import utils
from tp3.ej1.grapher import Grapher
from tp3.initializer import Initializer
from tp3.loader import CSVLoader
from tp3.perceptron import Perceptron
from tp3.tester import Tester


def main(config_path=None, data_path=None):
    if data_path is None:
        data_path = "tp3/ej1/fonts.csv"

    data_column = ["x" + str(i) for i in range(1, 36)]
    data, exp = CSVLoader.load(data_path, False, data_column, None, False)

    perceptron = Perceptron.load()

    for i in range(0, len(data)):
        Grapher.graph_in_out(data[i], perceptron.predict(data[i]))


if __name__ == "__main__":
    main("config.json", "../fonts.csv")