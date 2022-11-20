import time

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from tp2.ej3.c.wrapper import Wrapper
from tp2.loader import CSVLoader


def plot_probabilities(perceptron, data, exp, filename):
    fig = plt.figure(figsize=(14, 9))
    colors = [
        "red",
        "blue",
        "green",
        "yellow",
        "purple",
        "orange",
        "brown",
        "pink",
        "olive",
        "indigo",
    ]

    exp = np.squeeze(exp[:10, :])

    historic = []
    stds = []
    for i in range(0, 10):
        aux = []
        for j in range(0, 5):
            res = perceptron.predict(data[i + j * 10])
            # print(exp[k], np.argmax(res))
            aux.append(np.divide(np.add(res, 1), 2))
        # print(np.mean(aux, axis=0))
        historic.append(np.mean(aux, axis=0))
        stds.append(np.std(aux, axis=0))
        # k += 1

    width = 0.1
    x_axis = np.arange(len(exp))

    plt.grid(axis="y", c="lightgray", linewidth=0.5, linestyle="-")
    bars = []
    for j in range(0, 10):
        bars.append(
            plt.bar(
                x_axis + width * j,
                historic[j],
                width,
                yerr=stds[j],
                ecolor="black",
                capsize=3,
                color=colors[j],
            )
        )

    # plt.xlim([-0.5, 10.5])
    plt.ylim([-0.2, 1.2])
    plt.xlabel("Valor esperado")
    plt.title("Valor esperado vs porcentaje de certeza")
    plt.ylabel("Porcentaje de certeza")
    plt.xticks(x_axis + width * 4.5, map(str, exp))
    plt.legend(tuple(bars), tuple(list(map(str, exp))))
    plt.close()
    fig.savefig(filename)


if __name__ == "__main__":
    wrapper = Wrapper.load()
    data_column = ["x" + str(i) for i in range(1, 36)]
    expected_column = ["y"]
    csv = [
        "noisy_digitsformat-normal.csv",
        "noisy_digitsformat-normal-fuerte.csv",
        "noisy_digitsformat-normal-masfuerte.csv",
        "noisy_digitsformat-uniform.csv",
    ]
    name = ["n", "f", "mf", "u"]
    for i in range(0, len(csv)):
        data, exp = CSVLoader.load(csv[i], False, data_column, expected_column, True)
        plot_probabilities(
            wrapper.perceptron,
            data,
            exp,
            "probability-"
            + wrapper.perceptron.optimizer.__name__
            + "-"
            + name[i]
            + ".png",
        )
