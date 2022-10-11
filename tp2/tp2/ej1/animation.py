import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from tp2.ej2.wrapper import Wrapper

matplotlib.use("TkAgg")

from matplotlib.animation import FuncAnimation, PillowWriter


def create_animation(data, historic, layer_historic, perceptron, learning, g_function):
    fig = plt.figure(figsize=(14, 9))
    ax = plt.axes()

    def animate(i):
        ax.clear()
        ax.set_ylim([-1.5, 1.5])
        ax.set_xlim([-1.5, 1.5])
        ax.set_title(
            "ej1 ~ activation: "
            + g_function.__name__
            + " ~ "
            + learning
            + "\n"
            + "eta: "
            + perceptron.eta.__str__()
            + "  optimizer: "
            + perceptron.optimizer.__name__
            + " ~ iter: " + str(i)
        )
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        ax.axhline(linewidth=0.4, color="black")
        ax.axvline(linewidth=0.4, color="black")

        res = np.squeeze(historic[i])
        for u in range(0, len(data)):
            color = "black"
            if res[u] > 0:
                color = "blue"
            elif res[u] < 0:
                color = "red"
            ax.plot(
                data[u][1],
                data[u][2],
                marker="o",
                markerfacecolor=color,
                markeredgecolor=color,
            )

        matr = layer_historic[i][0][0]
        x = np.linspace(-2, 2, 50)
        y = -(matr[1] * x + matr[0]) / matr[2]
        ax.plot(x, y, "gray")

    ani = FuncAnimation(
        fig, animate, frames=len(layer_historic), interval=400, repeat=False
    )
    plt.close()
    fps = round(len(layer_historic) / 4)
    ani.save(
        "anim-"
        + g_function.__name__.__str__()
        + "-"
        + learning
        + "-"
        + perceptron.optimizer.__name__
        + ".gif",
        fps=fps,
    )


if __name__ == "__main__":
    wrapper = Wrapper.load()
    if wrapper.layer_historic:
        create_animation(
            wrapper.data,
            wrapper.historic,
            wrapper.layer_historic,
            wrapper.perceptron,
            wrapper.learning,
            wrapper.g_function,
        )
