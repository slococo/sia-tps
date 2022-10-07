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

    def animate_batch(i):
        ax.clear()
        ax.set_ylim([-1.5, 1.5])
        ax.set_xlim([-1.5, 1.5])
        ax.set_title("ej1 ~ activation: " + g_function + " ~ " + learning +  "\n" + "eta: " + perceptron.eta.__str__() + "  optimizer: " + perceptron.optimizer.__name__)
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        ax.axhline(linewidth=0.4, color='black')
        ax.axvline(linewidth=0.4, color='black')

        res = historic[i]
        for u in range(0, len(data)):
            color = 'black'
            if res[u] > 0:
                color = 'blue'
            elif res[u] < 0:
                color = 'red'
            ax.plot(data[u][1], data[u][2], marker='o', markerfacecolor=color, markeredgecolor=color)

        matr = layer_historic[i][0][0]
        x = np.linspace(-2, 2, 50)
        y = - (matr[1] * x + matr[0]) / matr[2]
        ax.plot(x, y, 'gray')

    def animate_online(i):
        ax.clear()
        ax.set_ylim([-1.5, 1.5])
        ax.set_xlim([-1.5, 1.5])
        ax.set_title("ej1 ~ activation: " + g_function + " ~ " + learning +  "\n" + "eta: " + perceptron.eta.__str__() + "  optimizer: " + perceptron.optimizer.__name__)
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        ax.axhline(linewidth=0.4, color='black')
        ax.axvline(linewidth=0.4, color='black')

        for u in range(0, len(historic[i])):
            res = historic[i][u][-1]
            color = 'black'
            if res > 0:
                color = 'blue'
            elif res < 0:
                color = 'red'
            ax.plot(historic[i][u][1], historic[i][u][2], marker='o', markerfacecolor=color, markeredgecolor=color)

        matr = layer_historic[i][0][0]
        x = np.linspace(-2, 2, 50)
        y = - (matr[1] * x + matr[0]) / matr[2]
        ax.plot(x, y, 'gray')

    if learning == "batch":
        animate = animate_batch
    elif learning == "online":
        animate = animate_online
    else:
        raise "Invalid learning technique"
    ani = FuncAnimation(fig, animate, frames=len(layer_historic), interval=1000, repeat=False)
    plt.close()
    fps = round(len(layer_historic) / 4)
    ani.save("anim-" + g_function + "-" + learning + "-" + perceptron.optimizer.__name__ + ".gif", fps=fps)


if __name__ == "__main__":
    wrapper = Wrapper.load()
    if wrapper.layer_historic:
        create_animation(wrapper.data, wrapper.historic, wrapper.layer_historic, wrapper.perceptron, wrapper.learning, wrapper.g_function)
