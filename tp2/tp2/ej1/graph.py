import matplotlib

from tp2.ej1.wrapper import Wrapper

matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm


def plot(perceptron=None):
    if perceptron is None:
        perceptron = Wrapper.load().perceptron

    x = np.outer(np.linspace(-3, 3, 32), np.ones(32))
    y = x.copy().T
    coefs = perceptron.matrix_arr[0][0]
    z = np.tanh(coefs[0] + x * coefs[1] + y * coefs[2])
    plt.figure(figsize=(14, 9))
    ax = plt.axes(projection="3d")
    ax.plot_surface(
        x, y, z, cmap=cm.coolwarm, linewidth=0, antialiased=False, alpha=0.85
    )
    x = np.array([1, 1, -1, -1])
    y = np.array([1, -1, -1, 1])
    ax.scatter(x, y, color="black", depthshade=False, alpha=1)
    plt.show()


if __name__ == "__main__":
    plot()
