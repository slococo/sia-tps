import matplotlib

matplotlib.use("TkAgg")
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tp2 import utils


def read_and_plot(df=None):
    if df is None:
        df = pd.read_csv("dataset.csv")
    columns = ["x1", "x2", "x3"]
    data = df[columns].to_numpy()
    res, _, _ = utils.normalise(df["y"].to_numpy(), -1, 1)

    fig = plt.figure(figsize=(14, 9))
    ax = plt.axes(projection="3d")
    plot(data, res, ax)
    plt.close()
    fig.savefig("colors-" + round(time.time()).__str__() + ".png", dpi=fig.dpi)


def plot(data, res, ax):
    ax.set_ylim([-2, 8])
    ax.set_xlim([-2, 8])
    ax.set_zlim([-2, 8])
    for i in range(0, len(data)):
        color_i = float_rgb(res[i], -1, 1)
        val = "#{:02x}{:02x}{:02x}".format(
            round(color_i[0]), round(color_i[1]), round(color_i[2])
        )
        ax.scatter(data[i][0], data[i][1], data[i][2], color=val)


def float_rgb(mag, cmin, cmax):
    try:
        x = float(mag - cmin) / (cmax - cmin)
    except ZeroDivisionError:
        x = 0.5
    blue = min((max((4 * (0.75 - x), 0.0)), 1.0))
    red = min((max((4 * (x - 0.25), 0.0)), 1.0))
    green = min((max((4 * np.fabs(x - 0.5) - 1.0, 0.0)), 1.0))
    return np.multiply([red, green, blue], 255)


if __name__ == "__main__":
    read_and_plot()
