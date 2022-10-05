import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

from tp2 import utils

matplotlib.use("TkAgg")


def plot(df=None):
    if df is None:
        df = pd.read_csv("dataset.csv")

    fig = plt.figure(figsize=(14, 9))
    ax = plt.axes(projection="3d")
    ax.set_ylim([-2, 8])
    ax.set_xlim([-2, 8])
    ax.set_zlim([-2, 8])
    X = df["x1"]
    Y = df["x2"]
    Z = df["x3"]
    res = utils.normalise(df["y"], -1, 1)
    for i in range(0, len(X)):
        color = 0xFFFFFF * ((res[i] + 1) / 2)
        ax.scatter(X[i], Y[i], Z[i], color="#{:06x}".format(round(color)))

    plt.show()


if __name__ == "__main__":
    plot()
