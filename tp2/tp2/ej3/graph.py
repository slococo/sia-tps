import matplotlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("TkAgg")


def plot(df=None):
    if df is None:
        df = pd.read_csv("digits.csv", sep=' ', header=None)

    color = np.array([255, 255, 255])
    df = df.to_numpy()
    dfi = np.subtract(1, df)

    data = None
    for j in range(0, dfi.shape[0]):
        if j % 7 == 0:
            if not data:
                data = []
            else:
                plt.imshow(np.array([[255, 255, 255, 255, 255]]), interpolation='nearest')
                plt.imshow(data, interpolation='nearest')
                plt.show()
                data = []

        aux = []
        for i in range(0, 5):
            aux.append(color * (dfi[j][i]))
        data.append(aux)

    plt.imshow(np.array([[255, 255, 255, 255, 255]]), interpolation='nearest')
    plt.imshow(data, interpolation='nearest')
    plt.show()


if __name__ == "__main__":
    plot()
