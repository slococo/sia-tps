import csv
import math

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("TkAgg")

import matplotlib.pyplot as plt


def plot(df=None):
    if df is None:
        df = pd.read_csv("digits.csv", sep=" ", header=None)

    uniform = False
    normal = True

    color = np.array([1, 1, 1])
    df = df.to_numpy()
    dfi = np.subtract(1, df)

    noisy_dataset = []
    data = None

    for f in range(0, dfi.shape[0] * 5):
        j = f % dfi.shape[0]
        if j % 7 == 0:
            noisy_dataset.append([])
            if data is None:
                data = np.zeros((7, 5, 3))
            else:
                plt.imshow(data, interpolation="nearest")
                # plt.show()
                data = np.zeros((7, 5, 3))
        aux = np.zeros((5, 3))
        for i in range(0, 5):
            val = dfi[j][i]
            if uniform:
                if np.random.uniform(0, 1) <= 0.05:
                    val = 1 - dfi[j][i]
            elif normal:
                val = min(max(np.random.normal(loc=dfi[j][i], scale=0.5), 0), 1)
            noisy_dataset[math.trunc(f / 7)].append(val)
            aux[i] = color * val
        data[j % 7] = aux
    plt.imshow(data, interpolation="nearest")
    # plt.show()

    q = 0
    with open("noisy_digitsformat-normal-masfuerte.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile, delimiter=",")
        for row in noisy_dataset:
            row.append(q % 10)
            q += 1
            writer.writerow(row)


if __name__ == "__main__":
    plot()
