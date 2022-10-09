import math
import matplotlib
import numpy as np
import pandas as pd
import csv
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


def plot_error(errors):
    means = np.mean(errors, axis=0)
    stds = np.std(errors, axis=0)
    plt.plot(means, color='red')
    plt.fill_between(range(0, len(means)), np.subtract(means, stds), np.add(means, stds), alpha=0.15, color='red')
    print(stds)
    plt.show()


if __name__ == "__main__":
    plot_error([[1, 3, 2], [2, 3, 6], [4, 5, 2]])
