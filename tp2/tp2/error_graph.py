import matplotlib
import numpy as np

matplotlib.use("TkAgg")

import matplotlib.pyplot as plt


class ErrorGraph:
    @classmethod
    def plot_error(cls, errors):
        cls.make_plt(errors, 'red')
        plt.show()

    @classmethod
    def make_plt(cls, errors, color):
        errors = np.atleast_2d(errors)

        print(errors)
        means = np.mean(errors, axis=0)
        stds = np.std(errors, axis=0)
        plt.title("Error vs epoch")
        plt.xlabel("epoch")
        plt.ylabel("error")
        plt.plot(means, color=color)
        plt.fill_between(range(0, len(means)), np.subtract(means, stds), np.add(means, stds), alpha=0.15, color=color, label="_nolegend_")
