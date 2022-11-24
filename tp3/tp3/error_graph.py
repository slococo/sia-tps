import time

import matplotlib
import numpy as np

matplotlib.use("TkAgg")

import matplotlib.pyplot as plt


class ErrorGraph:
    @classmethod
    def plot_error(cls, errors=None, means=None, stds=None):
        # fig = plt.figure(figsize=(14, 9))
        if errors is not None:
            cls.make_plt(errors=errors)
        else:
            cls.make_plt(means=means, stds=stds)
        # plt.close()
        # fig.savefig("error" + round(time.time()).__str__() + ".png")
        plt.show()

    @classmethod
    def make_plt(cls, errors=None, color="red", means=None, stds=None):
        if errors is not None:
            errors = np.atleast_2d(errors)
            means = np.mean(errors, axis=0)
            stds = np.std(errors, axis=0)

        if means is None or stds is None:
            raise "means is None or stds is None"

        plt.title("Error vs epoch")
        plt.xlabel("epoch")
        plt.ylabel("error")
        plt.plot(means, color=color)
        plt.fill_between(
            range(0, len(means)),
            np.subtract(means, stds),
            np.add(means, stds),
            alpha=0.1,
            color=color,
            label="_nolegend_",
        )

    @classmethod
    def bar_predict_graph(cls, errors, colors, legends):
        for i in range(0, len(errors)):
            mean = np.mean(errors[i])
            std = np.std(errors[i])
            plt.grid(axis="y", c="lightgray", linewidth=0.5, linestyle="-")
            plt.bar(
                legends[i], mean, yerr=std, ecolor="black", capsize=10, color=colors[i]
            )
