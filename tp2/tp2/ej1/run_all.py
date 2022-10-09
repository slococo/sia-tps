import json
import time

import matplotlib
import numpy as np

matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

import tp2.utils as utils
from tp2.optimizer import *
from tp2.run_all_generic import run_all_generic


def create_graph(errors, name):
    fig = plt.figure(figsize=(14, 9))
    plt.rcParams.update({'font.size': 12})
    plt.title(name)
    plt.plot(range(1, len(errors) + 1), errors)
    plt.xlabel("Epoch")
    plt.ylabel("Error")
    plt.ylim(0, 1)
    fig.savefig("fig-" + round(time.time()).__str__(), dpi=fig.dpi)
    plt.close()


def run_all():
    try:
        with open("data.json") as f:
            data = json.load(f)
    except FileNotFoundError:
        raise RuntimeError("Couldn't load data.json")

    optimizers = [gradient, momentum]
    g_functions = [
        [utils.tanh_arr, utils.tanh_diff],
        [utils.identity, utils.ident_diff],
        [utils.logistic_arr, utils.logistic_diff],
        [np.sign, utils.ident_diff],
    ]
    train_methods = ["batch", "online"]
    keep_probs_arr = [[1, 1]]
    eta_adapt = [None, adaptative_eta]

    run_all_generic(
        optimizers=optimizers,
        g_functions=g_functions,
        train_methods=train_methods,
        keep_probs_arr=keep_probs_arr,
        eta_adapt=eta_adapt,
        eta_init_arr=np.linspace(1e-4, 1e-1, 4),
        tanh_beta_arr=np.linspace(0.01, 2, 4),
        data_dim=len(data["and"][0]) - 1,
        neuron_dim=[1],
        data=data,
        max_iter=2000,
        max_error=0,
        graph=create_graph,
    )


if __name__ == "__main__":
    run_all()
