import json
import time

import numpy as np
from tp2.optimizer import *
from tp2.perceptron import Perceptron
import tp2.utils as utils


def run_all():
    """Function designed to test all possible combinations of the perceptron,
    including the different optimizers, the different activation functions,
    and the different datasets.
    """
    with open("data.json") as f:
        try:
            data = json.load(f)
        except FileNotFoundError:
            raise RuntimeError("Couldn't load data.json")

    optimizers = [gradient, momentum, momentum, rms_prop, adam]
    g_functions = [
        [utils.tanh_arr, utils.tanh_diff],
        [utils.identity, utils.ident_diff],
        [utils.logistic_arr, utils.logistic_diff],
        [np.sign, utils.ident_diff]
    ]
    train_methods = ["batch", "online"]
    keep_probs_arr = [
        [1, 1],
        [0.8, 0.5]
    ]

    eta_adapt = [None, adaptative_eta]
    for g_func_pair in g_functions:
        for opt in optimizers:
            for eta_ad in eta_adapt:
                for eta_init in np.linspace(10 ** -4, 10 ** -1, 4):
                    for tanh_beta in np.linspace(0.01, 2, 4):
                        for method in train_methods:
                            for keep_probs in keep_probs_arr:
                                times = []
                                for dataset in data:
                                    utils.set_b(tanh_beta)
                                    perceptron = Perceptron(
                                        len(data[dataset][0]) - 1,
                                        [1],
                                        opt,
                                        g_func_pair[0],
                                        g_func_pair[1],
                                        eta_init,
                                        eta_ad,
                                        input_keep_prob=keep_probs[0],
                                        hidden_keep_prob=keep_probs[1]
                                    )
                                    start_time = time.time()
                                    historic, errors, layer_historic = perceptron.train(
                                        data[dataset],
                                        1000,
                                        0.01,
                                        method,
                                    )
                                    times.append((time.time() - start_time) / 1000)
                                avg_time = sum(times) / len(times)
                                print("Average time: {:.8f}ms".format(avg_time))


if __name__ == "__main__":
    run_all()