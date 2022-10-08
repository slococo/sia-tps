import json
import time
from typing import List


from numpy import ndarray

import tp2.utils as utils
from tp2.optimizer import *
from tp2.perceptron import Perceptron


def run_all_generic(
    optimizers: List,
    g_functions: List,
    train_methods: List,
    keep_probs_arr: List,
    eta_adapt: List,
    eta_init_arr: ndarray,
    tanh_beta_arr: ndarray,
    data_dim: int,
    neuron_dim: List,
    data: dict,
):
    """Function designed to test all possible combinations of the perceptron,
    including the different optimizers, the different activation functions,
    and the different datasets.
    """
    for g_func_pair in g_functions:
        for opt in optimizers:
            for eta_ad in eta_adapt:
                for eta_init in eta_init_arr:
                    for tanh_beta in tanh_beta_arr:
                        for method in train_methods:
                            for keep_probs in keep_probs_arr:
                                print("Starting new run")
                                print("Optimizer: {}".format(opt.__name__))
                                print("G function: {}".format(g_func_pair[0].__name__))
                                print(
                                    "Eta adapt: {}".format(
                                        "True" if eta_init is not None else "False"
                                    )
                                )
                                print("Eta init: {}".format(eta_init))
                                print("Tanh beta: {}".format(tanh_beta))
                                print("Train method: {}".format(method))
                                print("Keep probs: {}".format(keep_probs))
                                times = []
                                errors_arr = []
                                layers = []
                                for dataset in data:
                                    utils.set_b(tanh_beta)
                                    perceptron = Perceptron(
                                        data_dim,
                                        neuron_dim,
                                        opt,
                                        g_func_pair[0],
                                        g_func_pair[1],
                                        eta_init,
                                        eta_ad,
                                        input_keep_prob=keep_probs[0],
                                        hidden_keep_prob=keep_probs[1],
                                    )
                                    start_time = time.time()
                                    historic, errors, layer_historic = perceptron.train(
                                        data[dataset],
                                        1000,
                                        0.01,
                                        method,
                                    )
                                    times.append((time.time() - start_time) / 1000)
                                    errors_arr.append(errors)
                                    layers.append(layer_historic)
                                avg_time = np.average(times)
                                avg_error = np.average(errors_arr)
                                avg_layers = np.average(layers)
                                print("End of run")
                                print("Average time: {:.8f}ms".format(avg_time))
                                print("Average error: {:.8f}".format(avg_error))
                                print("Average layers: {:.8f}".format(avg_layers))
                                print(
                                    "-----------------------------------------------------------------"
                                )


if __name__ == "__main__":
    run_all()
