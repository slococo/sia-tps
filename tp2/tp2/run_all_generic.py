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
    max_error: float = 0.01,
    max_iter: int = 1000,
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
                                print(
                                    "-----------------------------------------------------------------"
                                )
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
                                train_errors_arr = []
                                predict_errors_arr = []
                                for dataset in data:
                                    print("")
                                    print("Dataset: {}".format(dataset))
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
                                        error_max=max_error,
                                        max_iter=max_iter,
                                        method=method,
                                    )
                                    times.append((time.time() - start_time) / 1000)
                                    train_errors_arr.append(errors[-1])

                                    for u in data[dataset]:
                                        predict_errors_arr.append(
                                            np.abs(
                                                np.subtract(
                                                    perceptron.predict(u[:-1]), u[-1]
                                                )
                                            )
                                        )
                                    print("")
                                avg_time = np.average(times)
                                avg_error = np.average(train_errors_arr)
                                avg_predict_error = np.average(predict_errors_arr)
                                print("End of run")
                                print("Average time: {:.8f}ms".format(avg_time))
                                print("Average train error: {:.8f}".format(avg_error))
                                print(
                                    "Average predict error: {:.8f}".format(
                                        avg_predict_error
                                    )
                                )
