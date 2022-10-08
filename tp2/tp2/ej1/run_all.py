import json
import time
import tp2.utils as utils
from tp2.optimizer import *
from tp2.perceptron import Perceptron


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
                                print("Starting new run")
                                print("Optimizer: {}".format(opt.__name__))
                                print("G function: {}".format(g_func_pair[0].__name__))
                                print("Eta adapt: " + "True" if eta_ad is not None else "False")
                                print("Eta init: {}".format(eta_init))
                                print("Tanh beta: {}".format(tanh_beta))
                                print("Train method: {}".format(method))
                                print("Keep probs: {}".format(keep_probs))
                                times = []
                                errors = []
                                layers = []
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
                                avg_error = sum(errors) / len(errors)
                                avg_layers = sum(layers) / len(layers)
                                print("End of run")
                                print("Average time: {:.8f}ms".format(avg_time))
                                print("Average error: {:.8f}".format(avg_error))
                                print("Average layers: {:.8f}".format(avg_layers))
                                print("-----------------------------------------------------------------")


if __name__ == "__main__":
    run_all()
