import json
import tp2.utils as utils
from tp2.optimizer import *
from tp2.run_all_generic import run_all_generic


def run_all():
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
        [np.sign, utils.ident_diff],
    ]
    train_methods = ["batch", "online"]
    keep_probs_arr = [[1, 1], [0.8, 0.5]]
    eta_adapt = [None, adaptative_eta]

    run_all_generic(
        optimizers=optimizers,
        g_functions=g_functions,
        train_methods=train_methods,
        keep_probs_arr=keep_probs_arr,
        eta_adapt=eta_adapt,
        eta_init_arr=np.linspace(10**-4, 10**-1, 4),
        tanh_beta_arr=np.linspace(0.01, 2, 4),
        data_dim=len(data["and"][0]) - 1,
        neuron_dim=[1],
        data=data,
    )


if __name__ == "__main__":
    run_all()
