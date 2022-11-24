import json

from tp3 import utils
from tp3.autoencoder import Autoencoder
from tp3.den_autoencoder import DenoisingAutoencoder
from tp3.noise import *
from tp3.optimizer import *
from tp3.perceptron import Perceptron


class Initializer:
    @classmethod
    def initialize(cls, path, matr_dims, data_size, type):
        try:
            with open(path) as f:
                data = json.load(f)
                dataset = data.get("dataset", None)
                learning = data["learning"]
                optimizer = data["optimizer"]
                g_function = data["g_function"]
                eta_adapt = data["eta_adapt"]
                beta = data["beta"]
                max_iter = data["max_iter"]
                error = data["error"]
                eta = data["eta"]
                if type == "dae":
                    noise_dist = data["noise_dist"]
        except FileNotFoundError:
            raise "Couldn't find config path"

        optimizer = globals()[optimizer]
        if eta_adapt is not None:
            eta_adapt = globals()[eta_adapt]
        match g_function:
            case "tanh":
                g_function = utils.tanh_arr
                g_diff = utils.tanh_diff
            case "identity":
                g_function = utils.identity
                g_diff = utils.ident_diff
            case "logistic":
                g_function = utils.logistic_arr
                g_diff = utils.logistic_diff
            case "relu":
                g_function = utils.relu
                g_diff = utils.relu_diff
            case _:
                g_function = utils.step
                g_diff = utils.ident_diff
        utils.set_b(beta)

        if type == "dae":
            noise_dist = globals()[noise_dist]

        match type:
            case "ae":
                perceptron = Autoencoder(
                    data_size + 1,
                    matr_dims,
                    optimizer,
                    g_function,
                    g_diff,
                    eta,
                    eta_adapt,
                )
            case "dae":
                perceptron = DenoisingAutoencoder(
                    data_size + 1,
                    matr_dims,
                    optimizer,
                    g_function,
                    g_diff,
                    eta,
                    eta_adapt,
                    noise_dist,
                )
            case _:
                raise "your hands"

        return perceptron, max_iter, error, learning, eta, dataset
