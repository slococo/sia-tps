import numpy as np

from tp3.autoencoder import Autoencoder


class DenoisingAutoencoder(Autoencoder):
    def __init__(
        self,
        data_dim,
        dims,
        optimizer,
        g,
        g_diff,
        eta,
        eta_adapt,
        noise_distribution,
        input_keep_prob=1,
        hidden_keep_prob=1,
    ):
        self.noise_distribution = noise_distribution
        Autoencoder.__init__(
            self,
            data_dim,
            dims,
            optimizer,
            g,
            g_diff,
            eta,
            eta_adapt,
            input_keep_prob,
            hidden_keep_prob,
        )

    def train(self, data, expected, error_max, max_iter, method, exp=None):
        return Autoencoder.train(
            self, self.noise_distribution(0, 0.1, data), data[:, 1:], error_max, max_iter, method, exp
        )
