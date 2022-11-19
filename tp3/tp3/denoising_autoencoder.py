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
#        aux_dims = dims[:-1]
#        n = len(aux_dims) // 2
#        if not np.all(aux_dims[:n] == aux_dims[-1:-n - 1:-1]):
#            raise "your hands"
            # raise "Dims array does not correspond to symmetrical Autoencoder architecture"
        self.noise_distribution = noise_distribution
        Autoencoder.__init__(self, data_dim, dims, optimizer, g, g_diff, eta, eta_adapt, input_keep_prob, hidden_keep_prob)
#        self.z = np.ceil(len(dims) / 2)

    def train(self, data, expected, error_max, max_iter, method, exp=None):
        return Autoencoder.train(self, self.noise_distribution(data), data, error_max, max_iter, method, exp)
