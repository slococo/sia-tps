import numpy as np
from tp3.perceptron import Perceptron


class Autoencoder(Perceptron):
    def __init__(
            self,
            data_dim,
            dims,
            optimizer,
            g,
            g_diff,
            eta,
            eta_adapt,
            input_keep_prob=1,
            hidden_keep_prob=1,
    ):
        aux_dims = dims[:-1]
        n = len(aux_dims) // 2
        if not np.all(aux_dims[:n] == aux_dims[-1:-n - 1:-1]):
            raise "your hands"
            # raise "Dims array does not correspond to symmetrical Autoencoder architecture"
        Perceptron.__init__(self, data_dim, dims, optimizer, g, g_diff, eta, eta_adapt, input_keep_prob, hidden_keep_prob)
        self.z = np.ceil(len(dims) / 2)

    def get_latent_layer(self, data):
        out = np.array(data)
        for layer in self.matrix_arr[:self.z]:
            aux = np.squeeze(np.atleast_1d(layer @ np.atleast_2d(out).T))
            out = self.g(aux)
        return out

