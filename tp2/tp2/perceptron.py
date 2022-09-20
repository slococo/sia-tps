import math
import random

import numpy as np

class Perceptron:
    def __init__(self, matrix_arr, optimizer, g, g_diff, p, eta):
        self.matrix_arr = matrix_arr
        self.optimizer = optimizer
        self.g = g
        self.g_diff = g_diff
        self.weights = np.zeros(p + 1)
        self.eta = eta

    def train(self, data, error_max, max_iter, method):
        match method:
            case "online":
                self.online(data, error_max, max_iter)
            case "batch":
                self.batch(data, error_max, max_iter)
            case _:
                raise RuntimeError("Unknown method " + method)

    def predict(self, data):
        pass

    def batch(self, data, error_max, max_iter):
        min_error = math.inf
        n = 0
        dw = 0
        while min_error > error_max or n < max_iter:
            for u in data:
                aux, out = np.array(u[:-1])
                for layer in self.matrix_arr:
                    aux = layer @ out.T
                    out = self.g(aux)

                dw += self.eta * (u[-1:] - out) * self.g_diff(aux) * u[-1:]
                error = 1 / 2 * (u[-1:] - out) ** 2
                if error < min_error:
                    min_error = error
            # w += dw
            n += 1

    def online(self, data, error_max, max_iter):
        error = math.inf
        n = 0
        random.shuffle(data)
        while error > error_max or n < max_iter:
            for u in data:
                aux, out = np.array(u[:-1])
                for layer in self.matrix_arr:
                    aux = layer @ out.T
                    out = self.g(aux)

                dw += self.eta * (u[-1:] - out) * self.g_diff(aux) * u[-1:]
                error = 1 / 2 * (u[-1:] - out) ** 2
                if error < min_error:
                    min_error = error
                # w += dw
            n += 1