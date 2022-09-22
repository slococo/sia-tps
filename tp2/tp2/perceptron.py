import math
import random

import numpy as np


class Perceptron:
    def __init__(self, matrix_arr, optimizer, g, g_diff, p, eta):
        self.matrix_arr = matrix_arr
        self.optimizer = optimizer
        self.g = g
        self.g_diff = g_diff
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
        aux = np.array(data)
        out = 0
        for layer in self.matrix_arr:
            # print(layer)
            h = np.array([layer @ aux.T])
            out = self.g(h)
        return out

    def batch(self, data, error_max, max_iter):
        min_error = math.inf
        n = 0
        d = np.zeros((len(self.matrix_arr), 3))

        while min_error > error_max and n < max_iter:
            error = 0
            for u in data:
                h, v = [], []
                h.append(np.array(u[:-1]))
                for layer in self.matrix_arr:
                    h.append(np.array([layer @ h[-1].T]))
                    v.append(self.g(h[-1:]))

                # print(self.eta * (np.subtract(u[-1:], v[-1])) * self.g_diff(h[-1]) * h[-2])
                d[0] = np.add(d[0], self.eta * (np.subtract(u[-1:], v[-1])) * self.g_diff(h[-1]) * h[-2])

                error += ((1 / 2) * (np.subtract(u[-1:], v[-1])) ** 2)[0]

                j = 1
                matr_aux = self.matrix_arr[::-1]
                for _ in matr_aux[:-1]:
                    d[j] = np.add(d[j], self.eta * (np.subtract(u[-1:] - v[-1])) * self.g_diff(h[-j]) * matr_aux[-j] * self.g_diff(h[-(j + 1)]) * u[-1:])
                    j += 1

            if error < min_error:
                min_error = error
                if min_error <= error_max:
                    break

            i = 0
            for layer in self.matrix_arr:
                # print(d[-(i + 1)])
                layer += (d[-(i + 1)] / len(data))

            n += 1

        print(n)
        print(min_error)

    def online(self, data, error_max, max_iter):
        min_error = math.inf
        d = np.zeros((len(self.matrix_arr), len(self.matrix_arr[0])))
        n = 0
        while min_error > error_max and n < max_iter:
            random.shuffle(data)
            for u in data:
                # print("data: ", u[:-1])
                n += 1
                h, v = [], []
                v.append(np.array(u[:-1]))
                for layer in self.matrix_arr:
                    h.append(np.array([layer @ v[-1].T]))
                    v.append(self.g(h[-1:]))

                d[0] = np.subtract(u[-1:], v[-1]) * self.g_diff(h[-1])
                j = 0
                matr_aux = self.matrix_arr[::-1]
                for layer in matr_aux:
                    if j != 0:
                        d[j] = self.g_diff(h[-(j + 1)]) * aux * d[j - 1]
                    aux = layer
                    # print("? ", v)
                    layer += self.eta * v[-(j + 2)] * d[j]
                    j += 1

                error = ((1 / 2) * (np.subtract(u[-1:], v[-1])) ** 2)[0][0]

                if error < min_error:
                    min_error = error

                if min_error <= error_max or n >= max_iter:
                    print(n)
                    print(min_error)
                    break
