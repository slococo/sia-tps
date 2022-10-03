import copy
import math
import pickle
import random

import numpy as np


class Perceptron:
    def __init__(self, matrix_arr, optimizer, g, g_diff, p, eta):
        self.matrix_arr = matrix_arr
        self.optimizer = optimizer
        self.g = g
        self.g_diff = g_diff
        self.eta = eta

    def save(self, file_name=None):
        if file_name is None:
            file_name = "perceptron.obj"
        with open(file_name, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, file_name=None):
        if file_name is None:
            file_name = "perceptron.obj"
        with open(file_name, "rb") as f:
            return pickle.load(f)

    def train(self, data, error_max, max_iter, method, exp=None):
        match method:
            case "online":
                self.online(data, error_max, max_iter, exp)
            case "batch":
                self.batch(data, error_max, max_iter, exp)
            case _:
                raise RuntimeError("Unknown method " + method)

    def predict(self, data):
        out = np.array(data)
        for layer in self.matrix_arr:
            aux = np.atleast_1d([layer @ out.T])
            print(aux)
            out = self.g(aux)
        return out

    def batch(self, data, error_max, max_iter, exp=None):
        min_error = math.inf
        n = 0
        while min_error > error_max and n < max_iter:
            error = 0

            dw = []
            for i in self.matrix_arr[::-1]:
                dw.append(np.atleast_2d(np.zeros_like(i)))

            n += 1
            for u in data:
                h, v = [], []
                v.append(np.array(u[:-1]))
                for layer in self.matrix_arr:
                    h.append(np.atleast_1d(layer @ v[-1]))
                    v.append(np.atleast_1d(np.array(self.g(h[-1]))))

                res = v[-1]
                expected = u[-1]
                if exp:
                    res = exp(v[-1])

                    expected = np.zeros_like(res)
                    expected[round(u[-1:][0])] = 1

                d = np.atleast_2d(np.subtract(expected, res) * self.g_diff(h[-1]))
                dw[0] += self.eta * (d.T.dot(np.atleast_2d(v[-2])))

                j = 0
                for layer in self.matrix_arr[::-1]:
                    if j != 0:
                        d = np.atleast_2d(aux.T.dot(d.T) * (np.atleast_2d(self.g_diff(h[-(j + 1)])).T)).T
                        dw[j] += self.eta * d.dot((np.atleast_2d(v[-(j + 1)])).T)
                    aux = layer
                    j += 1

                error += np.average((1 / 2) * (np.subtract(expected, res)) ** 2)

            j = 0
            for layer in self.matrix_arr[::-1]:
                layer += dw[j] / len(data)
                j += 1

            if error < min_error:
                min_error = error

        print(n)
        print(min_error)
        print(self.matrix_arr)

    def online(self, data, error_max, max_iter, exp=None):
        min_error = math.inf
        d = np.zeros((len(self.matrix_arr), len(self.matrix_arr[0])))
        n = 0
        data_copy = copy.copy(data)
        while min_error > error_max and n < max_iter:
            random.shuffle(data_copy)
            n += 1
            for u in data:
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
                    layer += self.eta * v[-(j + 2)] * d[j]
                    j += 1

                error = 0
                for a in data_copy:
                    he, ve = [], []
                    ve.append(np.array(a[:-1]))
                    for layer in self.matrix_arr:
                        he.append(np.array([layer @ ve[-1].T]))
                        ve.append(self.g(he[-1:]))

                    error += (1 / 2) * (np.subtract(a[-1:], ve[-1])) ** 2

                if error < min_error:
                    min_error = error

                if min_error <= error_max or n >= max_iter:
                    print(n)
                    print(min_error)
                    print(self.matrix_arr)
                    break
