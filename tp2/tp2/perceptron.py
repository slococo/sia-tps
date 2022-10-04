import copy
import math
import pickle
import random

import numpy as np
from matplotlib import pyplot as plt, cm


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
            aux = np.atleast_1d(layer @ np.atleast_2d(out).T)
            out = self.g(aux)
        return out

    def batch(self, data, error_max, max_iter, exp=None):
        errors = []
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
                    expected = np.full_like(res, fill_value=-1)
                    expected[round(u[-1:][0])] = 1

                d = np.atleast_2d(np.subtract(expected, res) * self.g_diff(h[-1]))
                dw[0] += self.eta * (d.T.dot(np.atleast_2d(v[-2])))

                j = 0
                for layer in self.matrix_arr[::-1]:
                    if j != 0:
                        d = np.atleast_2d(aux.T.dot(d.T) * np.atleast_2d(self.g_diff(h[-(j + 1)])).T).T
                        dw[j] += self.eta * d.T.dot((np.atleast_2d(v[-(j + 2)])))
                    aux = layer
                    j += 1

                error += np.average((np.subtract(expected, res) / 2) ** 2)

            j = 0
            for layer in self.matrix_arr[::-1]:
                layer += dw[j]
                j += 1

            error = error / len(data)
            errors.append(error)

            if error < min_error:
                min_error = error

        fig = plt.figure(figsize=(14, 9))
        plt.plot(range(1, n + 1), errors)
        plt.show()
        print(n)
        print(min_error)
        print(self.matrix_arr)

    def online(self, data, error_max, max_iter, exp=None):
        min_error = math.inf
        n = 0
        data_copy = copy.copy(data)
        while min_error > error_max and n < max_iter:
            random.shuffle(data_copy)
            dw = []
            for i in self.matrix_arr[::-1]:
                dw.append(np.atleast_2d(np.zeros_like(i)))

            n += 1
            for u in data_copy:
                h, v = [], []
                v.append(np.array(u[:-1]))
                for layer in self.matrix_arr:
                    h.append(np.atleast_1d(layer @ v[-1]))
                    v.append(np.atleast_1d(np.array(self.g(h[-1]))))

                res = v[-1]
                expected = u[-1]
                if exp:
                    res = exp(v[-1])
                    expected = np.full_like(res, fill_value=-1)
                    expected[round(u[-1:][0])] = 1

                d = np.atleast_2d(np.subtract(expected, res) * self.g_diff(h[-1]))
                dw[0] = self.eta * (d.T.dot(np.atleast_2d(v[-2])))

                j = 0
                for layer in self.matrix_arr[::-1]:
                    if j != 0:
                        d = np.atleast_2d(aux.T.dot(d.T) * np.atleast_2d(self.g_diff(h[-(j + 1)])).T).T
                        dw[j] = self.eta * d.T.dot((np.atleast_2d(v[-(j + 2)])))
                    aux = layer
                    layer += dw[j]
                    j += 1

                error = 0
                for a in data_copy:
                    he, ve = [], []
                    ve.append(np.array(a[:-1]))
                    for layer in self.matrix_arr:
                        he.append(np.atleast_1d(layer @ ve[-1]))
                        ve.append(np.atleast_1d(np.array(self.g(he[-1]))))

                    error += np.average((1 / 2) * (np.subtract(a[-1:], ve[-1])) ** 2)

                if error < min_error:
                    min_error = error

                if min_error <= error_max or n >= max_iter:
                    print(n)
                    print(min_error)
                    print(self.matrix_arr)
                    break
