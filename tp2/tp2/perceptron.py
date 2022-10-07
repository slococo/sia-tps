import copy
import math
import pickle
import random

import numpy as np
from tp2.optimizer import gradient


class Perceptron:
    def __init__(self, data_dim, dims, optimizer, g, g_diff, eta, eta_adapt, input_keep_prob=1, hidden_keep_prob=1):
        self.matrix_arr = []
        aux = data_dim
        for dim in dims:
            self.matrix_arr.append(np.random.rand(dim, aux))
            aux = dim
        self.optimizer = optimizer
        self.g = g
        self.g_diff = g_diff
        self.eta = eta
        self.eta_adapt = eta_adapt
        self.input_keep_prob = input_keep_prob
        self.hidden_keep_prob = hidden_keep_prob

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
                return self.online(data, error_max, max_iter, exp)
            case "batch":
                return self.batch(data, error_max, max_iter, exp)
            case _:
                raise RuntimeError("Unknown method " + method)

    def predict(self, data):
        out = np.array(data)
        for layer in self.matrix_arr:
            aux = np.atleast_1d(layer @ np.atleast_2d(out).T)
            out = self.g(aux)
        return out

    def batch(self, data, error_max, max_iter, exp=None):
        errors, historic, layer_historic = [], [], []
        error = math.inf
        n = 0
        while error > error_max and n < max_iter:
            error = 0

            dw = []
            for i in self.matrix_arr[::-1]:
                dw.append(np.atleast_2d(np.zeros_like(i)))

            n += 1
            if len(layer_historic) <= n - 1:
                layer_historic.append([])

            for layer in self.matrix_arr:
                layer_historic[n - 1].append(layer.copy())

            for u in data:
                h, v = [], []
                v.append(np.array(u[:-1]))
                for layer in self.matrix_arr:
                    if layer is self.matrix_arr[0]:
                        prob = self.input_keep_prob
                    else:
                        prob = self.hidden_keep_prob
                    layer_cpy = layer.copy()
                    for i in range(len(layer)):
                        if random.uniform(0, 1) > prob:
                            print("Node dropped")
                            layer_cpy[i] = np.zeros(len(layer[i]))
                    h.append(np.atleast_1d(layer_cpy @ v[-1]))
                    v.append(np.atleast_1d(np.array(self.g(h[-1]))))

                if len(historic) <= n - 1:
                    historic.append([])
                historic[n - 1].append(v[-1])

                res = v[-1]
                expected = u[-1]
                if exp:
                    expected = exp(v[-1], expected)

                if self.eta_adapt:
                    self.eta = self.eta_adapt(
                        np.average(np.subtract(expected, res)), self.eta
                    )
                d = np.atleast_2d(np.subtract(expected, res) * self.g_diff(h[-1]))
                dw[0] += self.optimizer(d.T.dot(np.atleast_2d(v[-2])), self.eta, 0)

                j = 0
                for layer in self.matrix_arr[::-1]:
                    if j != 0:
                        d = np.atleast_2d(
                            aux.T.dot(d.T) * np.atleast_2d(self.g_diff(h[-(j + 1)])).T
                        ).T
                        dw[j] += self.optimizer(
                            d.T.dot((np.atleast_2d(v[-(j + 2)]))), self.eta, j
                        )
                    aux = layer
                    j += 1

                error += np.average((np.subtract(expected, res) / 2) ** 2)

            error = error / len(data)
            errors.append(error)

            if error <= error_max or n >= max_iter:
                break

            j = 0
            for layer in self.matrix_arr[::-1]:
                aux = layer.copy()
                layer += dw[j]
                j += 1

        for layer_idx in range(len(self.matrix_arr)):
            if layer_idx == 0:
                prob = self.input_keep_prob
            else:
                prob = self.hidden_keep_prob
            self.matrix_arr[layer_idx] = np.array([prob * i for i in self.matrix_arr[layer_idx]])

        print("Times: ", n)
        print("Error: ", error)
        return historic, errors, layer_historic

    def online(self, data, error_max, max_iter, exp=None):
        errors, historic, layer_historic = [], [], []
        error = math.inf
        n = 0
        k = 0
        while error > error_max and n < max_iter:
            error = 0

            dw = []
            for i in self.matrix_arr[::-1]:
                dw.append(np.atleast_2d(np.zeros_like(i)))

            n += 1

            data_copy = data.copy()
            random.shuffle(data_copy)
            for u in data_copy:
                k += 1
                h, v = [], []
                v.append(np.array(u[:-1]))
                for layer in self.matrix_arr:
                    h.append(np.atleast_1d(layer @ v[-1]))
                    v.append(np.atleast_1d(np.array(self.g(h[-1]))))

                res = v[-1]
                expected = u[-1]
                if exp:
                    expected = exp(v[-1], expected)

                if self.eta_adapt:
                    self.eta = self.eta_adapt(
                        np.average(np.subtract(expected, res)), self.eta
                    )
                d = np.atleast_2d(np.subtract(expected, res) * self.g_diff(h[-1]))
                dw[0] += self.optimizer(d.T.dot(np.atleast_2d(v[-2])), self.eta, 0)

                j = 0
                if len(layer_historic) <= k - 1:
                    layer_historic.append([])
                for layer in self.matrix_arr[::-1]:
                    if j != 0:
                        d = np.atleast_2d(
                            aux.T.dot(d.T) * np.atleast_2d(self.g_diff(h[-(j + 1)])).T
                        ).T
                        dw[j] = self.optimizer(
                            d.T.dot((np.atleast_2d(v[-(j + 2)]))), self.eta, j
                        )
                    layer += dw[j]
                    aux = layer
                    layer_historic[k - 1].append(layer.copy())
                    j += 1

                for a in data_copy:
                    he, ve = [], []
                    ve.append(np.array(a[:-1]))
                    for layer in self.matrix_arr:
                        he.append(np.atleast_1d(layer @ ve[-1]))
                        ve.append(np.atleast_1d(np.array(self.g(he[-1]))))
                    res = ve[-1]
                    expected = a[-1]

                    if len(historic) <= k - 1:
                        historic.append([])
                    aux = a[:-1].copy()
                    aux.append(ve[-1])
                    historic[k - 1].append(aux)

                    if exp:
                        expected = exp(ve[-1], expected)
                    error += np.average((np.subtract(expected, res) / 2) ** 2)

                error = error / len(data)
                errors.append(error)

                if error <= error_max or n >= max_iter:
                    break

        print("Times: ", n)
        print("Error: ", error)
        return historic, errors, layer_historic
