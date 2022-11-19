import math
import pickle
import random

import numpy as np

from tp3 import utils


class Perceptron:
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
        self.dropout = False
        if self.input_keep_prob != 1 or self.hidden_keep_prob != 1:
            self.dropout = True

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

    def train(self, data, expected, error_max, max_iter, method, exp=None):
        match method:
            case "online":
                return self.online(data, expected, error_max, max_iter, exp)
            case "batch":
                return self.batch(data, expected, error_max, max_iter, exp)
            case _:
                raise RuntimeError("Unknown method " + method)

    def predict(self, data):
        _, v = self.calc_out(data, False)
        return v[-1]

    def predict_old(self, data):
        out = np.array(data)
        for layer in self.matrix_arr:
            aux = np.squeeze(np.atleast_1d(layer @ np.atleast_2d(out).T))
            out = self.g(aux)
        return out

    def batch(self, data, expected, error_max, max_iter, exp=None):
        errors, historic, layer_historic = [], [], []
        error, n = math.inf, 0
        while error > error_max and n < max_iter:
            error, dw = self.initialize_values(n, layer_historic)
            n += 1

            for i in range(0, len(data)):
                u = data[i]
                h, v = self.calc_out(u, True)

                if len(historic) <= n - 1:
                    historic.append([])
                historic[n - 1].append(v[-1])

                res = v[-1]
                expec = expected[i]
                if exp:
                    expec = exp(v[-1], expected[i])

                if self.eta_adapt:
                    self.eta = self.eta_adapt(
                        np.average(np.subtract(expec, res)), self.eta
                    )
                d = self.calc_d(res=res, exp=expec, h=h[-1])
                dw[0] += self.calc_dw(d, v[-2], 0)

                j = 0
                for layer in self.matrix_arr[::-1]:
                    if j != 0:
                        d = self.calc_d(d=d, w=aux, h=h[-(j + 1)])
                        dw[j] += self.calc_dw(d, v[-(j + 2)], j)
                    aux = layer
                    j += 1

                error += np.average((np.subtract(expec, res) / 2) ** 2)

            error = error / len(data)
            errors.append(error)

            if error <= error_max or n >= max_iter:
                break

            self.update_weights(dw)

        if self.dropout:
            self.scale_weights()

        return historic, errors, layer_historic, n

    def online(self, data, expected, error_max, max_iter, exp=None):
        errors, historic, layer_historic = [], [], []
        error, n, k = math.inf, 0, 0
        while error > error_max and n < max_iter:
            error, dw = self.initialize_values(n)
            n += 1

            temp = list(zip(data, expected))
            random.shuffle(temp)
            data_copy, expected_copy = zip(*temp)

            errors_aux = []
            for i in range(0, len(data_copy)):
                k += 1
                u = data_copy[i]
                h, v = self.calc_out(u, True)

                res = v[-1]
                expec = expected_copy[i]
                if exp:
                    expec = exp(v[-1], expected_copy[i])

                if self.eta_adapt:
                    self.eta = self.eta_adapt(
                        np.average(np.subtract(expec, res)), self.eta
                    )
                d = self.calc_d_log(res=res, exp=expec, h=h[-1])
                dw[0] = self.calc_dw(d, v[-2], 0)

                j = 0
                for layer in self.matrix_arr[::-1]:
                    if j != 0:
                        d = self.calc_d_log(d=d, w=aux, h=h[-(j + 1)])
                        dw[j] = self.calc_dw(d, v[-(j + 2)], j)
                    aux = layer
                    j += 1

                self.update_weights(dw)

                layer_historic.append([])
                for layer in self.matrix_arr:
                    layer_historic[k - 1].append(layer.copy())
                historic.append([])

                for j in range(0, len(data)):
                    a = data[j]
                    he, ve = self.calc_out(a, True)

                    res = ve[-1]
                    expec = expected[j]
                    if exp:
                        expec = exp(ve[-1], expected[j])

                    historic[k - 1].append(ve[-1])
                    error += np.average((np.subtract(expec, res) / 2) ** 2)

                error = error / len(data)
                errors_aux.append(error)

                if error <= error_max or n >= max_iter:
                    errors.append(np.average(errors_aux))
                    break

            errors.append(np.average(errors_aux))

        if self.dropout:
            self.scale_weights()

        return historic, errors, layer_historic, n

    def scale_weights(self):
        for layer_idx in range(len(self.matrix_arr)):
            if layer_idx == 0:
                prob = self.input_keep_prob
            else:
                prob = self.hidden_keep_prob
            self.matrix_arr[layer_idx] = np.array(
                [prob * i for i in self.matrix_arr[layer_idx]]
            )

    def initialize_values(self, n, layer_historic=None):
        dw = []
        if layer_historic is not None:
            layer_historic.append([])
        for i in range(0, len(self.matrix_arr)):
            if layer_historic is not None:
                layer_historic[n].append(self.matrix_arr[i].copy())
            dw.append(np.atleast_2d(np.zeros_like(self.matrix_arr[-(i + 1)])))

        return 0, dw

    def calc_d(self, h, d=None, w=None, res=None, exp=None):
        if w is not None and d is not None:
            return np.atleast_2d(d.dot(w) * np.atleast_2d(self.g_diff(h)))
        elif res is not None and exp is not None:
            return np.atleast_2d(np.subtract(res, exp) * self.g_diff(h))
        else:
            raise "your hands"

    def calc_d_log(self, h, d=None, w=None, res=None, exp=None):
        if w is not None and d is not None:
            return np.atleast_2d(d.dot(w) * np.atleast_2d(self.g_diff(h)))
        elif res is not None and exp is not None:
            return utils.get_b() * np.subtract(res, exp)
        else:
            raise "your hands"

    def calc_dw(self, d, v, j):
        return self.optimizer(d.T.dot(np.atleast_2d(v)), self.eta, j)

    def calc_out(self, data, training):
        h, v = [], []
        v.append(data)
        for layer in self.matrix_arr:
            if self.dropout:
                if layer is self.matrix_arr[0]:
                    prob = self.input_keep_prob
                else:
                    prob = self.hidden_keep_prob
                layer_cpy = layer.copy()
                for i in range(len(layer)):
                    if random.uniform(0, 1) > prob:
                        layer_cpy[i] = np.zeros(len(layer[i]))
                h.append(np.atleast_1d(layer_cpy @ v[-1]))
            else:
                h.append(np.atleast_1d(layer @ v[-1]))
            v.append(np.atleast_1d(np.array(self.g(h[-1]))))

        return h, v

    def update_weights(self, dw):
        j = 0
        for layer in self.matrix_arr[::-1]:
            layer += dw[j]
            j += 1
