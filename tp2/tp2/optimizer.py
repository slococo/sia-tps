import numpy as np

b = 0.1
a = 0.00001
prev_e = 0
k0 = 10
k = k0
tendency = 0


def adaptative_eta(e, eta):
    global tendency, prev_e, k, k0
    sign = np.sign(e - prev_e)
    if tendency * sign > 0:
        k -= 1
        if k <= 0:
            if sign == -1:
                return eta - b * eta
            else:
                return eta + a
    elif tendency * sign < 0:
        tendency = sign
        k = k0
    return eta


dw = []
alpha = 0.3


def momentum(diff, eta, j):
    global dw
    if j == len(dw):
        dw.append(0)
    elif j > len(dw):
        raise "Invalid j"
    dw[j] = alpha * dw[j] + eta * diff
    return dw[j]


def gradient(diff, eta, j):
    return eta * diff


def rms_prop():
    pass


def adam():
    pass
