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


S = []
e = 1e-8
gamma = 0.95


def rms_prop(diff, eta, j):
    global S, e, gamma
    print()
    if j == len(S):
        S.append(0)
    elif j > len(S):
        raise "Invalid j"
    S[j] = S[j] * gamma + (1 - gamma) * np.power(diff, 2)
    return eta * np.divide(diff, np.sqrt(S[j] + e))


beta1, beta2, eps, t = 0.9, 0.999, 1e-8, 0
m, v, theta = [], [], []
preveta, cumeta, cumetas, cumdelta = 0, 0, 0, 0


def adam(diff, eta, j):
    # diff = -diff
    global m, v, theta, t, cumeta, cumetas, preveta, cumdelta
    # cumeta += eta
    # cumetas += eta ** 2
    # cumdelta += eta - preveta
    # print(cumetas/cumeta)
    # print(cumdelta/cumeta)
    # print()
    if not j:
        t += 1
    if j == len(m):
        m.append(0)
        v.append(0)
        theta.append(0)
    elif j > len(m):
        raise "Invalid j"
    # for _ in range(0, 15):
    prev_theta = np.inf
    while not np.allclose(prev_theta, theta[j]):
        m[j] = beta1 * m[j] - (1 - beta1) * diff
        v[j] = beta2 * v[j] + (1 - beta2) * np.power(diff, 2)
        m_hat = - np.divide(m[j], 1 - np.power(beta1, t))
        v_hat = np.divide(v[j], 1 - np.power(beta2, t))
        prev_theta = theta[j]
        theta[j] = theta[j] + eta * m_hat / (np.sqrt(v_hat) + eps)
    # print(prev_theta)
    preveta = eta
    return theta[j]


b1, b2, ta, va = 0.9, 0.999, 0, 0
ma, w = [], []


def adamax(diff, eta, j):
    global ma, va, w, ta
    if not j:
        ta += 1
    if j == len(ma):
        ma.append(0)
        w.append(0)
    elif j > len(ma):
        raise "Invalid j"
    ma[j] = beta1 * ma[j] - (1 - beta1) * diff
    ma_hat = np.divide(ma[j], 1 - np.power(b1, t))
    va = np.maximum(b2 * va, np.abs(diff))
    w[j] = w[j] + eta * ma_hat / v
    return w[j]


def gradient(diff, eta, _):
    return eta * diff
