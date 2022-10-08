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
    dw[j] = alpha * dw[j] - eta * diff
    return dw[j]


S = []
e = 1e-8
gamma = 0.95


def rms_prop(diff, eta, j):
    global S, e, gamma
    if j == len(S):
        S.append(0)
    elif j > len(S):
        raise "Invalid j"
    S[j] = S[j] * gamma + (1 - gamma) * np.power(diff, 2)
    return - eta * np.divide(diff, np.sqrt(S[j] + e))


beta1, beta2, eps, t = 0.9, 0.999, 1e-8, 0
m, v, theta = [], [], []


def adam(diff, eta, j):
    global m, v, theta, t
    if j == len(m):
        m.append(0)
        v.append(0)
        theta.append(0)
    elif j > len(m):
        raise "Invalid j"
    prev_theta = np.inf
    while not np.allclose(prev_theta, theta[j]):
        t += 1
        m[j] = beta1 * m[j] + (1 - beta1) * diff
        v[j] = beta2 * v[j] + (1 - beta2) * np.power(diff, 2)
        m_hat = - np.divide(m[j], 1 - np.power(beta1, t))
        v_hat = np.divide(v[j], 1 - np.power(beta2, t))
        prev_theta = np.atleast_1d(theta[j]).copy()
        theta[j] -= eta * m_hat / (np.sqrt(v_hat) + eps)
    t = 0
    return theta[j]


b1, b2, ta = 0.9, 0.999, 0
ma, va, w = [], [], []


def adamax(diff, eta, j):
    global ma, va, w, ta
    if j == len(ma):
        ma.append(0)
        va.append(0)
        w.append(0)
    elif j > len(ma):
        raise "your hands"
    prev_w = np.inf
    while not np.allclose(prev_w, w[j]):
        ta += 1
        ma[j] = beta1 * ma[j] + (1 - beta1) * diff
        ma_hat = np.divide(ma[j], 1 - np.power(b1, ta))
        va[j] = np.maximum(b2 * va[j], np.abs(diff))
        prev_w = np.atleast_1d(w[j]).copy()
        w[j] -= eta * ma_hat / va[j]
    ta = 0
    return w[j]


# def nadam(diff, eta, j):
#     global m, v, theta, t
#     if j == len(m):
#         m.append(0)
#         v.append(0)
#         theta.append(0)
#     elif j > len(m):
#         raise "Invalid j"
#     # for _ in range(0, 15):
#     prev_theta = np.inf
#     while not np.allclose(prev_theta, theta[j]):
#         t += 1
#         m[j] = beta1 * m[j] + (1 - beta1) * diff
#         v[j] = beta2 * v[j] + (1 - beta2) * np.power(diff, 2)
#         m_hat = - np.divide(m[j], 1 - np.power(beta1, t))
#         v_hat = np.divide(v[j], 1 - np.power(beta2, t))
#         prev_theta = theta[j]
#         theta[j] -= eta * m_hat / (np.sqrt(v_hat) + eps)
#     # print(prev_theta)
#     # preveta = eta
#     t = 0
#     return theta[j]

def gradient(diff, eta, _):
    return - eta * diff
