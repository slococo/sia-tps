import numpy as np

b = 0.1
a = None
prev_e = 0
k0 = 10
k = k0
tendency = 0


def adaptative_eta(e, eta):
    global tendency, prev_e, k, k0, a
    if not a:
        a = eta
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
    return -eta * np.divide(diff, np.sqrt(S[j] + e))


beta1, beta2, eps, t = 0.9, 0.999, 1e-8, 0
m, v = [], []


def reset_state():
    global m, v, ma, va,  mn, vn, tn, mg, vg, vg_hat, Eg, Ex, dx, dw, S, t, prev_e, tendency, ta, tn, k0, k, a
    m, v, ma, va,  mn, vn, tn, mg, vg, vg_hat, Eg, Ex, dx, dw, S = [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
    t, prev_e, tendency, ta, tn = 0, 0, 0, 0, 0
    k0, k, a = 10, 10, None


def adam(diff, eta, j):
    global m, v, t
    if j == len(m):
        m.append(0)
        v.append(0)
    elif j > len(m):
        raise "Invalid j"
    t += 1
    m[j] = beta1 * m[j] + (1 - beta1) * diff
    v[j] = beta2 * v[j] + (1 - beta2) * np.power(diff, 2)
    m_hat = np.divide(m[j], 1 - np.power(beta1, t))
    v_hat = np.divide(v[j], 1 - np.power(beta2, t))
    return -eta * m_hat / (np.sqrt(v_hat) + eps)


ta = 0
ma, va = [], []


def adamax(diff, eta, j):
    global ma, va, ta
    if j == len(ma):
        ma.append(0)
        va.append(0)
    elif j > len(ma):
        raise "Invalid j"
    ta += 1
    ma[j] = beta1 * ma[j] + (1 - beta1) * diff
    ma_hat = np.divide(ma[j], 1 - np.power(beta1, ta))
    va[j] = np.maximum(beta2 * va[j], np.abs(diff))
    return -eta * ma_hat / (va[j] + eps)


mn, vn, tn = [], [], 0


def nadam(diff, eta, j):
    global mn, vn, tn
    if j == len(mn):
        mn.append(0)
        vn.append(0)
    elif j > len(mn):
        raise "Invalid j"
    tn += 1
    mn[j] = beta1 * mn[j] + (1 - beta1) * diff
    vn[j] = beta2 * vn[j] + (1 - beta2) * np.power(diff, 2)
    m_hat = np.divide(
        mn[j],
        (1 - np.power(beta1, tn))
        + np.divide((1 - beta1) * diff, (1 - np.power(beta1, tn))),
    )
    v_hat = np.divide(vn[j], 1 - np.power(beta2, tn))
    return -eta * m_hat / (np.sqrt(v_hat) + eps)


mg, vg, vg_hat = [], [], []


def amsgrad(diff, eta, j):
    global mg, vg, vg_hat
    if j == len(mg):
        mg.append(0)
        vg.append(0)
    elif j > len(mg):
        raise "Invalid j"
    mg[j] = beta1 * mg[j] + (1 - beta1) * diff
    vg[j] = beta2 * vg[j] + (1 - beta2) * np.power(diff, 2)
    vg_hat = np.maximum(vg[j], vg_hat)
    return -eta * mg[j] / (np.add(np.sqrt(vg_hat), eps))


Eg, Ex, rho, dx = [], [], 0.9, []


def adadelta(diff, _, j):
    global Eg, Ex, dx
    if j == len(Eg):
        Eg.append(0)
        Ex.append(0)
        dx.append(0)
    elif j > len(Eg):
        raise "Invalid j"

    Eg[j] = rho * Eg[j] + (1 - rho) * np.power(diff, 2)
    rmsEg = np.sqrt(Eg[j] + e)
    rmsEx = np.sqrt(Ex[j] + e)
    dx[j] = -np.divide(rmsEx * diff, rmsEg)
    Ex[j] = rho * Ex[j] + (1 - rho) * np.power(dx[j], 2)

    return dx[j]


def gradient(diff, eta, _):
    return -eta * diff
