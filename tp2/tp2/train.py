import numpy as np
import math
# import pandas as pd
import random


# def online(data, weights, g, error_max, max_iter):
#     error = math.inf
#     n = 0
#     while error > error_max or n < max_iter:
#         i = random.randint(0, data.length - 1)
#
#
#
# def batch(data, weights, eta, g, g_diff, error_max, max_iter):
#     error = math.inf
#     n = 0
#     while error > error_max or n < max_iter:
#         for u in data:
#             aux = np.array(u[::-1])
#
#             o = weights @ aux.T
#             error = 1/2 * (u[-1] - o) ** 2
#             dw =
