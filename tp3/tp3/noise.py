import numpy as np

def gaussian_noise(mu, sigma, data):
    ret_data = np.array(data)
    noise = np.random.normal(mu, sigma, np.shape(ret_data))
    return ret_data + noise

def uniform_noise(a, b, data):
    ret_data = np.array(data)
    noise = np.random.normal(a, b, np.shape(ret_data))
    return ret_data + noise