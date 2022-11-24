import json

import matplotlib
import numpy as np

from tp3 import utils

matplotlib.use("TkAgg")

import pandas as pd


class Loader:
    def __init__(self):
        pass


class CSVLoader(Loader):
    @classmethod
    def load(cls, path, has_bias, data_columns, _, normalised=False):
        try:
            with open(path) as f:
                df = pd.read_csv(f)
                data = df[data_columns].to_numpy()
        except FileNotFoundError:
            raise "Data file path is incorrect"

        if not normalised:
            data, _, _ = utils.normalise(data, -1, 1)

        data_b = data
        if not has_bias:
            data_b = np.insert(data, 0, 1, axis=1)

        return data_b, data


class JSONLoader(Loader):
    @classmethod
    def load(cls, path, dataset):
        try:
            with open(path) as f:
                data = json.load(f)
                data = np.array(data[dataset])
        except FileNotFoundError:
            raise "Data file path is incorrect"

        return data
