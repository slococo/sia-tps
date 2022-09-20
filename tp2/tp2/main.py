import json

import matplotlib.pyplot as plt
import numpy as np


def main(config_path=None):
    if config_path is None:
        config_path = "tp2/config.json"
    with open(config_path) as f:
        data = json.load(f)


if __name__ == "__main__":
    main("config.json")
