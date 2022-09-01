import json

import numpy as np


def main():
    with open('config.json') as f:
        data = json.load(f)

        palette = np.array(data['palette'])
        target = np.array(data['target'])


if __name__ == '__main__':
    main()
