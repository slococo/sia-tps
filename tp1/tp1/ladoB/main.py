import json

import numpy as np

from tp1.ladoB.crossover import uniform_cross
from tp1.ladoB.selection import elite_selection
from tp1.ladoB.structure import Color, GeneticExecutor


def cut(generation: [Color], gen_n, target):
    if gen_n > 100:
        return True
    for color in generation:
        if color.is_goal(target):
            return True
    return False


def main():
    with open('config.json') as f:
        data = json.load(f)

        palette = np.array(data['palette'])
        target = np.array(data['target'])

    x = [Color(np.array([255, 0, 255])), Color(np.array([10, 0, 127])),
         Color(np.array([20, 255, 0])), Color(np.array([127, 127, 127])),
         Color(np.array([220, 55, 111])), Color(np.array([167, 127, 227])),
         Color(np.array([20, 10, 10])), Color(np.array([12, 27, 34])),
         Color(np.array([44, 25, 111])), Color(np.array([34, 12, 222])),
         Color(np.array([127, 255, 110]))]
    y = Color(np.array([127, 255, 127]))
    GeneticExecutor(x, y, uniform_cross, elite_selection, cut).start()


if __name__ == '__main__':
    main()
