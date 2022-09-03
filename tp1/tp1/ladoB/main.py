import json
import random
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt

from threading import Thread
import numpy as np

from tp1.ladoB.crossover import uniform_cross, avg_cross, two_point_cross
from tp1.ladoB.selection import elite_selection, boltzmann_selection, roulette_selection, tournament_selection_det, \
    tournament_selection_prob
from tp1.ladoB.structure import Color, GeneticExecutor


def cut(generation: [Color], gen_n, target):
    if gen_n > 10000:
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
         Color(np.array([120, 255, 110])), Color(np.array([127, 255, 110])),
         Color(np.array([40, 255, 110])), Color(np.array([17, 5, 11])),
         Color(np.array([37, 255, 32])), Color(np.array([212, 23, 5])),
         Color(np.array([121, 255, 10])), Color(np.array([0, 255, 23])),
         Color(np.array([240, 255, 9])), Color(np.array([12, 32, 1])),
         Color(np.array([232, 255, 1])), Color(np.array([0, 23, 42]))]
    y = Color(np.array([127, 255, 127]))
    acc = 0

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(GeneticExecutor(x, y, uniform_cross, roulette_selection, cut).start)
                   for _ in range(0, 10)]

    for fut in futures:
        acc += fut.result()
        print(fut.result())

    print("avg: " + (acc / 10).__str__())

    # x = np.linspace(-10, 10)
    # plt.plot(x, -25 * np.arctan(x / 300 - 2) + 35)
    # plt.axis('tight')
    # plt.show()


if __name__ == '__main__':
    main()
