import json
import random
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Thread

import matplotlib.pyplot as plt
import numpy as np

from tp1.ladoB.crossover import avg_cross, two_point_cross, uniform_cross
from tp1.ladoB.selection import (
    boltzmann_selection,
    elite_selection,
    roulette_selection,
    tournament_selection_det,
    tournament_selection_prob,
)
from tp1.ladoB.structure import Color, GeneticExecutor


def cut(generation: [Color], gen_n, target):
    if gen_n > 10000:
        return True
    for color in generation:
        if color.is_goal(target):
            print("Goal reached")
            print("Generation: " + gen_n.__str__())
            print("Color: " + color.__str__())
            return True
    return False


# def wrapper(x, y, cross, selection):
#     def start():
#         start_time = time.time()
#         ret = GeneticExecutor(x, y, cross, selection, cut).start()
#         return ret, (time.time() - start_time) * 1000
#     return start


def main():
    with open("config.json") as f:
        data = json.load(f)

        palette = np.array(data["palette"])
        target = np.array(data["target"])

    x = [
        Color(np.array([255, 0, 255])),
        Color(np.array([10, 0, 127])),
        Color(np.array([20, 255, 0])),
        Color(np.array([127, 127, 127])),
        Color(np.array([220, 55, 111])),
        Color(np.array([167, 127, 227])),
        Color(np.array([20, 10, 10])),
        Color(np.array([12, 27, 34])),
        Color(np.array([44, 25, 111])),
        Color(np.array([34, 12, 222])),
        Color(np.array([120, 255, 110])),
        Color(np.array([127, 255, 110])),
        Color(np.array([40, 255, 110])),
        Color(np.array([17, 5, 11])),
        Color(np.array([37, 255, 32])),
        Color(np.array([212, 23, 5])),
        Color(np.array([121, 255, 10])),
        Color(np.array([0, 255, 23])),
        Color(np.array([240, 255, 9])),
        Color(np.array([12, 32, 1])),
        Color(np.array([232, 255, 1])),
        Color(np.array([0, 23, 42])),
    ]
    y = Color(np.array([82, 15, 207]))
    acc = 0
    num_tries = 10

    with ThreadPoolExecutor() as executor:
        uniform_futures = [
            executor.submit(
                GeneticExecutor(x, y, two_point_cross, elite_selection, cut).start
            )
            for _ in range(0, num_tries)
        ]
        # print("Start Two Point Futures")
        # two_point_futures = [
        #     executor.submit(
        #         GeneticExecutor(x, y, two_point_cross, roulette_selection, cut).start
        #     )
        #     for _ in range(0, num_tries)
        # ]

        for fut in uniform_futures:
            acc += fut.result()

        print("avg: " + (acc / num_tries).__str__())

        # print("two_point cross: ")
        # acc = 0
        # for fut in two_point_futures:
        #     acc += fut.result()
        #
        # print("avg: " + (acc / num_tries).__str__())
    exit(0)

    # x = np.linspace(-10, 10)
    # plt.plot(x, -25 * np.arctan(x / 300 - 2) + 35)
    # plt.axis('tight')
    # plt.show()


if __name__ == "__main__":
    main()
