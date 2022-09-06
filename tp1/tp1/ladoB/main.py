import json

import matplotlib.pyplot as plt
import numpy as np

from tp1.ladoB.crossover import avg_cross, two_point_cross, uniform_cross
from tp1.ladoB.mutation import (complete_mutate, limited_multigen_mutate,
                                one_gen_mutate, uniform_mutate)
from tp1.ladoB.selection import (boltzmann_selection, elite_selection,
                                 roulette_selection, tournament_selection_det,
                                 tournament_selection_prob)
from tp1.ladoB.structure import Color, GeneticExecutor


def main(config_path=None):
    if config_path is None:
        config_path = "tp1/ladoB/config.json"
    with open(config_path) as f:
        data = json.load(f)

        print(
            GeneticExecutor(
                [Color(x) for x in data["palette"]],
                Color(data["target"]),
                globals()[data["crossover"]],
                globals()[data["selection"]],
                globals()[data["mutation"]],
                max_gen=data["max_gen"],
                gen_size=data["gen_size"],
            ).start()
        )


if __name__ == "__main__":
    main("config.json")
