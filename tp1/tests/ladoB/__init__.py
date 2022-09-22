import json

from tp1.ladoB.crossover import two_point_cross, uniform_cross
from tp1.ladoB.mutation import (
    complete_mutate,
    limited_multigen_mutate,
    one_gen_mutate,
    uniform_mutate,
)
from tp1.ladoB.selection import (
    boltzmann_selection,
    elite_selection,
    roulette_selection,
    tournament_selection_det,
    tournament_selection_prob,
)
from tp1.ladoB.structure import Color

tries_per_run = 10
with open("config.json") as f:
    data = json.load(f)
    params = []
    for elem in data:
        aux_dict = {
            "target": Color(elem["target"]),
            "palette": [Color(x) for x in elem["palette"]],
        }
        params.append(aux_dict)

    all_crossovers = [two_point_cross, uniform_cross]
    all_selections = [
        elite_selection,
        roulette_selection,
        tournament_selection_det,
        tournament_selection_prob,
        boltzmann_selection,
    ]
    all_mutations = [
        uniform_mutate,
        complete_mutate,
        one_gen_mutate,
        limited_multigen_mutate,
    ]
