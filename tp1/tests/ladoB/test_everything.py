import time

import matplotlib.pyplot as plt
import numpy as np
from tests.ladoB import (all_crossovers, all_mutations, all_selections, params,
                         tries_per_run)

from tp1.ladoB.structure import GeneticExecutor

"""Test each combination of crossover, selection and mutation methods,
 each 10 times, for each input-output pair in config.json"""


def test_everything():
    x = []
    y = []
    for cross in all_crossovers:
        print(cross)
        for elem in params:
            for selection in all_selections:
                print(selection)
                name = []
                time_aux = []
                gen_aux = []
                for mut in all_mutations:
                    print(mut)
                    print(f"{cross.__name__} - {selection.__name__} - {mut.__name__}")
                    acc = 0
                    acc_time = 0
                    for i in range(tries_per_run):
                        start_time = time.time()
                        gen_exec = GeneticExecutor(
                            elem["palette"],
                            elem["target"],
                            cross,
                            selection,
                            mut,
                            max_gen=2000,
                            gen_size=50,
                        )
                        aux = gen_exec.start()
                        acc += aux
                        exec_time = (time.time() - start_time) * 1000
                        acc_time += exec_time
                        if gen_exec.found_sol:
                            print(
                                f"FAILED {cross.__name__} - {selection.__name__} - {mut.__name__}"
                            )
                        time_aux.append(exec_time)
                        gen_aux.append(aux)
                    print(f"Average: {acc / tries_per_run}")
                    print(f"Average time: {acc_time / tries_per_run} ms")
                    x.append(time_aux)
                    y.append(gen_aux)
                    time_aux = []
                    gen_aux = []
                    name.append(
                        cross.__name__[:2] + selection.__name__[:2] + mut.__name__[:2]
                    )
                print_bar_graph(
                    x, "Tiempo", name, "selection" + time.time().__str__() + ".png"
                )
                print_bar_graph(
                    y,
                    "Generaciones",
                    name,
                    "generation" + time.time().__str__() + ".png",
                )
                x = []
                y = []


def print_bar_graph(val, title, x_names, fig_name):
    mean = []
    std = []
    print(len(x_names))
    for i in range(0, len(x_names)):
        print(val)
        mean.append(np.mean(np.array([val[0][i], val[1][i], val[2][i], val[3][i]])))
        std.append(np.std(np.array([val[0][i], val[1][i], val[2][i], val[3][i]])))

    print(mean)
    print(std)
    plt.title(title)
    plt.bar(x_names, mean, yerr=std, ecolor="black", capsize=10)
    plt.grid(axis="y", c="lightgray", linewidth=0.5, linestyle="-")
    plt.axis(ymin=0)
    plt.savefig(fig_name, dpi=300)
    plt.show()
