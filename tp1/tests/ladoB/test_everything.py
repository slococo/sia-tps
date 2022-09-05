from tests.ladoB import (
    all_crossovers,
    all_mutations,
    all_selections,
    params,
    tries_per_run,
)
from tp1.ladoB.structure import GeneticExecutor

"""Test each combination of crossover, selection and mutation methods,
 each 10 times, for each input-output pair in config.json"""


def test_everything():
    for cross in all_crossovers:
        for elem in params:
            for selection in all_selections:
                for mut in all_mutations:
                    print(f"{cross.__name__} - {selection.__name__} - {mut.__name__}")
                    acc = 0
                    for i in range(tries_per_run):
                        gen_exec = GeneticExecutor(
                            elem["palette"],
                            elem["target"],
                            cross,
                            selection,
                            mut,
                            max_gen=5000,
                        )
                        acc += gen_exec.start()
                        if gen_exec.found_sol:
                            print(
                                f"FAILED {cross.__name__} - {selection.__name__} - {mut.__name__}"
                            )
                    print(f"Average: {acc / tries_per_run}")
