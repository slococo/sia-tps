from tests.ladoB import all_crossovers, all_selections, params, tries_per_run
from tp1.ladoB.crossover import two_point_cross, uniform_cross
from tp1.ladoB.structure import GeneticExecutor

"""Functions to test different cross methods and see how they compare"""


def test_crossovers():
    for cross in all_crossovers:
        print("Starting cross" + cross.__name__)
        for elem in params:
            for selection in all_selections:
                acc = 0
                for i in range(tries_per_run):
                    gen_exec = GeneticExecutor(
                        elem["palette"], elem["target"], cross, selection, max_gen=5000
                    )
                    acc += gen_exec.start()
                print("Selection: ", selection.__name__, "Avg: ", acc / tries_per_run)
