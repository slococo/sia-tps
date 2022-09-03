from random import randint, uniform

from tp1.ladoB.structure import Color

MUTATION_PROB = 0.1


def uniform_mutate(color: Color):
    for gen in color.genes:
        mut = uniform(0, 1)
        if mut < MUTATION_PROB:
            gen ^= 1  # XOR Gen
    color.calc_rgb()


def one_gen_mutate(color: Color):
    prob = uniform(0, 1)
    if prob < MUTATION_PROB:
        gen_idx = randint(0, len(color.genes) - 1)
        color.genes[gen_idx] ^= 1
    color.calc_rgb()


def limited_multigen_mutate(color: Color):
    gen_len = len(color.genes)
    mutated = []
    gen_idx = randint(0, gen_len - 1)
    for i in range(0, randint(1, gen_len)):
        while gen_idx in mutated:
            gen_idx = randint(0, gen_len - 1)
        mutated = [*mutated, gen_idx]
        if uniform(0, 1) < MUTATION_PROB:
            color.genes[gen_idx] ^= 1
    color.calc_rgb()


def complete_mutate(color: Color):
    if uniform(0, 1) < MUTATION_PROB:
        for gen in color.genes:
            gen ^= 1
    color.calc_rgb()
