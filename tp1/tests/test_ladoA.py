import resource
import time

import pytest

from tp1.ladoA import __version__

import matplotlib.pyplot as plt
import numpy as np

from tp1.ladoA.algorithms import a_star, bfs, dfs, greedy, iddfs
from tp1.ladoA.game import Game
from tp1.ladoA.heuristics import (amount_of_nodes, frontier_color_count,
                                  graph_max_distance)
from tp1.ladoA.search_tree_node import SearchTreeNode
from tp1.ladoA.structure import Matrix, Cell


@pytest.fixture(autouse=True)
def before():
    with open('/proc/meminfo', 'r') as mem:
        free_memory = 0
        for i in mem:
            sline = i.split()
            if str(sline[0]) in ('MemFree:', 'Buffers:', 'Cached:'):
                free_memory += int(sline[1])
    print(free_memory)
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS, (round(free_memory * 1024 * 0.9), hard))

    yield


@pytest.fixture
def create_matrix():
    size = 8
    num_colors = 5
    matrix = Matrix(size)
    for i in range(size):
        for j in range(size):
            cell = Cell(np.random.randint(1, num_colors + 1), i, j)
            matrix[i][j] = cell
    return matrix


def tree_node_root(matrix):
    game = Game(8, 5, matrix)
    game.key = 0
    return SearchTreeNode(game, None)


def test_admissible(create_matrix):
    x = []
    y = []
    for i in range(0, 5):
        root = tree_node_root(create_matrix)
        time_aux = []
        nodes_aux = []
        for heuristic in [frontier_color_count, graph_max_distance]:
            start_time = time.time()
            sol, nodes = a_star(root, heuristic)
            time_aux.append((time.time() - start_time) * 1000)
            nodes_aux.append(nodes)
        start_time = time.time()
        sol, nodes = iddfs(root, heuristic)
        time_aux.append((time.time() - start_time) * 1000)
        nodes_aux.append(nodes)
        start_time = time.time()
        sol, nodes = bfs(root, heuristic)
        time_aux.append((time.time() - start_time) * 1000)
        nodes_aux.append(nodes)

        x.append(time_aux)
        y.append(nodes_aux)

    print_bar_graph(x, 'Tiempo promedio (en ms)', ['A* (colores)', 'A* (distancia)', 'IDDFS', 'BFS'], 'tiempo_admisible.png')
    print_bar_graph(y, 'Nodos expandidos', ['A* (colores)', 'A* (distancia)', 'IDDFS', 'BFS'], 'nodos_admisible.png')

    
def print_bar_graph(val, title, x_names, fig_name):
    mean = []
    std = []
    print(len(x_names))
    for i in range(0, len(x_names)):
        mean.append(np.mean(np.array([val[0][i], val[1][i], val[2][i], val[3][i], val[4][i]])))
        std.append(np.std(np.array([val[0][i], val[1][i], val[2][i], val[3][i], val[4][i]])))

    print(mean)
    print(std)
    plt.title(title)
    plt.bar(x_names, mean, yerr=std, ecolor='black', capsize=10)
    plt.grid(axis='y', c='lightgray', linewidth=0.5, linestyle='-')
    plt.axis(ymin=0)
    plt.savefig(fig_name, dpi=300)
    plt.show()


def test_not_admissible(create_matrix):
    x = []
    y = []
    for i in range(0, 5):
        root = tree_node_root(create_matrix)
        time_aux = []
        nodes_aux = []
        for algo in [a_star, dfs, greedy]:
            start_time = time.time()
            sol, nodes = algo(root, amount_of_nodes)
            time_aux.append((time.time() - start_time) * 1000)
            nodes_aux.append(nodes)

        x.append(time_aux)
        y.append(nodes_aux)

    print_bar_graph(x, 'Tiempo promedio (en ms)', ['A* (cantidad)', 'DFS', 'Greedy'], 'tiempo_no_admisible.png')
    print_bar_graph(y, 'Nodos expandidos', ['A* (cantidad)', 'DFS', 'Greedy'], 'nodos_no_admisible.png')
