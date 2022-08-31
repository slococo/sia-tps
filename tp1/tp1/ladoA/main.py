import sys

from tp1.ladoA.algorithms import a_star
from tp1.ladoA.game import Game
from tp1.ladoA.heuristics import graph_max_distance, frontier_color_count
from tp1.ladoA.search_tree_node import SearchTreeNode


def main():
    # print(sys.setrecursionlimit(100000))
    game = Game(3, 5)
    matrix = game.matrix
    print(matrix)

    print(a_star(SearchTreeNode(game, None), frontier_color_count))

    # print(matrix)
    # print("----------")
    # print_graph(game.first_node, set(), 1)
    # print()
    # print("---------------------------------------------")
    # game.first_node.absorb(2)
    # print_graph(game.first_node, set(), 1)


if __name__ == "__main__":
    main()
