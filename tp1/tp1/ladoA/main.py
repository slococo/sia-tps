import argparse
import logging
import time

from tp1.ladoA.algorithms import a_star, bfs, dfs, greedy
from tp1.ladoA.game import Game
from tp1.ladoA.heuristics import (amount_of_nodes, frontier_color_count,
                                  graph_max_distance)
from tp1.ladoA.search_tree_node import SearchTreeNode

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--size",
        help="Size of the game matrix",
        type=int,
        required=False,
        default=14,
    )  # TODO: Change to 14
    parser.add_argument(
        "-c",
        "--colors",
        help="Number of different colors in game",
        type=int,
        required=False,
        default=4,
    )  # TODO: Change to 6
    parser.add_argument(
        "-a",
        "--algorithm",
        help="Preferred algorithm",
        choices={"bfs", "dfs", "greedy", "astar"},
        required=False,
        default="astar",
    )
    parser.add_argument(
        "-k",
        "--heuristic",
        help="Preferred heuristic to use",
        choices={"frontier", "max_count"},
        required=False,
        default="max_count",
    )
    args = parser.parse_args()
    size = args.size
    colors = args.colors
    algorithm = args.algorithm
    heuristic = args.heuristic

    if size <= 0:
        logger.error(f"Game size must be positive. Number provided was {size}")
        raise argparse.ArgumentTypeError(
            f"Game size must be positive. Number provided was {size}"
        )
    if colors <= 0:
        logger.error(f"Number of colors must be positive. Number provided was {colors}")
        raise argparse.ArgumentTypeError(
            f"Number of colors must be positive. Number provided was {colors}"
        )

    game = Game(size, colors)
    game.key = 0

    tree_node_root = SearchTreeNode(game, None)

    if algorithm == "dfs":
        print(dfs(tree_node_root))
    if algorithm == "bfs":
        print(bfs(tree_node_root))
    if algorithm == "greedy":
        if heuristic == "frontier":
            print(greedy(tree_node_root, frontier_color_count))
        else:
            print(greedy(tree_node_root, graph_max_distance))
    if algorithm == "astar":
        if heuristic == "frontier":
            print(a_star(tree_node_root, frontier_color_count))
        else:
            start_time = time.time()
            print(a_star(tree_node_root, frontier_color_count))
            print("{:.2f}".format((time.time() - start_time) * 1000) + "ms")

    # start_time = time.time()
    # print(a_star(tree_node_root, graph_max_distance))
    # print("{:.2f}".format((time.time() - start_time) * 1000) + "ms")
    #
    # start_time = time.time()
    # print(a_star(tree_node_root, amount_of_nodes))
    # print("{:.2f}".format((time.time() - start_time) * 1000) + "ms")
    #
    # print(bfs(tree_node_root))


if __name__ == "__main__":
    main()
