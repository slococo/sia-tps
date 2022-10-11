import argparse
import logging
import time

from tp1.ladoA.algorithms import a_star, bfs, dfs, greedy, iddfs
from tp1.ladoA.game import Game
from tp1.ladoA.heuristics import (
    amount_of_nodes,
    frontier_color_count,
    graph_max_distance,
)
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
        default=10,
    )
    parser.add_argument(
        "-c",
        "--colors",
        help="Number of different colors in game",
        type=int,
        required=False,
        default=2,
    )
    parser.add_argument(
        "-a",
        "--algorithm",
        help="Preferred algorithm",
        choices={"bfs", "dfs", "greedy", "astar", "iddfs"},
        required=False,
        default="astar",
    )
    parser.add_argument(
        "-k",
        "--heuristic",
        help="Preferred heuristic to use",
        choices={"frontier", "max_distance", "amount_nodes"},
        required=False,
        default="amount_nodes",
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

    start_time = time.time()
    if algorithm == "dfs":
        print(dfs(tree_node_root, None))
    if algorithm == "bfs":
        print(bfs(tree_node_root, None))
    if algorithm == "greedy":
        if heuristic == "frontier":
            print(greedy(tree_node_root, frontier_color_count))
        elif heuristic == "max_distance":
            print(greedy(tree_node_root, graph_max_distance))
        else:
            print(greedy(tree_node_root, amount_of_nodes))
    if algorithm == "astar":
        if heuristic == "frontier":
            print(a_star(tree_node_root, frontier_color_count))
        elif heuristic == "max_distance":
            print(a_star(tree_node_root, graph_max_distance))
        else:
            print(a_star(tree_node_root, amount_of_nodes))
    if algorithm == "iddfs":
        if heuristic == "frontier":
            print(iddfs(tree_node_root, frontier_color_count))
        elif heuristic == "max_distance":
            print(iddfs(tree_node_root, graph_max_distance))
        else:
            print(iddfs(tree_node_root, amount_of_nodes))
    print("{:.2f}".format((time.time() - start_time) * 1000) + "ms")


if __name__ == "__main__":
    main()
