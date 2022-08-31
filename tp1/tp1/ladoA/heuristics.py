from typing import Set

from tp1.ladoA.structure import Node

import math


def graph_max_distance(starting_node: Node, nodes):
    dijkstra(starting_node, nodes)
    dist = 0
    for node in nodes:
        dist = max(dist, node.distance)
    return dist


def dijkstra(initial_node: Node, nodes: Set[Node]):
    unvisited = nodes.copy()
    node = initial_node
    for node in unvisited:
        node.distance = math.inf
    node.distance = 0
    while not unvisited:
        for front in node.frontier:
            front.distance = max(node.distance + 1, front.distance)
        node.visited = True
        unvisited.remove(node)
        if not unvisited:
            node = unvisited.get(0)
            for min_node in unvisited:
                if min_node.distance < node.distance:
                    node = min_node


def frontier_color_count(starting_node: Node, nodes):
    frontier_colors = set()
    for frontier_node in starting_node.frontier:
        frontier_colors.add(frontier_node.get_color())
    return len(frontier_colors)
