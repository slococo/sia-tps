from typing import Set

from tp1.ladoA.structure import Node


def graph_max_distance(starting_node: Node, nodes):
    dijkstra(starting_node, nodes)
    dist = 0
    for node in nodes:
        if dist != float("inf"):
            dist = max(dist, node.distance)

    return dist


def dijkstra(initial_node: Node, nodes: Set[Node]):
    unvisited = set()

    for aux in nodes:
        if aux is initial_node:
            aux.distance = 0
        else:
            aux.distance = float("inf")
        unvisited.add(aux)

    while unvisited:
        dist = float("inf")
        for aux_node in unvisited:
            if aux_node.distance <= dist:
                node = aux_node
                dist = node.distance
        unvisited.remove(node)

        for front_node in node.frontier:
            front_node.distance = min(front_node.distance, node.distance + 1)


def frontier_color_count(starting_node: Node, nodes):
    frontier_colors = set()
    for frontier_node in starting_node.frontier:
        frontier_colors.add(frontier_node.color)
    return len(frontier_colors)


def amount_of_nodes(starting_node: Node, nodes):
    return len(nodes)
