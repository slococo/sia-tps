import time

from tp1.ladoA.search_tree_node import SearchTreeNode
from tp1.utils import Queue, Stack
from queue import PriorityQueue


def hpa(root: SearchTreeNode, h, w):
    expanded_nodes = 1
    to_expand = PriorityQueue()
    if h is None and w != 0:
        raise Exception
    elif h is None:
        to_expand.put((root.cost, root))
    else:
        to_expand.put((h(root.game.first_node, root.game.nodes), root))
    root.cost = 0
    while to_expand:
        prio, current = to_expand.get()
        if current.is_goal():
            print("Expanded nodes: " + expanded_nodes.__str__())
            return get_solution(current, root)

        for child_node in current.get_children():
            expanded_nodes += 1
            if h is None and w != 0:
                raise Exception
            elif h is None:
                to_expand.put((child_node.cost, child_node))
            else:
                to_expand.put((((1 - w) * child_node.cost + w * h(child_node.game.first_node,
                                                                  child_node.game.nodes)) * 2, child_node))


def get_solution(current_node: SearchTreeNode, root: SearchTreeNode):
    solution = []
    solution = [current_node.game.first_node.color, *solution]
    current_node = current_node.parent
    while current_node != root and current_node is not None:
        solution = [current_node.game.first_node.color, *solution]
        current_node = current_node.parent
    return solution


def a_star(root: SearchTreeNode, h):
    return hpa(root, h, 0.5)


def greedy(root: SearchTreeNode, h):
    return hpa(root, h, 1)


def generic_search(root: SearchTreeNode, pending_nodes):
    pending_nodes.enqueue(root)

    while pending_nodes.__len__() > 0:
        current_node = pending_nodes.dequeue()

        if current_node.is_goal():
            return get_solution(current_node, root)

        for child_node in current_node.get_children():
            pending_nodes.enqueue(child_node)

    return None


def dfs(root: SearchTreeNode):
    return generic_search(root, Stack())


def bfs(root: SearchTreeNode):
    return generic_search(root, Queue())
