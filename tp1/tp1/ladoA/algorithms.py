import math
import time
from concurrent.futures import ThreadPoolExecutor
from queue import PriorityQueue

from tp1.ladoA.search_tree_node import SearchTreeNode
from tp1.utils import Queue, Stack


# def hpa(root: SearchTreeNode, h, w):
#     expanded_nodes = 1
#     to_expand = PriorityQueue()
#     if h is None and w != 0:
#         raise Exception
#     elif h is None:
#         to_expand.put((root.cost, root))
#     else:
#         to_expand.put((h(root, root.game.first_node, root.game.nodes), root))
#     root.cost = 0
#     while to_expand:
#         prio, current = to_expand.get()
#         expanded_nodes += 1
#         if current.is_goal():
#             print("Expanded nodes: " + expanded_nodes.__str__())
#             return get_solution(current, root)
#
#         for child_node in current.get_children():
#             if h is None:
#                 to_expand.put((child_node.cost, child_node))
#             else:
#                 res = h(current, child_node.game.first_node, child_node.game.nodes)
#                 to_expand.put(
#                     (
#                         (
#                             (1 - w) * child_node.cost + w * res[0]
#                         )
#                         * 2,
#                         child_node,
#                     )
#                 )


def hpa_thread(root: SearchTreeNode, h, w):
    expanded_nodes = 1
    to_expand = PriorityQueue()
    if h is None and w != 0:
        raise Exception
    elif h is None:
        to_expand.put((root.cost, root))
    else:
        to_expand.put((h(root, root.game.first_node, root.game.nodes), root))
    root.cost = 0
    while to_expand:
        prio, current = to_expand.get()
        expanded_nodes += 1
        if current.is_goal():
            # print("Expanded nodes: " + expanded_nodes.__str__())
            return get_solution(current, root), expanded_nodes

        if h is None:
            for child_node in current.get_children():
                to_expand.put((child_node.cost, child_node))

        else:
            with ThreadPoolExecutor() as executor:
                futures = [
                    executor.submit(
                        h, child_node, child_node.game.first_node, child_node.game.nodes
                    )
                    for child_node in current.get_children()
                ]

            for fut in futures:
                res = fut.result()
                to_expand.put((((1 - w) * res[1].cost + w * res[0]) * 2, res[1]))


def get_solution(current_node: SearchTreeNode, root: SearchTreeNode):
    solution = []
    solution = [current_node.game.first_node.color, *solution]
    current_node = current_node.parent
    while current_node != root and current_node is not None:
        solution = [current_node.game.first_node.color, *solution]
        current_node = current_node.parent
    return solution


def a_star(root: SearchTreeNode, h):
    return hpa_thread(root, h, 0.5)


def greedy(root: SearchTreeNode, h):
    return hpa_thread(root, h, 1)


def generic_search(root: SearchTreeNode, pending_nodes):
    pending_nodes.enqueue(root)
    expanded_nodes = 0

    while pending_nodes.__len__() > 0:
        current_node = pending_nodes.dequeue()
        expanded_nodes += 1

        if current_node.is_goal():
            # print("Expanded nodes: " + expanded_nodes.__str__())
            return get_solution(current_node, root), expanded_nodes

        for child_node in current_node.get_children():
            pending_nodes.enqueue(child_node)

    return None, None



def dls(root: SearchTreeNode, max_dist):
    pending_nodes = Stack()
    pending_nodes.enqueue(root)
    expanded_nodes = 0

    while pending_nodes.__len__() > 0:
        current_node = pending_nodes.dequeue()
        expanded_nodes += 1

        if current_node.is_goal():
            # print("Expanded nodes: " + expanded_nodes.__str__())
            return get_solution(current_node, root), expanded_nodes

        if current_node.cost >= max_dist:
            continue

        for child_node in current_node.get_children():
            pending_nodes.enqueue(child_node)

    return None, None


def iddfs(root: SearchTreeNode, h):
    res = None
    exp = -1
    b = 50
    a = 0
    while dls(root, b) is None:
        b *= 2
        a = b/2
    t = round((a + b) / 2)
    while (a != b and b != t) or res is None:
        res, exp = dls(root, t)
        if res is None:
            a = t + 1
        else:
            b = t
        t = math.ceil((a + b) / 2)
    return res, exp


def dfs(root: SearchTreeNode, h):
    return generic_search(root, Stack())


def bfs(root: SearchTreeNode, h):
    return generic_search(root, Queue())
