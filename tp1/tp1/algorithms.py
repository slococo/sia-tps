from structure import Node
from collections import deque
from heapq import heappop, heappush
from itertools import count


# Asumimos que dado un "goal node" n la heuristica cumple que h(n)=0
# Asumimos que el grafo tiene nodo inicial
def a_star(root, h):
    pending_nodes = PriorityQueue()
    root.enqueue_with_priority(root.getCost()+h(root), root)
    while pending_nodes.__len__() > 0:
        current_node = pending_nodes.dequeue()
        # El nodo es goal entonces recorro el path hasta el origen y devuelve la sucesion de colores
        if h(current_node) == 0:
            return 0

        for child_node in current_node.getChildren:
            pending_nodes.enqueue_with_priority(child_node.get_cost + h(child_node), child_node)

    return 0


def greedy():
    return 0


def dfs():
    return 0


def bfs():
    return 0


# https://realpython.com/queue-in-python/#building-a-priority-queue-data-type
class PriorityQueue:
    def __init__(self):
        self._elements = []
        self._counter = count()

    def enqueue_with_priority(self, priority, value):
        element = (-priority, next(self._counter), value)
        heappush(self._elements, element)

    def dequeue(self):
        return heappop(self._elements)[-1]

    def __len__(self):
        return len(self._elements)

    def __iter__(self):
        while len(self) > 0:
            yield self.dequeue()
