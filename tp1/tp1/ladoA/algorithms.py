from collections import deque
from heapq import heappop, heappush
from itertools import count

from tp1.ladoA.search_tree_node import SearchTreeNode


def a_star(root: SearchTreeNode, h):
    pending_nodes = PriorityQueue()
    pending_nodes.enqueue_with_priority(h(root.get_game().first_node, root.get_game().nodes), root)

    while pending_nodes.__len__() > 0:
        current_node = pending_nodes.dequeue()

        if current_node.is_goal():
            solution = []
            solution = [current_node.get_game().first_node.get_color(), *solution]
            current_node = current_node.get_parent()
            while current_node != root and current_node is not None:
                solution = [current_node.get_game().first_node.get_color(), *solution]
                current_node = current_node.get_parent()
            return solution

        for child_node in current_node.get_children():
            pending_nodes.enqueue_with_priority(
                child_node.get_cost() + h(child_node.get_game().first_node, root.get_game().nodes), child_node
            )

    return None


def greedy(root: SearchTreeNode, h):
    pending_nodes = PriorityQueue()
    pending_nodes.enqueue_with_priority(h(root), root)

    while pending_nodes.__len__() > 0:
        current_node = pending_nodes.dequeue()

        if current_node.is_goal():
            solution = []
            solution = [current_node.get_graph.get_color, *solution]
            while current_node != root:
                solution = [current_node.get_graph.get_color, *solution]
                current_node = current_node.get_parent
            return solution

        for child_node in current_node.get_children:
            pending_nodes.enqueue_with_priority(h(child_node), child_node)

    return 0


def dfs(root: SearchTreeNode):
    pending_nodes = Queue()
    pending_nodes.enqueue(root)

    while pending_nodes.__len__() > 0:
        current_node = pending_nodes.dequeue()

        if current_node.get_goal():
            solution = []
            solution = [current_node.get_graph.get_color, *solution]
            while current_node != root:
                solution = [current_node.get_graph.get_color, *solution]
                current_node = current_node.get_parent
            return solution

        for child_node in current_node.get_children:
            pending_nodes.enqueue(child_node)

    return 0


def bfs(root: SearchTreeNode):
    pending_nodes = Stack()
    pending_nodes.enqueue(root)

    while pending_nodes.__len__() > 0:
        current_node = pending_nodes.dequeue()

        if not current_node.get_goal():
            solution = []
            solution = [current_node.get_graph.get_color, *solution]
            while current_node != root:
                solution = [current_node.get_graph.get_color, *solution]
                current_node = current_node.get_parent
            return solution

        for child_node in current_node.get_children:
            pending_nodes.enqueue(child_node)

    return 0


class Queue:
    def __init__(self, *elements):
        self._elements = deque(elements)

    def __len__(self):
        return len(self._elements)

    def __iter__(self):
        while len(self) > 0:
            yield self.dequeue()

    def enqueue(self, element):
        self._elements.append(element)

    def dequeue(self):
        return self._elements.popleft()


class Stack(Queue):
    def dequeue(self):
        return self._elements.pop()


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
