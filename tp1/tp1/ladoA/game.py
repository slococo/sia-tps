import numpy as np
from tp1.ladoA.structure import Cell, Matrix, Node
from tp1.utils import Queue


def create_graph(matrix, size, nodeset: set):
    if size <= 0:
        return
    count = 0

    new_groups = Queue()
    group = Queue()
    first_node = Node(matrix[0][0].color, count)
    count += 1
    is_first = True
    new_groups.enqueue(matrix[0][0])

    while new_groups:
        new = new_groups.dequeue()
        if new.node is not None:
            continue
        if not is_first:
            node = Node(new.color, count)
            count += 1
        else:
            is_first = False
            node = first_node
        nodeset.add(node)
        group.enqueue(new)
        while group:
            current = group.dequeue()
            current.node = node

            if current.i > 0:
                enqueue_aux(
                    current, current.i - 1, current.j, matrix, group, new_groups
                )

            if current.j > 0:
                enqueue_aux(
                    current, current.i, current.j - 1, matrix, group, new_groups
                )

            if current.i < size - 1:
                enqueue_aux(
                    current, current.i + 1, current.j, matrix, group, new_groups
                )

            if current.j < size - 1:
                enqueue_aux(
                    current, current.i, current.j + 1, matrix, group, new_groups
                )

    return first_node


def enqueue_aux(current, i, j, matrix, groups_queue, new_groups_queue):
    if current.color == matrix[i][j].color:
        if matrix[i][j].node is None:
            groups_queue.enqueue(matrix[i][j])
    elif matrix[i][j].node is not None:
        matrix[i][j].node.frontier.add(current.node)
        current.node.frontier.add(matrix[i][j].node)
    else:
        new_groups_queue.enqueue(matrix[i][j])


class Game:
    def __init__(self, size: int, num_colors: int, matrix=None):
        if size < 0 or num_colors < 0:
            raise ValueError("Game must have positive size and amount of colors")
        self.size = size
        self.num_colors = num_colors
        self.nodes = set()
        if matrix is None:
            self.matrix = Matrix(size)
            for i in range(size):
                for j in range(size):
                    cell = Cell(np.random.randint(1, self.num_colors + 1), i, j)
                    self.matrix[i][j] = cell
        else:
            self.matrix = matrix
        print(self.matrix)
        self.first_node = create_graph(self.matrix, size, self.nodes)
        colors = [0] * num_colors
        for i in range(1, num_colors + 1):
            for node in self.nodes:
                if node.color == i:
                    colors[i - 1] += 1
        print(self.nodes)
        self.matrix = None

    def shuffle(self):
        for i in range(self.size):
            for j in range(self.size):
                cell = Cell(np.random.randint(1, self.num_colors + 1), i, j)
                self.matrix[i][j] = cell

    def is_goal(self):
        return len(self.get_frontier_colors()) == 0

    def get_frontier_colors(self):
        frontier_color = set()
        for node in self.first_node.frontier:
            frontier_color.add(node.color)
        return frontier_color

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        new_nodes = {}

        for node in self.nodes:
            new_nodes[node.key] = Node(node.color, node.key)

        for node in self.nodes:
            new_node = new_nodes.get(node.key)
            new_node.frontier = set()

            for front in node.frontier:
                aux = new_nodes.get(front.key)
                new_node.frontier.add(aux)

        result.nodes = set()
        for k, v in new_nodes.items():
            result.nodes.add(v)

        result.first_node = new_nodes.get(self.first_node.key)
        result.size = self.size
        result.key = self.key + 1

        return result


def print_graph(node: Node, checked: set, level):
    if node not in checked:
        checked.add(node)
        print(node, end="")
    for aux in node.frontier:
        if aux not in checked:
            print("\n|" + "-" * level, end="")
            print(aux, end="")
            checked.add(aux)
            print_graph(aux, checked, level + 1)
