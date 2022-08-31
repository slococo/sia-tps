import numpy as np

from tp1.ladoA.structure import Cell, Matrix, Node


def create_graph(matrix, i, j, size, nodeset: set, current_node=None):
    color = matrix[i][j].color
    if matrix[i][j].node is not None:
        return matrix[i][j].node, nodeset
    if current_node is None:
        current_node = Node(color)
        nodeset.add(current_node)

    matrix[i][j].node = current_node

    if i > 0:
        if matrix[i - 1][j].color == color:
            create_graph(matrix, i - 1, j, size, nodeset, current_node)
        else:
            new_node, nodeset = create_graph(matrix, i - 1, j, size, nodeset)
            current_node.frontier.add(new_node)

    if j > 0:
        if matrix[i][j - 1].color == color:
            create_graph(matrix, i, j - 1, size, nodeset, current_node)
        else:
            new_node, nodeset = create_graph(matrix, i, j - 1, size, nodeset)
            current_node.frontier.add(new_node)

    if i < size - 1:
        if matrix[i + 1][j].color == color:
            create_graph(matrix, i + 1, j, size, nodeset, current_node)
        else:
            new_node, nodeset = create_graph(matrix, i + 1, j, size, nodeset)
            current_node.frontier.add(new_node)

    if j < size - 1:
        if matrix[i][j + 1].color == color:
            create_graph(matrix, i, j + 1, size, nodeset, current_node)
        else:
            new_node, nodeset = create_graph(matrix, i, j + 1, size, nodeset)
            current_node.frontier.add(new_node)

    return [matrix[i][j].node, nodeset]


class Game:
    def __init__(self, size: int, num_colors: int):
        if size < 0 or num_colors < 0:
            raise ValueError("Game must have positive size and amount of colors")
        self.size = size
        self.num_colors = num_colors
        self.matrix = Matrix(size)
        for i in range(size):
            for j in range(size):
                cell = Cell(np.random.randint(1, self.num_colors + 1))
                self.matrix[i][j] = cell
        self.first_node, self.nodes = create_graph(self.matrix, 0, 0, size, set())

    def shuffle(self):
        for i in range(self.size):
            for j in range(self.size):
                cell = Cell(np.random.randint(1, self.num_colors + 1))
                self.matrix[i][j] = cell

    def is_goal(self):
        return len(self.get_frontier_colors()) == 0

    def get_frontier_colors(self):
        frontier_color = set()
        for node in self.first_node.get_frontier():
            frontier_color.add(node.color)
        return frontier_color


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
