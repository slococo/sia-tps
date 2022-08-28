import numpy as np
from structure import Node, Matrix, Cell


# TODO: hacerla iterativa
def create_graph(matrix, i, j, size, current_node=None) -> Node:
    color = matrix[i][j].color
    if matrix[i][j].node is not None:
        return matrix[i][j].node
    if current_node is None:
        current_node = Node(color)

    matrix[i][j].node = current_node

    if i > 0:
        if matrix[i - 1][j].color == color:
            create_graph(matrix, i - 1, j, size, current_node)
        else:
            current_node.frontier.add(create_graph(matrix, i - 1, j, size))

    if j > 0:
        if matrix[i][j - 1].color == color:
            create_graph(matrix, j - 1, j, size, current_node)
        else:
            current_node.frontier.add(create_graph(matrix, i, j - 1, size))

    if i < size - 1:
        if matrix[i + 1][j].color == color:
            create_graph(matrix, i + 1, j, size, current_node)
        else:
            current_node.frontier.add(create_graph(matrix, i + 1, j, size))

    if j < size - 1:
        if matrix[i][j + 1].color == color:
            create_graph(matrix, i, j + 1, size, current_node)
        else:
            current_node.frontier.add(create_graph(matrix, i, j + 1, size))

    return matrix[i][j].node


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
        self.first_node = create_graph(self.matrix, 0, 0, size)

    def shuffle(self):
        for i in range(self.size):
            for j in range(self.size):
                cell = Cell(np.random.randint(1, self.num_colors + 1))
                self.matrix[i][j] = cell


def print_graph(node: Node, checked: set, level):
    if node not in checked:
        checked.add(node)
        print(node, end='')
    for aux in node.frontier:
        if aux not in checked:
            print('\n|' + '-' * level, end='')
            print(aux, end='')
            checked.add(aux)
            print_graph(aux, checked, level + 1)


game = Game(2134, 35)
matrix = game.matrix
print(matrix)
print('----------')
print_graph(game.first_node, set(), 1)
print()
print('---------------------------------------------')
game.first_node.absorb(2)
print_graph(game.first_node, set(), 1)

"""print('----------')
for i in range(9):
    for j in range(9):
        print(matrix[i][j].node)
print('----------')
for i in range(9):
    for j in range(9):
        print(matrix[i][j].node.frontier)
"""
