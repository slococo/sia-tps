# from game import Game


class Cell:
    def __init__(self, color: int):
        self.color = color
        self.node = None

    def __str__(self):
        return self.color.__str__()


class Matrix:
    def __init__(self, size):
        self.matrix = [[0] * size for _ in range(size)]
        self.size = size

    def assign(self, n, m, value: Cell):
        self.matrix[n][m] = value

    def retrieve(self, n, m):
        return self.matrix[n][m]

    def __getitem__(self, tup):
        return self.matrix[tup]

    def __str__(self):
        string = ''
        for i in range(self.size):
            for j in range(self.size):
                string += self.matrix[i][j].__str__()
                string += '\t'
            string += '\n'
        return string


class Node:
    def __init__(self, color: int):
        self.color = color
        self.frontier = set()

    # def absorb(self, node: "Node"):
    #     if self.frontier is not None:
    #         self.frontier.remove(node)
    #         self.frontier.add(node.frontier)
    #     else:
    #         self.frontier = node.frontier

    def absorb(self, color):
        self.color = color
        for node in self.frontier.copy():
            if node.color == color:
                self.frontier.remove(node)
                for aux in node.frontier:
                    self.frontier.add(aux)
                    aux.frontier.discard(node)

    def __str__(self):
        return self.color.__str__()

# def parse_game(game: Game) -> Node:
#     matrix = game.matrix
#     size = game.size
#     init_node = Node(matrix[0][0], None)
#
#     current_node = init_node
#     current_x = 0
#     current_y = 0



