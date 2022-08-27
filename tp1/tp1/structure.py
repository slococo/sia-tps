from game import Game


class Node:
    def __init__(self, color: int, frontier: set | None):
        self.color = color
        self.frontier = frontier

    def absorb(self, node: "Node"):
        if self.frontier is not None:
            self.frontier.remove(node)
            self.frontier.add(node.frontier)
        else:
            self.frontier = node.frontier


def parse_game(game: Game) -> Node:
    matrix = game.matrix
    size = game.size
    init_node = Node(matrix[0][0], None)

    current_node = init_node
    current_x = 0
    current_y = 0





