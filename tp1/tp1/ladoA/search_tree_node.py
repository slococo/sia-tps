import copy

from tp1.ladoA.game import Game


class SearchTreeNode:
    def __init__(self, game: Game, parent):
        self.game = game
        self.parent = parent
        self.children = []
        if parent is None:
            self.cost = 0
        else:
            self.cost = parent.cost + 1

    def is_goal(self):
        return self.game.is_goal()

    def get_children(self):
        if self.children.__len__() == 0:
            for i in self.game.get_frontier_colors():
                child_game = copy.deepcopy(self.game)
                child_game.first_node.absorb(i, child_game.nodes)
                self.children.append(SearchTreeNode(child_game, self))
        return self.children

    def __lt__(self, other):
        return 1
