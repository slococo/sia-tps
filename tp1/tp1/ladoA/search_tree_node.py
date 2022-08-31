from copy import deepcopy

from tp1.ladoA.game import Game


class SearchTreeNode:
    def __init__(self, game: Game, parent):
        self.game = game
        self.parent = parent
        self.children = []
        if parent is None:
            self.cost = 0
        else:
            self.cost = parent.get_cost() + 1

    def is_goal(self):
        return self.game.is_goal()

    def get_cost(self):
        return self.cost

    def get_game(self):
        return self.game

    def get_parent(self):
        return self.parent

    def get_children(self):
        if self.children.__len__() == 0:
            for i in self.game.get_frontier_colors():
                child_game = deepcopy(self.game)
                child_game.first_node.absorb(i)
                self.children.append(SearchTreeNode(child_game, self))
        print(len(self.children))
        return self.children
