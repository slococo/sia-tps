
class State:
    def __init__(self, frontier):
        self.frontier = frontier
        self.transitions = []

    def is_goal(self):
        return 0

    def get_transitions(self):
        return 0

    def get_frontier_colors(self):
        return 0
