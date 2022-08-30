# La idea seria tener un objeto que represente un estado del tablero
# Este tiene que saber su costo y tiene que saber como generar los estados que le suceden
# Estos nodos son wrappers de los estados y conforman el arbo de busqueda

# Hay que ver como generar y guardar los estados para no usar memoria al pedo
# ya que mas de un nodo puede tener el mismo estado
class SearchTreeNode:
    def __init__(self, state, parent):
        self.state = state
        self.parent = parent
        self.children = []
        self.cost = parent.get_cost + 1

    def is_goal(self):
        return self.state.is_goal()

    def get_cost(self):
        return self.cost

    def get_state(self):
        return self.state

    def get_parent(self):
        return self.parent

    def get_children(self):
        if len(self.children) == 0:
            child_states = self.state.get_children
            for child_state in child_states:
                self.children.append(SearchTreeNode(child_state, self))
        return self.children
