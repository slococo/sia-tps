from copy import deepcopy


class Cell:
    def __init__(self, color: int, i: int, j: int):
        self.color = color
        self.node = None
        self.i = i
        self.j = j

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
        string = ""
        for i in range(self.size):
            for j in range(self.size):
                string += self.matrix[i][j].__str__()
                string += "\t"
            string += "\n"
        return string

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memo))
        return result


class Node:
    def __init__(self, color: int, key=-1):
        self.key = key
        self.color = color
        self.frontier = set()

    def absorb(self, color, nodeset):
        self.color = color
        for node in self.frontier.copy():
            if node.color == self.color:
                for aux in node.frontier:
                    aux.frontier.remove(node)
                    if aux.key != self.key:
                        self.frontier.add(aux)
                        aux.frontier.add(self)
                nodeset.remove(node)

    def __str__(self):
        return self.color.__str__()

    def frontier_size(self):
        return self.frontier.__len__()

    def __eq__(self, other):
        return self.key == other.key

    def __ne__(self, other):
        return self.key != other.key

    def __gt__(self, other):
        return self.key > other.key

    def __hash__(self) -> int:
        return hash(self.key)

    def __cmp__(self, other):
        return self.key.__cmp__(other.key)

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memo))
        return result
