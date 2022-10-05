import pickle


class Wrapper:
    def __init__(self, perceptron, data, historic, learning):
        self.perceptron = perceptron
        self.data = data
        self.historic = historic
        self.learning = learning

    def save(self, file_name=None):
        if file_name is None:
            file_name = "wrapper.obj"
        with open(file_name, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, file_name=None):
        if file_name is None:
            file_name = "wrapper.obj"
        with open(file_name, "rb") as f:
            return pickle.load(f)
