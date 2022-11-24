import pickle


class Wrapper:
    def __init__(self, perceptron, data, historic, errors):
        self.perceptron = perceptron
        self.data = data
        self.historic = historic
        self.errors = errors

    def save(self, file_name=None):
        if file_name is None:
            file_name = "wrapper.obj"
        with open(file_name, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, file_name=None):
        if file_name is None:
            file_name = "wrapper_200_35_15_2..obj"
            # file_name = "wrapper.obj"
        with open(file_name, "rb") as f:
            return pickle.load(f)
