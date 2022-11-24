from tp3.ej1.grapher import Grapher
from tp3.loader import CSVLoader
from tp3.perceptron import Perceptron


def main(data_path=None):
    if data_path is None:
        data_path = "tp3/ej1/fonts.csv"

    perceptron = Perceptron.load("perceptron_200_35-25-17-2.obj")

    data_column = ["x" + str(i) for i in range(1, 36)]
    data, exp = CSVLoader.load(data_path, False, data_column, None, False)

    chars = ["`", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t",
             "u", "v", "w", "x", "y", "z", "{", "|", "}", "~", "DEL"]
    Grapher.graph_latent(perceptron, data, chars)


if __name__ == "__main__":
    main("../fonts.csv")
