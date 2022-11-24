from tp3 import utils
from tp3.ej1.grapher import Grapher
from tp3.loader import CSVLoader
from tp3.perceptron import Perceptron
from tp3.tester import Tester


def main():
    perceptron = Perceptron.load("perceptron_200_35-25-17-2.obj")

    data_column = ["x" + str(i) for i in range(1, 36)]
    data, exp = CSVLoader.load("../fonts.csv", False, data_column, None, False)

    # Grapher.graph_char(perceptron.create_from_latent([.04, .05]))
    # Grapher.generate_and_graph(perceptron)

    Grapher.graph_chars(data[:, 1:], 5, 7, 8)

    predicts = []
    for i in range(0, len(data)):
        predicts.append(perceptron.predict(data[i]))

    predict_error = Tester.test(
        perceptron, data, exp, utils.quadratic_error, res_fun=None
    )
    print(predict_error)

    Grapher.graph_chars(predicts, 5, 7, 8)


if __name__ == "__main__":
    main()
