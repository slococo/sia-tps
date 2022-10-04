import numpy as np
from perceptron import Perceptron


class TestingResult:
    def __init__(self, expected, results):
        # Se asume que cada metodo trabajar con n perceptrones y cada uno de estos sera testeado con m_i valores
        # Esa data se guarda en results y expected
        expected = expected
        results = results


def holdout(dataset, training_probability, activation_function, eta, neurone_matrix, error, max_iter, learning):
    dataset = np.array(dataset)
    np.random.shuffle(dataset)

    perceptron = Perceptron(
        neurone_matrix, None, activation_function, activation_function, len(neurone_matrix) + 1, eta
    )

    perceptron.train(dataset[:int(len(dataset)*training_probability)+1], error, max_iter, learning)

    results = perceptron.predict(dataset[int(len(dataset)*training_probability)+1:, :-1])

    expected = np.squeeze(dataset[int(len(dataset)*training_probability)+1:, -1:])

    testing_result = TestingResult(expected, results)

    return testing_result


def k_fold(dataset, k, activation_function, eta, neurone_matrix, error, max_iter, learning):
    dataset = np.array(dataset)
    partition_size = int(len(dataset)/k)

    if partition_size*k == len(dataset):
        partitioned_dataset = dataset.reshape(k, partition_size, 4)
    else:
        partitioned_dataset = dataset[:partition_size*k-len(dataset)].reshape(k, partition_size, 4)

    testing_result = TestingResult()

    for i in range(k):
        perceptron = Perceptron(
            neurone_matrix, None, activation_function, activation_function, len(neurone_matrix) + 1, eta
        )

        perceptron.train(np.delete(partitioned_dataset, i, 0).reshape((k-1)*partition_size, 4), error, max_iter, learning)

        results = perceptron.predict(partitioned_dataset[i])

        expected = np.squeeze(partitioned_dataset[i, :, -1:])


    return testing_result
