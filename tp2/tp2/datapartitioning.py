import numpy as np
from perceptron import Perceptron


class TestingResult:
    def __init__(self, expected, results):
        self.expected = expected
        self.results = results


def holdout(
    dataset,
    training_probability,
    activation_function,
    eta,
    neuron_matrix,
    error,
    max_iter,
    learning,
):
    dataset = np.array(dataset)
    np.random.shuffle(dataset)

    perceptron = Perceptron(
        neuron_matrix, None, activation_function, activation_function, eta
    )

    perceptron.train(
        dataset[: int(len(dataset) * training_probability) + 1],
        error,
        max_iter,
        learning,
    )

    results = perceptron.predict(
        dataset[int(len(dataset) * training_probability) + 1 :, :-1]
    )

    expected = np.squeeze(dataset[int(len(dataset) * training_probability) + 1:, -1:])

    return TestingResult(expected, results)


def k_fold(
    dataset, k, activation_function, eta, neurone_matrix, error, max_iter, learning
):
    dataset = np.array(dataset)
    partition_size = int(len(dataset) / k)

    if partition_size * k == len(dataset):
        partitioned_dataset = dataset.reshape(k, partition_size, 4)
    else:
        partitioned_dataset = dataset[: partition_size * k - len(dataset)].reshape(
            k, partition_size, 4
        )

    results = np.zeros(shape=(k, partition_size))
    expected = np.zeros(shape=(k, partition_size))

    for i in range(k):
        perceptron = Perceptron(
            neurone_matrix, None, activation_function, activation_function, eta
        )

        perceptron.train(
            np.delete(partitioned_dataset, i, 0).reshape((k - 1) * partition_size, 4),
            error,
            max_iter,
            learning,
        )

        results[i] = perceptron.predict(partitioned_dataset[i])

        expected[i] = np.squeeze(partitioned_dataset[i, :, -1:])

    return TestingResult(expected, results)
