import numpy as np
from tp2.perceptron import Perceptron


class TestingResult:
    def __init__(self, expected, results):
        self.expected = expected
        self.results = results

    def get_expected(self):
        return self.expected

    def get_results(self):
        return self.results


def holdout(
    data_dim,
    dims,
    optimizer,
    g,
    g_diff,
    eta,
    eta_adapt,
    training_probability,
    dataset,
    error,
    max_iter,
    learning
):

    perceptron = Perceptron(
        data_dim,
        dims,
        optimizer,
        g,
        g_diff,
        eta,
        eta_adapt
    )

    dataset = np.array(dataset)
    np.random.shuffle(dataset)

    train_size = int(len(dataset) * training_probability)

    historic, errors, _ = perceptron.train(
        dataset[: train_size + 1],
        error,
        max_iter,
        learning,
    )

    results = np.zeros(shape=(1, len(dataset) - train_size - 1))
    expected = np.zeros(shape=(1, len(dataset) - train_size - 1))

    # print(dataset)
    # print(dataset[train_size + 1:, :-1])
    results[0] = perceptron.predict(
        dataset[train_size + 1:, :-1]
    )

    expected[0] = np.squeeze(dataset[train_size + 1:, -1:])

    return historic, errors, TestingResult(expected, results)


def k_fold(
    data_dim,
    dims,
    optimizer,
    g,
    g_diff,
    eta,
    eta_adapt,
    k,
    dataset,
    error,
    max_iter,
    learning
):
    dataset = np.array(dataset)
    partition_size = int(len(dataset) / k)

    if partition_size * k == len(dataset):
        partitioned_dataset = dataset.reshape(k, partition_size, 5)
    else:
        partitioned_dataset = dataset[: partition_size * k - len(dataset)].reshape(
            k, partition_size, 5
        )

    results = np.zeros(shape=(k, partition_size))
    expected = np.zeros(shape=(k, partition_size))

    historic, errors = 0, 0

    for i in range(k):
        perceptron = Perceptron(
            data_dim,
            dims,
            optimizer,
            g,
            g_diff,
            eta,
            eta_adapt
        )

        historic, errors, _ = perceptron.train(
            np.delete(partitioned_dataset, i, 0).reshape((k - 1) * partition_size, 5),
            error,
            max_iter,
            learning,
        )

        results[i] = perceptron.predict(np.delete(partitioned_dataset[i], 0, 2))

        expected[i] = np.squeeze(partitioned_dataset[i, :, -1:])

    return historic, errors, TestingResult(expected, results)
