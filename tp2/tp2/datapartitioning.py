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

    train_size = int(len(dataset) * training_probability) + 1
    historic, errors = [], []

    historic.append([])
    errors.append([])

    historic[0], errors[0], _ = perceptron.train(
        dataset[: train_size],
        error,
        max_iter,
        learning,
    )

    if dims[-1] > 1:
        expected = np.zeros(shape=(1, len(dataset) - train_size, dims[-1]))
        results = np.zeros(shape=(1, len(dataset) - train_size, dims[-1]))
    else:
        results = np.zeros(shape=(1, len(dataset) - train_size))
        expected = np.zeros(shape=(1, len(dataset) - train_size))

    results[0] = np.squeeze(perceptron.predict(
        dataset[train_size:, :-dims[-1]]
    ))

    expected[0] = np.squeeze(dataset[train_size:, -dims[-1]:])

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
    np.random.shuffle(dataset)

    partition_size = int(len(dataset) / k)

    if partition_size * k == len(dataset):
        partitioned_dataset = dataset.reshape(k, partition_size, data_dim + dims[-1])
    else:
        partitioned_dataset = dataset[: partition_size * k - len(dataset)].reshape(
            k, partition_size, data_dim + dims[-1]
        )

    if dims[-1] > 1:
        expected = np.zeros(shape=(k, partition_size, dims[-1]))
        results = np.zeros(shape=(k, partition_size, dims[-1]))
    else:
        results = np.zeros(shape=(k, partition_size))
        expected = np.zeros(shape=(k, partition_size))

    historic, errors = [], []

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

        historic.append([])
        errors.append([])

        historic[i], errors[i], _ = perceptron.train(
            np.delete(partitioned_dataset, i, 0).reshape((k - 1) * partition_size, data_dim + dims[-1]),
            error,
            max_iter,
            learning,
        )

        results[i] = np.squeeze(perceptron.predict(np.delete(partitioned_dataset[i], 0, 1)))

        expected[i] = np.squeeze(partitioned_dataset[i, :, -dims[-1]:])

    return historic, errors, TestingResult(expected, results)
