import numpy as np
from tp2 import utils
from tp2.initializer import Initializer
from tp2.tester import Tester


def holdout(
    config_path,
    data_dim,
    dims,
    training_probability,
    dataset,
):

    historic, errors = [], []
    historic.append([])
    errors.append([])

    perceptron, max_iter, errors[0], learning, eta, _ = Initializer.initialize(
        config_path, dims, data_dim - 1
    )

    dataset = np.array(dataset)
    np.random.shuffle(dataset)

    train_size = int(len(dataset) * training_probability) + 1

    historic[0], aux, layer_historic, epoch = perceptron.train(
        dataset[:train_size], errors[0], max_iter, learning
    )

    if dims[-1] > 1:
        expected = np.zeros(shape=(1, len(dataset) - train_size, dims[-1]))
        results = np.zeros(shape=(1, len(dataset) - train_size, dims[-1]))
    else:
        results = np.zeros(shape=(1, len(dataset) - train_size))
        expected = np.zeros(shape=(1, len(dataset) - train_size))

    results[0] = Tester.test(
        perceptron,
        dataset[train_size:, : -dims[-1]],
        dataset[train_size:, -dims[-1] :],
        utils.quadratic_error,
    )

    expected[0] = np.squeeze(dataset[train_size:, -dims[-1] :])

    return historic, errors, expected, results


def k_fold(
    config_path,
    data_dim,
    dims,
    k,
    dataset,
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

        historic.append([])
        errors.append([])

        perceptron, max_iter, errors[i], learning, eta, _ = Initializer.initialize(
            config_path, dims, data_dim - 1
        )

        historic[i], aux, layer_historic, epoch = perceptron.train(
            np.delete(partitioned_dataset, i, 0).reshape(
                (k - 1) * partition_size, data_dim + dims[-1]
            ),
            errors[i],
            max_iter,
            learning,
        )

        results[i] = Tester.test(
            perceptron,
            partitioned_dataset[i][:, : -dims[-1]],
            partitioned_dataset[i][:, -dims[-1] :],
            utils.quadratic_error,
        )

        expected[i] = np.squeeze(partitioned_dataset[i, :, -dims[-1] :])

    return historic, errors, expected, results


def k_fold_training(
    config_path,
    data_dim,
    dims,
    k,
    dataset,
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
        expected = np.zeros(shape=(k, 1, dims[-1]))
        results = np.zeros(shape=(k, 1, dims[-1]))
    else:
        results = np.zeros(shape=(k, 1))
        expected = np.zeros(shape=(k, 1))

    historic, errors, perceptrons = [], [], []

    for i in range(0, k - 1):

        historic.append([])
        errors.append([])

        perceptron, max_iter, errors[i], learning, eta, _ = Initializer.initialize(
            config_path, dims, data_dim - 1
        )
        perceptrons.append(perceptron)

        historic[i], aux, layer_historic, epoch = perceptron.train(
            np.delete(partitioned_dataset, i, 0).reshape(
                (k - 1) * partition_size, data_dim + dims[-1]
            ),
            errors[i],
            max_iter,
            learning,
        )

        results[i] = Tester.test(
            perceptron,
            partitioned_dataset[i][:, : -dims[-1]],
            partitioned_dataset[i][:, -dims[-1] :],
            utils.quadratic_error,
        )

        # expected[i] = np.squeeze(partitioned_dataset[i, :, -dims[-1]:])

    aux = np.subtract(1, np.squeeze(results))
    aux = np.divide(aux, np.max(aux))
    aux = np.array(aux[:-1])

    weights = []
    for i in perceptrons[0].matrix_arr:
        weights.append(np.zeros_like(i))

    for i in range(0, len(aux)):
        for j in range(0, len(weights)):
            weights[j] = np.add(
                weights[j], np.multiply(perceptrons[i].matrix_arr[j], aux[i])
            )

    perceptron, _, errors, _, _, _ = Initializer.initialize(
        config_path, dims, data_dim - 1
    )
    perceptron.matrix_arr = weights
    results = Tester.test(
        perceptron,
        partitioned_dataset[k - 1][:, :-1],
        partitioned_dataset[k - 1][:, -1],
        utils.quadratic_error,
    )

    return None, errors, None, results
