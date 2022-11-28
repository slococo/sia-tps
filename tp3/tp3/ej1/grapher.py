import numpy as np
import matplotlib

matplotlib.use("TkAgg")

import matplotlib.pyplot as plt

delta_y = 0.01
delta_x = 0.005


class Grapher:
    @classmethod
    def graph_data(cls, data):
        plt.figure(figsize=(14, 9))

        for c in data:
            aux = []
            for j in c:
                aux2 = np.unpackbits(np.array(int(j, 16), dtype=np.uint8))[3:]
                aux2 = aux2
                if np.atleast_2d(aux).shape[1] == 0:
                    aux = aux2
                else:
                    aux = np.concatenate(
                        (np.atleast_2d(aux), np.atleast_2d(aux2)), axis=0
                    )

            aux = np.subtract(1, aux)
            plt.imshow(aux, interpolation="nearest", cmap="gray")
            plt.show()

    @classmethod
    def graph_in_out(cls, in_data, out):
        Grapher.graph_in_out_generic(in_data, out, 5, 7)

    @classmethod
    def graph_in_out_generic(cls, in_data, out, x, y):
        plt.figure(figsize=(14, 9))

        in_data = in_data[1:]

        aux = []
        for j in range(0, y):
            aux2 = []
            aux3 = []
            for i in range(0, x):
                aux2 = np.concatenate((aux2, [in_data[j * x + i]]))
                a = out[j * x + i]
                if a <= 0:
                    a = -1
                else:
                    a = 1
                aux3 = np.concatenate((aux3, [a]), axis=0)

            aux4 = np.concatenate((aux2, [-1], aux3))
            if not j:
                aux = aux4
            else:
                aux = np.concatenate((np.atleast_2d(aux), np.atleast_2d(aux4)), axis=0)
        aux = np.subtract(1, aux)
        plt.imshow(aux, interpolation="nearest", cmap="gray")
        plt.show()

    @classmethod
    def graph_latent(cls, autoencoder, data, tags):
        plt.figure(figsize=(14, 9))
        latents = []
        for i in range(0, len(data)):
            latents.append((tags[i], autoencoder.get_latent_layer(data[i])))

        for point in latents:
            plt.scatter(point[1][0], point[1][1])
            plt.annotate(point[0], (point[1][0] + delta_x, point[1][1] + delta_y))

        plt.show()

    @classmethod
    def graph_char(cls, data):
        plt.figure(figsize=(14, 9))
        aux = []
        for j in range(0, 7):
            aux2 = []
            for i in range(0, 5):
                aux2 = np.concatenate((aux2, [data[j * 5 + i]]))

            if not j:
                aux = aux2
            else:
                aux = np.concatenate((np.atleast_2d(aux), np.atleast_2d(aux2)), axis=0)

        aux = np.subtract(1, aux)
        plt.imshow(aux, interpolation="nearest", cmap="gray")
        plt.show()

    @classmethod
    def generate_and_graph(cls, perceptron):
        plt.figure(figsize=(14, 9))
        aux = []
        for i in np.arange(-1, 1, 0.05):
            aux2 = []
            for j in np.arange(-1, 1, 0.1):
                if j == -1:
                    aux2 = np.array(perceptron.create_from_latent([-i, j])).reshape(
                        7, 5
                    )
                else:
                    aux2 = np.concatenate(
                        (
                            aux2,
                            np.array(perceptron.create_from_latent([-i, j])).reshape(
                                7, 5
                            ),
                        ),
                        axis=0,
                    )

            if i == -1:
                aux = aux2
            else:
                aux = np.concatenate((np.atleast_2d(aux), aux2), axis=1)

        plt.imshow(aux, cmap="gray")
        plt.show()

    @classmethod
    def graph_chars(cls, data, x, y, line_size):
        plt.figure(figsize=(14, 9))
        aux = []
        test = None
        for k in range(0, len(data)):
            if not k % line_size:
                if test is not None:
                    if k == line_size:
                        aux = test
                    else:
                        aux = np.insert(aux, aux.shape[0], -1, axis=0)
                        aux = np.concatenate((aux, test), axis=0)
                test = np.array(data[k]).reshape(y, x)
            else:
                test = np.insert(test, test.shape[1], -1, axis=1)
                test = np.concatenate((test, np.array(data[k]).reshape(y, x)), axis=1)

        aux = np.insert(aux, aux.shape[0], -1, axis=0)
        aux = np.concatenate((aux, test), axis=0)
        aux = np.subtract(1, aux)
        plt.imshow(aux, interpolation="nearest", cmap="gray")
        plt.show()
