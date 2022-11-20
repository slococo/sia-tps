import time

import matplotlib
import numpy as np

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


class Grapher:
    @classmethod
    def graph_data(cls, data):
        fig = plt.figure(figsize=(14, 9))

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
        fig = plt.figure(figsize=(14, 9))

        in_data = in_data[1:]

        aux = []
        for j in range(0, 7):
            aux2 = []
            aux3 = []
            for i in range(0, 5):
                aux2 = np.concatenate((aux2, [in_data[j * 5 + i]]))
                a = out[j * 5 + i]
                # if a <= 0:
                #     a = -1
                # else:
                #     a = 1
                aux3 = np.concatenate((aux3, [a]), axis=0)
                # aux3 = np.concatenate((aux3, [out[j * 5 + i]]), axis=0)

            aux4 = np.concatenate((aux2, aux3))
            if not j:
                aux = aux4
            else:
                aux = np.concatenate((np.atleast_2d(aux), np.atleast_2d(aux4)), axis=0)

        aux = np.subtract(1, aux)
        plt.imshow(aux, interpolation="nearest", cmap="gray")
        plt.show()
