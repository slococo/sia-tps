import time

import matplotlib
import numpy as np

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


class Grapher:
    @classmethod
    def graph_data(cls, numbers):
        data = []
        color = np.array([1, 1, 1])
        fig = plt.figure(figsize=(14, 9))

        for j in range(0, 6):
            aux = []
            for i in range(0, 7):
                aux2 = []
                for k in range(0, 10):
                    aux2 = np.concatenate(
                        (aux2, numbers[(j * 10) + k][i * 5 : (i + 1) * 5]), 0
                    )
                if len(aux) == 0:
                    aux = [aux2]
                else:
                    aux = np.concatenate((aux, np.atleast_2d(aux2)), 0)
            if j != 0:
                data = np.concatenate((data, np.full_like(np.atleast_2d(aux2), 1)), 0)
            if np.atleast_2d(data).shape[1] == 0:
                data = aux
            else:
                data = np.concatenate((data, aux), 0)

        final = []
        for data_aux in data:
            final_aux = []
            for aux_aux in data_aux:
                final_aux.append(color * aux_aux)
            final.append(final_aux)

        plt.imshow(final, interpolation="nearest")
        fig.savefig("noisy-numbers" + round(time.time()).__str__() + ".png")
        plt.close()
