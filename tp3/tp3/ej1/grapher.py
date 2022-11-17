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
                    aux = np.concatenate((np.atleast_2d(aux), np.atleast_2d(aux2)), axis=0)

            aux = np.subtract(1, aux)
            plt.imshow(aux, interpolation="nearest", cmap="gray")
            plt.show()
