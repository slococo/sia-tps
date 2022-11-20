import csv

import numpy as np


class Transformer:
    @classmethod
    def transform_data(cls, data):
        transformed = []
        for c in data:
            aux = []
            for j in c:
                aux2 = np.unpackbits(np.array(int(j, 16), dtype=np.uint8))[3:]
                aux2 = aux2
                if np.atleast_2d(aux).shape[1] == 0:
                    aux = aux2
                else:
                    aux = np.concatenate(
                        (np.atleast_2d(aux), np.atleast_2d(aux2)), axis=1
                    )

            if np.atleast_2d(transformed).shape[1] == 0:
                transformed = aux
            else:
                transformed = np.concatenate(
                    (np.atleast_2d(transformed), np.atleast_2d(aux)), axis=0
                )

        with open("font_trans.csv", "w") as file:
            writer = csv.writer(file)
            for dat in transformed:
                writer.writerow(dat)
