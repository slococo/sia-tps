import csv

import numpy as np


class Transformer:
    @classmethod
    def transform_data(cls, data):
        Transformer.transform_data_generic(data, 5, "fonts.csv")

    @classmethod
    def transform_data_generic(cls, data, byte_size, file_name):
        if byte_size > 8 or byte_size <= 0:
            raise "your hands"
        transformed = []
        for c in data:
            aux = []
            for j in c:
                aux2 = np.unpackbits(np.array(int(j, 16), dtype=np.uint8))[
                    8 - byte_size :
                ]
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

        with open(file_name, "w") as file:
            writer = csv.writer(file)
            for dat in transformed:
                writer.writerow(dat)
