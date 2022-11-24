import numpy as np


class Tester:
    @classmethod
    def test(cls, perceptron, data, exp, error, exp_converter=None, res_fun=None):
        test_error = 0
        exp = np.squeeze(exp)
        for i in range(0, len(data)):
            if exp_converter:
                pred = perceptron.predict(data[i])
                test_error += error(exp_converter(pred, exp[i]), pred)
            elif res_fun:
                pred = perceptron.predict(data[i])
                test_error += error(exp[i], res_fun(pred, None))
            else:
                test_error += error(exp[i], perceptron.predict(data[i]))

        return test_error / len(data)
