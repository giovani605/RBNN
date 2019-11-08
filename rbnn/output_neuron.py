import numpy as np


class OutputNeuron:
    def __init__(self, n_inputs):
        self.__weights = np.random.random(n_inputs)

    def think(self, inputs):
        return inputs.dot(self.__weights)

    @property
    def weights(self):
        return self.__weights

    @weights.setter
    def weights(self, val):
        self.__weights = np.asarray(val)
