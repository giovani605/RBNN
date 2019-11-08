import numpy as np


class RadialNeuron:
    def __init__(self, n_inputs):
        self.__center = np.random.random(n_inputs)
        self.__sigma = (np.random.random(n_inputs) + 1)

    def think(self, inputs):
        return np.exp(np.sum(-((self.__center - inputs)**2)/self.__sigma))

    @property
    def sigma(self):
        return self.__sigma

    @sigma.setter
    def sigma(self, val):
        self.__sigma = np.asarray(val)

    @property
    def center(self):
        return self.__center

    @center.setter
    def center(self, val):
        self.__center = np.asarray(val)
