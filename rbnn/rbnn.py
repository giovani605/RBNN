import numpy as np
from .radial_neuron import RadialNeuron
from .output_neuron import OutputNeuron
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


class RBNN:
    def __init__(self, n_input, n_hidden, n_output):
        self._k = n_hidden
        self.__hidden = [RadialNeuron(n_inputs=n_input) for _ in range(n_hidden)]
        self.__output = [OutputNeuron(n_inputs=n_hidden) for _ in range(n_output)]

    def _think_hidden(self, inpt):
        if type(inpt) is not np.ndarray:
            inpt = np.asarray(inpt)
        output = []
        for hn in self.__hidden:
            output.append(hn.think(inpt))
        return np.asarray(output)

    def think(self, inpt):
        inpt = self._think_hidden(inpt)
        output = []
        for on in self.__output:
            output.append(on.think(inpt))
        return np.asarray(output)

    def train(self, inpts, outputs, err_min, val_size=0.1, max_up_before_stop=3):
        inpts = np.asarray(inpts)
        outputs = np.asarray(outputs)
        error = float('inf')
        up_cnt = 0
        train_X, Val_X, train_y, Val_y = train_test_split(inpts, outputs, test_size=val_size)
        while True:
            #  K-mean
            for i, c in enumerate(KMeans(n_clusters=self._k).fit(train_X).cluster_centers_):
                self.__hidden[i].center = c
            #  LR
            lr = LinearRegression()
            for i, output in enumerate(train_y.transpose()):
                lr.fit(np.asarray(list(map(lambda x: self._think_hidden(x), train_X))), output)
                self.__output[i].weights = lr.coef_
            # Validation
            prd = np.asarray(list(map(lambda x: self.think(x), Val_X)))
            loss = mean_squared_error(Val_y, prd)
            if loss <= err_min:
                return loss
            elif loss > error:
                up_cnt += 1
                if up_cnt > max_up_before_stop:
                    return loss
                error = loss
            else:
                error = loss
                continue

