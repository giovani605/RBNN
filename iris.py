import rbnn
from sklearn import datasets
from sklearn.model_selection import train_test_split
import random
import math
import numpy as np


def softmax(v):
    ns = list(map(lambda x: math.exp(x), v))
    d = sum(ns)
    return np.asarray([n/d for n in ns])


iris_in, iris_o = datasets.load_iris(True)
iris_out = list(map(lambda x:
                    [1, 0, 0] if x == 0 else
                    [0, 1, 0] if x == 1 else
                    [0, 0, 1], iris_o))

train_X, test_X, train_y, test_y = train_test_split(iris_in, iris_out, test_size=0.25)

nn = rbnn.RBNN(4, 5, 3)
nn.train(train_X, train_y, 0.5)

for i, o in zip(test_X, test_y):
    pred = nn.think(i)
    pred = np.where(pred == pred.max())[0][0]
    expt = o.index(max(o))

    print(f"Entrada: {i}")
    print(f"Esperado: {expt}")
    print(f"Saida: {pred}\n")
