import rbnn
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import random
import math
import numpy as np
from sklearn.metrics import accuracy_score


def softmax(v):
    ns = list(map(lambda x: math.exp(x), v))
    d = sum(ns)
    return np.asarray([n/d for n in ns])


iris_in, iris_o = datasets.load_iris(True)
iris_out = list(map(lambda x:
                    [1, 0, 0] if x == 0 else
                    [0, 1, 0] if x == 1 else
                    [0, 0, 1], iris_o))

train_X, test_X, train_y, test_y = train_test_split(
    iris_in, iris_out, test_size=0.25, random_state=21)


print('RBF')
for n in range(2, 10):
    nn = rbnn.RBNN(4, n, 3)
    nn.train(train_X, train_y, 0.5)
    total = 0
    acertos = 0

    for i, o in zip(test_X, test_y):
        pred = nn.think(i)
        pred = np.where(pred == pred.max())[0][0]
        expt = o.index(max(o))

        # print(f"Entrada: {i}")
        # print(f"Esperado: {expt}")
        # print(f"Saida: {pred}\n")
        total += 1
        if(expt == pred):
            acertos += 1

    print('Resultado de ', n, ' neuronios')
    print('total de acertos ', acertos/total)

train_X, test_X, train_y, test_y = train_test_split(
    iris_in, iris_o, test_size=0.25, random_state=21)


print('MLP')
resultadoAnterior = 0
melhorNeuronios = 0
melhorCamada = 0
for n in range(100):
    neuronios = random.randint(2, 10)
    camadas = random.randint(2, 10)
    nn = MLPClassifier(hidden_layer_sizes=np.full(
        camadas, neuronios), max_iter=1000)
    nn.fit(train_X, train_y)
    pred = nn.predict(test_X)
    total = 0
    acertos = 0
    acc = accuracy_score(test_y, pred)
    if acc > resultadoAnterior:
        resultadoAnterior = acc
        melhorCamada = camadas
        melhorNeuronios = neuronios

print('Resultado de ', melhorNeuronios,
      ' neuronios e ', melhorCamada, ' camadas')
print('total de acertos ', resultadoAnterior)
