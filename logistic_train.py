
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from trainingData import TrainingData
from logistic2 import sigmoid, cross_entropy

Data = TrainingData()

D = Data.X.train.shape[1]
W = np.random.randn(D)
b = 1

def forward(X, W, b):
    return sigmoid(X.dot(W) + b)

def classification_rate(Y, P):
    return np.mean(Y == P)

#train_loop
train_costs = []
test_costs = []
learning_rate = 0.001
for i in range(10000):
    pYtrain = forward(Data.X.train, W, b)
    pYtest = forward(Data.X.test, W, b)

    ctrain = cross_entropy(Data.Y.train, pYtrain)
    ctest = cross_entropy(Data.Y.test, pYtest)
    train_costs.append(ctrain)
    test_costs.append(ctest)

    #grad descent
    W -= learning_rate * Data.X.train.T.dot(pYtrain-Data.Y.train)
    b -= learning_rate * (pYtrain - Data.Y.train).sum()
    if i % 1000 == 0:
        print(i, ctrain, ctest)


print("Final train classification_rate: ", classification_rate(Data.Y.train, np.round(pYtrain)))
print("Final test classification_rate: ", classification_rate(Data.Y.test, np.round(pYtest)))

legend1, = plt.plot(train_costs, label='train cost')
legend2, = plt.plot(test_costs, label='test cost')
plt.legend([legend1, legend2])
plt.show()