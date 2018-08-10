from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
print("poopy1")
N = 100
D = 2

X = np.random.randn(N,D)

X[:50, :] = X[:50, :] - 2*np.ones((50, D))
X[50:, :] = X[50:, :] + 2*np.ones((50,D))

T = np.array([0]*50 + [1]*50)
ones = np.array([[1]*N]).T
Xb = np.concatenate((ones, X), axis=1)

w = np.random.randn(D + 1)

z = Xb.dot(w)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

Y = sigmoid(z)


def cross_entropy(T,Y):
    return -np.sum(T * np.log(Y) + (1-T)*np.log(1-Y))

print("derp")

print("Cross entropy:")
print(cross_entropy(T, Y))


w = np.array([0, 4, 4])

plt.scatter(X[:, 0], X[:, 1], c=T, s=100, alpha=1)
plt.show()
z= Xb.dot(w)
Y = sigmoid(z)

print("Closed form cross-entropy:")
print(cross_entropy(T, Y))