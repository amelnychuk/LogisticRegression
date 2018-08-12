from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

N = 100
D = 2

N_per_class = N//2

X = np.random.randn(N, D)

X[:N_per_class, :] = X[:N_per_class, :] - 2*np.ones((N_per_class, D))
X[N_per_class:, :] = X[N_per_class:, :] + 2*np.ones((N_per_class, D))

T = np.array([0]*N_per_class + [1]*N_per_class)

ones = np.ones((N,1))
Xb = np.concatenate((ones, X), axis=1)



def sigmoid(z):
    return 1 / (1 + np.exp(-z))



def cross_entropy(T, Y):
    return -np.sum(T*np.log(Y) + (1-T)*np.log(1-Y))
w = np.random.randn(D+1) / pow((D+1),(D*D)+1)
#w /= np.mean(w)

print ("Init w:", w)

z = Xb.dot(w)
Y = sigmoid(z)
learning_rate = .01
for i in range(1000):
    if i % 100 == 0:
        print(cross_entropy(T, Y))

    w += learning_rate * (Xb.T.dot(T-Y) - (.01 * w))
    Y = sigmoid(Xb.dot(w))

print ("Final w: ", w)

plt.scatter(X[:,0], X[:,1], c=T, s=100, alpha=0.5)
x_axis = np.linspace(-6, 6, 100)
y_axis = -(w[0] + x_axis*w[1]) / w[2]
plt.plot(x_axis, y_axis)
plt.show()