# 5.7.1 Data Structures and Functions in Numpy
import math
import numpy as np

W = np.array([[3, 4], [2, 3]])
b = np.array([-2, -4])
W2 = np.array([5, -5])
b2 = np.array([-2])

@np.vectorize
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

@np.vectorize
def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

x = np.array([1, 0])
t = np.array([1])

# 5.7.2 Forward Computation

s = W.dot(x) + b
h = sigmoid(s)

z = W2.dot(h) + b2
y = sigmoid(z)

# 5.7.3 Backward Computation

error = .5 * (t - y)**2
mu = 1

delta_2 = (t - y) * sigmoid_derivative(z)
delta_W2 = mu * delta_2 * h
delta_b2 = mu * delta_2

delta_1 = W * delta_2 * sigmoid_derivative(s)
delta_W = mu * np.array([ delta_1 ]).T * x
delta_b = mu * delta_1




