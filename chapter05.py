# 5.7.1 Data Structures and Functions in Numpy
import math
import numpy as np

# Refer to the example XOR neural network in Figure 5.4 on page 71.
# Refer to Table 5.2, page 78

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

# These are the input x0, x1:
# x0, x1,   t
#  0,  0    0
#  0,  1    1
#  1,  0    1
#  1,  1    0
x = np.array([1, 0])
t = np.array([1])

# 5.7.2 Forward Computation

s = W.dot(x) + b
h = sigmoid(s)

z = W2.dot(h) + b2
y = sigmoid(z)

############################################################
# 5.7.3 Backward Computation

error = 1/2 * (t - y)**2
mu = 1

delta_2 = (t - y) * sigmoid_derivative(z)
delta_W2 = mu * delta_2 * h
delta_b2 = mu * delta_2

delta_1 = W * delta_2 * sigmoid_derivative(s)
delta_W = mu * np.array([ delta_1 ]).T * x
delta_b = mu * delta_1

d_error_d_y = t - y

d_y_d_z = sigmoid_derivative( z )
d_error_d_z = d_error_d_y * d_y_d_z

d_z_d_W2 = h
d_error_d_W2 = d_error_d_z * d_z_d_W2

d_z_d_b2 = 1
d_error_d_b2 = d_error_d_z * d_z_d_b2
d_z_d_h = W2
d_error_d_h = d_error_d_z * d_z_d_h

d_s_d_h = sigmoid_derivative( s )
d_error_d_s = d_error_d_h * d_s_d_h
d_W_d_s = x
d_error_d_W = np.array([ d_error_d_s ]).T * d_W_d_s
d_b_d_s = 1
d_error_d_b = d_error_d_s * d_b_d_s
