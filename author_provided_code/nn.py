import math
import numpy as np

# neural network model

W = np.array([[3,4],[2,3]])
b = np.array([-2,-4])

W2 = np.array([5,-5])
b2 = np.array([-2])

# sigmoid function

@np.vectorize
def sigmoid(x):
  return 1 / (1 + math.exp(-x))

# derivative of sigmoid

@np.vectorize
def sigmoid_derivative(x):
  return sigmoid(x) * ( 1 - sigmoid(x) )

### forward computation

x = np.array([1,0])
t = np.array([1])

s = W.dot(x) + b
h = sigmoid( s )

z = W2.dot(h) + b2
y = sigmoid( z )
error = 1/2 * (t - y)**2

### backward computation

# learning rate
mu = 1

# hidden -> output updates
delta = ( t - y ) * sigmoid_derivative( z )
delta_W2 = mu * delta * h
delta_b2 = mu * delta

# input -> hidden updates
delta1 = delta * W2 * sigmoid_derivative( s )
delta_W = mu * np.array([ delta1 ]).T * x
delta_b = mu * delta1

### gradients for forward steps

d_error_d_y = t - y
gradient_y = d_error_d_y

d_y_d_z = sigmoid_derivative( z )
gradient_z = gradient_y * d_y_d_z

d_z_d_W2 = h
gradient_W2 = gradient_z * d_z_d_W2

d_z_d_b2 = 1
gradient_b2 = gradient_z * 1

d_z_d_h = W2
gradient_h = gradient_z * d_z_d_h

d_s_d_h = sigmoid_derivative( s )
gradient_s = gradient_h * d_s_d_h 

d_W_d_s = x
gradient_W = np.array([ gradient_s ]).T * d_W_d_s

d_b_d_s = 1
gradient_b = gradient_s * d_b_d_s


