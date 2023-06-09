from layer import Layer, HiddenLayer, ActivationLayer
from activation import sigmoid, grad_sigmoid, tanh, grad_tanh
from loss import mse, mse_prime
from network import Network
import numpy as np 
np.random.seed(1234)

X = np.array([[[0,0]], [[0,1]], [[1,0]], [[1,1]]])
Y = np.array([[[0]], [[1]], [[1]], [[0]]])
nn = Network(mse, mse_prime)
nn.add_layer(HiddenLayer(2, 2))
nn.add_layer(ActivationLayer(sigmoid, grad_sigmoid))
nn.add_layer(HiddenLayer(2, 1))
nn.add_layer(ActivationLayer(sigmoid, grad_sigmoid))
nn.train(X, Y, 500, 0.9)
nn.predict(X)