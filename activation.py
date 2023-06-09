import numpy as np

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def grad_sigmoid(x):
    sigm = sigmoid(x)
    return sigm*(1-sigm)


def tanh(x):
    return np.tanh(x)

def grad_tanh(x):
    return 1-tanh(x)**2