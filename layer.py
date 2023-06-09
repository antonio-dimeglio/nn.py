from abc import ABC 
import numpy as np

class Layer(ABC):
    def forward_prop(self, x:np.array):
        pass 

    def backward_prop(self, error, learning_rate):
        pass 


class ActivationLayer(Layer):
    def __init__(self, af, gradient_af):
        self.af = af 
        self.gradient_af = gradient_af

    def forward_prop(self, x:np.array):
        self.input = x 
        return self.af(x)

    def backward_prop(self, error, learning_rate):
        return self.gradient_af(self.input) * error 


class HiddenLayer(Layer):
    def __init__(self, input_size:int, output_size:int) -> None:
        self.weights = np.random.uniform(size = (input_size, output_size))
        self.biases = np.random.uniform(output_size)

    def forward_prop(self, x):
        self.input = x 
        return np.dot(x, self.weights) + self.biases
    
    def backward_prop(self, error, learning_rate):
        input_err = np.dot(error, self.weights.T)
        weights_err = np.dot(self.input.T, error)
        
        self.weights -= learning_rate * weights_err
        self.biases -= learning_rate * error 
        
        return input_err 