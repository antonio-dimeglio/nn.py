from layer import Layer

import numpy as np
class Network:
    def __init__(self, ef, grad_ef) -> None:
        self.layers:list[Layer] = []
        self.ef = ef 
        self.grad_ef = grad_ef


    def add_layer(self, layer:Layer):
        self.layers.append(layer)

    def train(self, X_train, Y_train, epochs:int, learning_rate:float):
        for epoch in range(epochs):
            epoch_error = 0 
            for i, x in enumerate(X_train):
                y_hat = x 

                for layer in self.layers:
                    y_hat = layer.forward_prop(y_hat)

                epoch_error += self.ef(Y_train[i], y_hat)
                error = self.grad_ef(Y_train[i], y_hat)

                for layer in reversed(self.layers):
                    error = layer.backward_prop(error, learning_rate)
            
            print(f"Epoch: {epoch}\tError: {epoch_error}")

        

    def predict(self, X_test) -> np.array:
        for i, x in enumerate(X_test):
            y_hat = x 

            for layer in self.layers:
                y_hat = layer.forward_prop(y_hat) 

            print(f"X: {x}\tPrediction: {y_hat}")

