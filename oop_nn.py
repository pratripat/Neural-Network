import matplotlib.pyplot as plt
import numpy as np
import nnfs
from nnfs.datasets import spiral_data
import random

nnfs.init()

# CLASSES 
# ================================================================
class Activation_ReLU:
    def forward(self, z):
        return np.maximum(0, z)
    
    def derivative(self, z):
        return np.where(z >= 0, 1, 0)

class Activation_Softmax:
    def forward(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def derivative(self, z):
        y = np.zeros_like(calculated_output)
        y[expected_output] = 1
        return 2 * (calculated_output - y)

class Layer:
    def __init__(self, n_inputs, n_neurons, activation_function):
        self.weights = np.random.randn(n_inputs, n_neurons) * 0.1
        self.biases = np.zeros((1, n_neurons))
        
        self.activation_function = activation_function()
    
    def forward(self, inputs):
        self.z = np.dot(inputs, self.weights) + self.biases

        return self.z
    
    def activation(self):
        self.a = self.activation_function.forward(self.z)

        return self.a

class Neural_Network:
    def __init__(self, layers, activation_functions):
        self.activation_functions = activation_functions
        self.create_layers(layers)
    
    def create_layers(self, layers):
        self.layers = []
        for i in range(len(layers)-1):
            layer = Layer(layers[i], layers[i+1], self.activation_functions[i])
            self.layers.append(layer)
    
    def forward(self, input):
        a = input
        for layer in self.layers:
            layer.forward(a)
            a = layer.activation()

        self.output = a
        return self.output


# INITIAL PARAMETERS
# ================================================================
total = 300
spirals = 3
X, y = spiral_data(total//spirals, spirals)

temp = list(zip(X, y))
random.shuffle(temp)

X, y = zip(*temp)
X = np.array(list(X))
y = np.array(list(y))

colors = {0: 'r', 1:'g', 2:'b'}

nn = Neural_Network([2, 5, 3], [Activation_ReLU, Activation_Softmax])

least_count = 0.01
overall_average_losses = [0]
batch_size = 10

# TRAINING THE NEURAL NETWORK
# ================================================================
for batch in range(total // batch_size):
    inputs = X[batch * batch_size:(batch + 1) * batch_size]
    targets = y[batch * batch_size:(batch + 1) * batch_size]

    nn.forward(inputs)
    
    print(nn.output)