import matplotlib.pyplot as plt
import numpy as np
import nnfs
from nnfs.datasets import spiral_data
import random

nnfs.init()

class Activation_Sigmoid:
    def forward(self, inputs):
        exp_values = np.exp(inputs)
        self.output = exp_values / (exp_values + 1)

        return self.output

    def differentiate(self, input):
        a = self.forward(np.array([input]))
        return (a * (1 - a))[0]

# ReLU - rectified linear activation function
class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

        return self.output

    def differentiate(self, input):
        return int(input > 0)

class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

        return self.output

    def differentiate(self, input):
        print('SOFTMAX FUNCTION: YOU HAVE NOT WRITTEN THE CODE FOR THE DIFFERENTIATION OF THE SOFTMAX FUNCTION')
        return 0

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons, activation_fn):
        # writing "* 0.1" makes sure that the weight values are between -1 and 1, so that after many layers, the input data does not exceed a very huge value
        self.weights = np.random.randn(n_inputs, n_neurons) * 0.1
        self.biases = np.zeros((1, n_neurons))

        self.activation_function = activation_fn()

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

        return self.output

    def activation(self):
        self.activation_output = self.activation_function.forward(self.output)

        return self.activation_output

class Neural_Network:
    def __init__(self, layers, activation_fns):
        self.activation_fns = activation_fns
        self.create_layers(layers)

        self.output_layer_cost_gradients = [0, 0]
        self.hidden_layer_cost_gradients = [[0, 0] for _ in range(len(self.layers)-1)]

    def create_layers(self, layers):
        self.layers = []
        for i in range(len(layers)-1):
            dense = Layer_Dense(layers[i], layers[i+1], self.activation_fns[i % len(self.activation_fns)])
            self.layers.append(dense)

    def forward(self, input):
        z = input
        for layer in self.layers:
            layer.forward(z)
            z = layer.activation()

        self.output = z
        return self.output

    def calculate_cost(self, expected_output):
        one_hot_encoded_output = np.array([1 if (i == expected_output) else 0 for i in range(len(self.output[0]))])

        diff = self.output - one_hot_encoded_output
        self.cost = np.sum(diff ** 2)

    def optimize(self, expected_output):
        one_hot_encoded_output = np.array([1 if (i == expected_output) else 0 for i in range(len(self.output[0]))])

        #=======================================================================
        # Calculating cost gradient for the output layer of the neural network
        node_values = []
        layer = self.output_layer

        # dc/dw2 = dc/da2 * da2/dz2 * dz2/dw2
        # node_values = dc/da2 * da2/dz2
        n_neurons = layer.weights.shape[1]

        for i in range(n_neurons):
            a2 = layer.activation_output[0][i]
            y = one_hot_encoded_output[i]

            dc_da2 = 2 * (a2 - y)
            da2_dz2 = layer.activation_function.differentiate(layer.output[0][i])

            node_values.append(dc_da2 * da2_dz2)

        node_values = np.array(node_values)

        # Correcting cost_gradient_wrt_weights calculation
        cost_gradient_wrt_weights = np.outer(layer.activation_output, node_values)
        # cost_gradient_wrt_weights = layer.weights * node_values

        cost_gradient_wrt_biases = node_values

        self.output_layer_cost_gradients[0] += cost_gradient_wrt_weights
        self.output_layer_cost_gradients[1] += cost_gradient_wrt_biases

        print("node_values", node_values)
        print(cost_gradient_wrt_weights)

        # layer.weights -= least_count * cost_gradient_wrt_weights
        # layer.biases -= least_count * cost_gradient_wrt_biases
        #========================================================================



        #========================================================================
        # Calculating the cost gradient for the hidden layers of the neural network

        old_node_values = node_values

        # new_node_values = da1/dz1 * dz2/da1 * old_node_values
        # new_node_values = da1/dz1 * w2 * old_node_values
        # dc/dw1 = dz1/dw1 * new_node_values

        # da1/dz1 = differentiation of a1 wrt z1, (d(RelU) -> Step function)

        for i, hidden_layer in enumerate(self.layers[len(self.layers)-2::-1]):
            da1_dz1 = np.array([hidden_layer.activation_function.differentiate(z1) for z1 in hidden_layer.output[0]])

            new_node_values = da1_dz1 * np.dot(self.layers[i+1].weights, old_node_values.T)

            # cost_gradient_wrt_weights = hidden_layer.weights * new_node_values
            cost_gradient_wrt_weights = np.outer(hidden_layer.activation_output, new_node_values)
            cost_gradient_wrt_biases = new_node_values

            self.hidden_layer_cost_gradients[i][0] += cost_gradient_wrt_weights
            self.hidden_layer_cost_gradients[i][1] += cost_gradient_wrt_biases

            old_node_values = new_node_values

            # hidden_layer.weights -= least_count * cost_gradient_wrt_weights
            # hidden_layer.biases -= least_count * cost_gradient_wrt_biases

        #========================================================================

    def apply_gradients(self, least_count, batch_size):
        self.output_layer.weights -= least_count * self.output_layer_cost_gradients[0] / batch_size
        self.output_layer.biases -= least_count * self.output_layer_cost_gradients[1] / batch_size

        for i, hidden_layer in enumerate(self.layers[len(self.layers)-2::-1]):
            print(i, hidden_layer.weights.shape, self.hidden_layer_cost_gradients[i][0].shape)
            hidden_layer.weights -= least_count * self.hidden_layer_cost_gradients[i][0] / batch_size
            hidden_layer.biases -= least_count * self.hidden_layer_cost_gradients[i][1] / batch_size

        self.output_layer_cost_gradients = [0, 0]
        self.hidden_layer_cost_gradients = [[0, 0] for _ in range(len(self.layers)-1)]

    @property
    def input_layer(self):
        return self.layers[0]

    @property
    def output_layer(self):
        return self.layers[-1]


X, y = spiral_data(500, 3)
#striver
temp = list(zip(X, y))
random.shuffle(temp)

X, y = zip(*temp)
X = np.array(list(X))
y = np.array(list(y))

colors = {0: 'r', 1:'g', 2:'b'}

for i in range(len(y)):
    plt.plot(X[i][0], X[i][1], 'o', color = colors[y[i]])

plt.show()

exit()

nn = Neural_Network([2, 10, 3], [Activation_ReLU, Activation_Sigmoid])
costs = []
least_count = 0.005
batch_size = 500

for i, input in enumerate(X):
    input = np.array([input])
    nn.forward(input)

    nn.calculate_cost(y[i])

    costs.append(nn.cost)

    nn.optimize(y[i])

    if (i+1) % batch_size == 0:
        nn.apply_gradients(least_count, batch_size)

print('COST:-')
plt.plot(range(1, len(costs)+1), costs, 'o', color='black')

plt.show()


print('OPTIMIZED VERSION:-')
input = np.array([0, 0])
nn.forward(input)

nn.calculate_cost(0)

print(nn.cost)
