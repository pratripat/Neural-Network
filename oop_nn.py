import matplotlib.pyplot as plt
import numpy as np
import nnfs
from nnfs.datasets import spiral_data
import random, json

nnfs.init()

# Cross-Entropy Loss
def cross_entropy_loss(calculated_output, expected_output):
    y = np.zeros_like(calculated_output)
    y[expected_output] = 1
    return -np.sum(y * np.log(calculated_output + 1e-15))

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

    def derivative(self, calculated_output, expected_output):
        y = np.zeros_like(calculated_output)
        y[expected_output] = 1
        return 2 * (calculated_output - y)

# class Activation_Softmax:
#     def forward(self, z):
#         exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
#         return exp_z / np.sum(exp_z, axis=1, keepdims=True)

#     def derivative(self, calculated_output, expected_output):
#         y = np.zeros_like(calculated_output)
#         y[expected_output] = 1
#         return (calculated_output - y)

class Layer:
    def __init__(self, n_inputs, n_neurons, activation_function):
        self.weights = np.random.randn(n_inputs, n_neurons) * 0.1
        self.biases = np.zeros((1, n_neurons))
        
        self.activation_function = activation_function()
    
    def forward(self, inputs):
        self.inputs = inputs
        self.z = np.dot(inputs, self.weights) + self.biases

        return self.z
    
    def activation(self):
        self.a = self.activation_function.forward(self.z)

        return self.a

class Neural_Network:
    def __init__(self, layers, activation_functions):
        self.activation_functions = activation_functions
        self.create_layers(layers)

        self.batch_size = 10
        self.least_count = 0.01
        self.lambda_reg = 0.01
        self.overall_average_losses = [0]
    
    def create_layers(self, layers):
        self.layers = []
        self.ip_neurons = layers[0]
        self.op_neurons = layers[-1]
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

    def update_least_count(self, batch):
        n = [3, 3/2]
        if batch in [total // (i*self.batch_size) for i in n]:
            self.least_count *= 0.1

    def train(self, batch, targets):
        self.cost_grad = [[np.zeros_like(layer.weights), np.zeros_like(layer.biases)] for layer in self.layers]
        losses = []

        self.update_least_count(batch)

        for i in range(self.batch_size):
            layer = self.output_layer

            calculated_output = self.output[i]
            expected_output = targets[i]

            # calculating loss (mean squared error) mse 
            # here np.arrange(4) = [0, 1, 2, 3]
            # when we do np.arrange(4) == 2, it gives the output as [False, False, True, False], basically one hot encoding the output
            loss = ((calculated_output - (np.arange(self.op_neurons) == expected_output))**2).sum()
            losses.append(loss)

            # loss = cross_entropy_loss(calculated_output, expected_output)
            # losses.append(loss)

            # calculating gradients for the output layer
            dc_da2 = layer.activation_function.derivative(calculated_output, expected_output)
            node_values = dc_da2

            dc_dw2 = np.outer(self.layers[-2].a[i], dc_da2)
            dc_db2 = dc_da2
        
            self.cost_grad[-1][0] += dc_dw2
            self.cost_grad[-1][1] += dc_db2

            # backpropagating
            # calculating the gradients for the hidden layers
            for layer_index in range(len(self.layers)-2, -1, -1):
                layer = self.layers[layer_index]

                da1_dz1 = layer.activation_function.derivative(layer.z[i])
                node_values = np.dot(node_values, self.layers[layer_index+1].weights.T) * da1_dz1

                dc_dw1 = np.outer(layer.inputs[i], node_values)
                dc_db1 = node_values    

                self.cost_grad[layer_index][0] += dc_dw1
                self.cost_grad[layer_index][1] += dc_db1

        # updating the paramteres according to the gradients
        for i, layer in enumerate(self.layers):
            # REGULARISATION (NOT DOING IT RN)
            # layer.weights -= self.least_count * (self.cost_grad[i][0] / self.batch_size + self.lambda_reg * layer.weights)
            # layer.biases -= self.least_count * (self.cost_grad[i][1] / self.batch_size + self.lambda_reg * layer.biases)

            layer.weights -= self.least_count * self.cost_grad[i][0] / self.batch_size
            layer.biases -= self.least_count * self.cost_grad[i][1] / self.batch_size

        # printing the average loss for analysis
        average_loss = sum(losses) / self.batch_size
        self.overall_average_losses.append(average_loss)
        print('Batch:', batch+1, 'Average Loss:', average_loss)

    @property
    def hidden_layers(self):
        return self.layers[:-1]

    @property
    def output_layer(self):
        return self.layers[-1]

# INITIAL PARAMETERS
# ================================================================
total = 1200000
spirals = 5
X, y = spiral_data(total//spirals, spirals)

# randomizing the order of X, y
temp = list(zip(X, y))
random.shuffle(temp)

X, y = zip(*temp)
X = np.array(list(X))
y = np.array(list(y))

colors = {0: 'r', 1: 'g', 2: 'b', 3: 'yellow', 4: 'black'}

nn = Neural_Network([2, 100, 100, spirals], [Activation_ReLU, Activation_ReLU, Activation_Softmax])

# TRAINING THE NEURAL NETWORK
# ================================================================
for batch in range(total // nn.batch_size):
    inputs = X[batch * nn.batch_size:(batch + 1) * nn.batch_size]
    targets = y[batch * nn.batch_size:(batch + 1) * nn.batch_size]

    nn.forward(inputs)
    nn.train(batch, targets)


# SAVING THE PARAMETERS AFTER CALCULATIONS FOR LATER USE 
# ==============================================
f = open('params_oop.json', 'w')
json.dump([[layer.weights.tolist(), layer.biases.tolist()] for layer in nn.layers], f)
f.close()


# PLOTTING THE LOSSES PER BATCH
# ==============================================
nn.overall_average_losses.pop(0)
plt.plot(range(1, len(nn.overall_average_losses)+1), nn.overall_average_losses, color='black')

plt.show()


# FINAL TEST TO CHECK IF THE NEURAL NETWORK LEARNT DECENTLY
# ================================================================
# Show the spirals
X, y = spiral_data(300, spirals)
print(X, y)
for i in range(len(y)):
    plt.plot(X[i][0], X[i][1], 'o', color = colors[y[i]])

plt.show()

# TAKING AN INPUT FOR CHECKING THE OUTPUT
input_test_point = eval(input('Enter a point: '))
nn.forward(np.array([input_test_point]))
print(nn.output)