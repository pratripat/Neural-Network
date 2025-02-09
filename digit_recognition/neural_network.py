import random, math, json
import numpy as np

random.seed(400)

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

# LAYERS
# =================================================================
class Layer:
    def __init__(self, n_inputs, n_neurons, activation_function):
        self.weights = np.random.randn(n_inputs, n_neurons) * 0.1
        self.biases = np.zeros((1, n_neurons))
        
        self.activation_function = activation_function()
    
    def set_params(self, weights, biases):
        self.weights = weights
        self.biases = biases
    
    def forward(self, inputs):
        self.inputs = inputs
        self.z = np.dot(inputs, self.weights) + self.biases

        return self.z
    
    def activation(self):
        self.a = self.activation_function.forward(self.z)

        return self.a

class Neural_Network:
    def __init__(self, layers, activation_functions, batch_size=10, least_count=0.01, target_least_count=0.0001, random_seed=400):
        random.seed(random_seed)

        self.load_activation_functions(activation_functions)
        self.create_layers(layers)

        self.total_inputs = 0
        self.batch_size = batch_size
        self.least_count = least_count
        self.target_least_count = target_least_count
        self.lambda_reg = 0.01
        self.overall_average_losses = [0]

        self.exp_const = math.log(least_count / target_least_count)
    
    def load_activation_functions(self, activation_functions):
        functions = {'relu': Activation_ReLU, 'softmax': Activation_Softmax}
        self.activation_functions = [functions[function] for function in activation_functions]
    
    def load_weights(self, json_file):
        data = json.load(open(json_file, 'r'))
        
        layers = []
        params = []

        for weights, biases in data:
            np_weights = np.array(weights)
            np_biases = np.array(biases)
            
            layers.append(len(weights))
            params.append([np_weights, np_biases])
        
        layers.append(len(weights[0]))

        # loading the layers
        activation_functions = (['relu'] * (len(layers)-2)) + ['softmax']
        self.load_activation_functions(activation_functions)
        self.create_layers(layers)

        for i, layer in enumerate(self.layers):
            layer.set_params(*params[i])

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
        n = [4, 4/3, 2, 5/4]
        # n = [3, 1.5]
        if batch in [self.total_inputs // (i*self.batch_size) for i in n]:
            self.least_count *= 0.1

        # # exp decay
        # self.least_count = ((self.least_count - self.target_least_count) * np.exp(-self.exp_const * batch)) + self.target_least_count

        # linear decay
        # self.least_count = self.target_least_count + ((1 - batch/self.total_number_of_batches)*(self.least_count - self.target_least_count))

        # self.least_count -= 0.00000185463

    def train(self, input_X, targets_Y, total):
        self.total_inputs = total
        self.total_number_of_batches = self.total_inputs // self.batch_size
        print(self.total_number_of_batches)
        self.exp_const /= total # updating the exp const acc to the total number of inputs 
        for batch in range(self.total_number_of_batches):
            inputs = input_X[batch * self.batch_size:(batch + 1) * self.batch_size]
            targets = targets_Y[batch * self.batch_size:(batch + 1) * self.batch_size]

            self.forward(inputs)
            self.update(batch, targets)
        
        self.exp_const *= total # putting the exp const back to normal for later purposes

    def update(self, batch, targets):
        self.cost_grad = [[np.zeros_like(layer.weights), np.zeros_like(layer.biases)] for layer in self.layers]
        losses = []
        correct_predictions = 0

        self.update_least_count(batch)

        for i in range(self.batch_size):
            layer = self.output_layer

            calculated_output = self.output[i]
            expected_output = targets[i]

            # storing predictions for calculating accuracy
            correct_predictions += int(list(calculated_output).index(max(calculated_output)) == expected_output)

            # calculating loss (mean squared error) mse 
            # here np.arrange(4) = [0, 1, 2, 3]
            # when we do np.arrange(4) == 2, it gives the output as [False, False, True, False], basically one hot encoding the output
            loss = ((calculated_output - (np.arange(self.op_neurons) == expected_output))**2).sum()
            losses.append(loss)

            # DEBUG THE LOSS
            # print(loss, list(calculated_output).index(max(calculated_output)), expected_output)

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
        accuracy = correct_predictions / self.batch_size*100
        print('Batch:', batch+1, 'Least Count:', self.least_count, 'Average Loss:', average_loss, 'Accuracy: ', accuracy , "%")

    @property
    def hidden_layers(self):
        return self.layers[:-1]

    @property
    def output_layer(self):
        return self.layers[-1]