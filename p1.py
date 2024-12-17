import matplotlib.pyplot as plt
import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        # writing "* 0.1" makes sure that the weight values are between -1 and 1, so that after many layers, the input data does not exceed a very huge value
        self.weights = np.random.randn(n_inputs, n_neurons) * 0.1
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

        return self.output

class Activation_Sigmoid:
    def forward(self, inputs):
        exp_values = np.exp(inputs)
        self.output = exp_values / (exp_values + 1)

        return self.output

# ReLU - rectified linear activation function
class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

        return self.output

class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

        return self.output

class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss

class Loss_CateogoricalClassentropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)

        # [1, 0, 2]
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]

        # [[0,1,0], [1,0,0], [0,0,1]]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

class Loss_ThreeBlueOneBrown(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)

        loss = []
        for index, true_value in enumerate(y_true):
            loss.append(np.sum((y_pred[index] - true_value) ** 2))

        return np.array(loss)

def one_hot_encode(l):
    if len(l.shape) == 2:
        return l

    length = max(l)
    one_hot_encoded_l = [[0]*(length+1) for _ in range(len(l))]
    for i, index in enumerate(l):
        one_hot_encoded_l[i][index] = 1

    return np.array(one_hot_encoded_l)

def calculate_cost_gradient(loss, y_pred, y_true):
    # da2/dz2 = A2(z2)[1-A2(z2)]
    # dz2/dw2 = A1(z1)

    da2_dz2 = a2 * (1 - a2)
    dz2_dw2 = a1

    node_values = (da2_dz2 * dz2_dw2)


    # dc/da2 = 2 * (y_pred - y_true)
    dc_da2 = 2 * (y_pred - y_true)
    loss_gradient = dc_da2 * node_values

    print(loss_gradient, dense2.weights)


X, y = spiral_data(100, 3)

colors = {0: 'r', 1:'g', 2:'b'}

for i in range(len(y)):
    plt.plot(X[i][0], X[i][1], 'o', color = colors[y[i]])

plt.show()
# here n_inputs = 2, as the data produced by the spiral_data function are points in the x-y plane. so according to their position we can determine which spiral the data belongs to, so the only two inputs for the point is it's position (x, y). Hence two values...

dense1 = Layer_Dense(2, 3)
activation1 = Activation_ReLU()

dense2 = Layer_Dense(3, 3)
activation2 = Activation_Softmax()

loss_function = Loss_ThreeBlueOneBrown()


for index in range(1):
    x = X[index]
    x = np.array([x])
    # First test
    z1 = dense1.forward(x)
    a1 = activation1.forward(z1)
    z2 = dense2.forward(a1)
    a2 = activation2.forward(z2)

    print(a2)

    # loss = loss_function.calculate(a2, y)


    print(a2[:5])
    print()
    print('Loss:', loss)
    print()

    # calculate_cost_gradient(loss, a2, y)


# Second test
# z1 = dense1.forward(X)
# a1 = activation1.forward(z1)
# z2 = dense2.forward(a1)
# a2 = activation2.forward(z2)
#
# loss_function = Loss_ThreeBlueOneBrown()
# loss = loss_function.calculate(a2, y)
#
#
# print('\n\n')
# print(a2[:5])
# print()
# print('Loss:', loss)
# print()
