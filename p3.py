import matplotlib.pyplot as plt
import numpy as np
import nnfs
from nnfs.datasets import spiral_data
from nnfs.datasets import sine
import random

import json

nnfs.init()


# LOADING IN THE DATA
# ==============================================
total = 3000
spirals = 3
X, y = spiral_data(total//spirals, spirals)
# X2, y2 = sine.create_data(total//spirals)

# shuffling the data
random.seed(13)
temp = list(zip(X, y))
random.shuffle(temp)

X, y = zip(*temp)
X = np.array(list(X))
y = np.array(list(y))

colors = {0:'r', 1:'g', 2:'b', 3:'yellow', 4:'black'}

# Show the spirals
for i in range(len(y)):
    plt.plot(X[i][0], X[i][1], 'o', color = colors[y[i]])

# for i in range(len(y2)):
#     plt.plot(X2[i][0], X[i][1], 'o', color = colors[y[i]])

plt.show()

exit()

# ACTIVATION FUNCTIONS
# ==============================================
def relu(z):
    return np.maximum(0, z)

def relu_derivate(z):
    return np.where(z >= 0, 1, 0)

def softmax(z):
    """Compute the softmax of a vector z."""
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def softmax_derivative(calculated_output, expected_output):
    """Gradient of MSE Loss with Softmax output."""
    y = np.zeros_like(calculated_output)
    y[expected_output] = 1  # One-hot encoding
    return 2 * (calculated_output - y)

# SETTING INITIAL PARAMS
# ==============================================
batch_size = 10

neurons = 200
op_neurons = 3
w1 = np.random.randn(2, neurons) * 0.1
b1 = np.zeros((1, neurons))
w2 = np.random.randn(neurons, op_neurons) * 0.1
b2 = np.zeros((1, op_neurons))

least_count = 0.01
overall_average_losses = [0]

# TRAINING
# ==============================================
for batch in range(total // batch_size):
    inputs = X[batch * batch_size:(batch + 1) * batch_size]
    targets = y[batch * batch_size:(batch + 1) * batch_size]

    a0 = inputs
    z1 = np.dot(a0, w1) + b1
    a1 = relu(z1)

    z2 = np.dot(a1, w2) + b2
    a2 = softmax(z2)

    # modifying and reducing the least count value to make sure the gradient reduces as it reaches the minima
    if batch == total // (2*batch_size) or batch == total // (4*batch_size) or batch == total // (8*batch_size):
        least_count *= 0.1


    cost_grad = [np.zeros_like(w2), np.zeros_like(w1), np.zeros_like(b2), np.zeros_like(b1)]
    losses = []

    for i in range(batch_size):
        calculated_output = a2[i]
        expected_output = targets[i]

        # calculating loss (mean squared error) mse
        # here np.arrange(4) = [0, 1, 2, 3]
        # when we do np.arrange(4) == 2, it gives the output as [False, False, True, False], basically one hot encoding the output
        loss = ((calculated_output - (np.arange(op_neurons) == expected_output))**2).sum()
        losses.append(loss)


        # calculating dc_dw2 and dc_db2
        dc_da2 = softmax_derivative(calculated_output, expected_output)
        node_values = dc_da2

        dc_dw2 = np.outer(a1[i], dc_da2)
        dc_db2 = dc_da2


        # calculating dc_dw1, dc_db1
        da1_dz1 = relu_derivate(z1[i])
        node_values = np.dot(node_values, w2.T) * da1_dz1

        dc_dw1 = np.outer(a0[i], node_values)
        dc_db1 = node_values


        # summing all the costs
        cost_grad[0] += dc_dw2
        cost_grad[1] += dc_dw1
        cost_grad[2] += dc_db2
        cost_grad[3] += dc_db1

    # updating all the weights and biases
    for i, param in enumerate([w2, w1, b2, b1]):
        param -= least_count * cost_grad[i] / batch_size

    # printing the average loss
    average_loss = sum(losses) / batch_size
    print("Batch:", batch, "Average Loss:", average_loss)
    overall_average_losses.append(average_loss)


# SAVING THE PARAMETERS AFTER CALCULATIONS FOR LATER USE
# ==============================================
f = open('params.json', 'w')
json.dump([w1.tolist(), b1.tolist(), w2.tolist(), b2.tolist()], f)
f.close()


# PLOTTING THE LOSSES PER BATCH
# ==============================================
overall_average_losses.pop(0)
plt.plot(range(1, len(overall_average_losses)+1), overall_average_losses, 'o', color='black')

plt.show()
