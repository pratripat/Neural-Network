import json
import numpy as np
import matplotlib.pyplot as plt

f = open('params_trained_3_spirals.json', 'r')
data = json.load(f)

# [w1, b1], [w2, b2],  [w3, b3] = data
[w1, b1], [w2, b2] = data
# w1, b1, w2, b2 = data

# w3 = np.array(w3)
w2 = np.array(w2)
w1 = np.array(w1)
# b3 = np.array(b3)
b2 = np.array(b2)
b1 = np.array(b1)

def relu(z):
    return np.maximum(0, z)

def softmax(z):
    """Compute the softmax of a vector z."""
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

a0 = np.array([eval(input('Enter the point: '))])

def calculate(a0):
    z1 = np.dot(a0, w1) + b1
    a1 = relu(z1)

    z2 = np.dot(a1, w2) + b2
    a2 = softmax(z2)

    # z3 = np.dot(a2, w3) + b3
    # a3 = softmax(z3)

    return a2

print(calculate(a0))

def convert_op_to_color(op):
    red = op[0][0] * np.array([1, 0, 0])
    green = op[0][1] * np.array([0, 1, 0])
    blue = op[0][2] * np.array([0, 0, 1])
    # yellow = op[0][3] * np.array([1, 1, 0])
    # black = op[0][4] * np.array([0, 0, 0])

    color = (red + green + blue)
    # color = (red + green + blue + yellow)

    # color = (red + green + blue + yellow + black)
    # color = [min(i, 1) for i in color]

    return color


visualize = eval(input('Visualize (1, 0): '))
if visualize:
    step_size = 0.025
    scaled_value = int(2 / step_size)

    for i in range(-scaled_value//2, scaled_value//2 + 1):
        for j in range(-scaled_value//2, scaled_value//2 + 1):
            op = calculate(np.array([[j * step_size, i * step_size]]))
            color = convert_op_to_color(op)
            plt.plot(j, i, 'o', color = [*color, 0.8])

    import nnfs
    from nnfs.datasets import spiral_data

    nnfs.init()

    # LOADING IN THE DATA
    # ==============================================
    total = 3000
    spirals = 3
    X, y = spiral_data(total//spirals, spirals)

    colors = {0:'r', 1:'g', 2:'b', 3:'yellow', 4:'black'}

    # Show the spirals
    for i in range(len(y)):
        plt.plot(np.array(X[i][0]) / step_size, np.array(X[i][1]) / step_size, 'o', color = colors[y[i]])

    plt.show()
