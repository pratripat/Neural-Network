import json
import numpy as np

f = open('params_trained_3_spirals.json', 'r')
data = json.load(f)

w1, b1, w2, b2 = data
w2 = np.array(w2)
w1 = np.array(w1)
b2 = np.array(b2)
b1 = np.array(b1)

def relu(z):
    return np.maximum(0, z)

def softmax(z):
    """Compute the softmax of a vector z."""
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


a0 = np.array([[0.217, 0]])

z1 = np.dot(a0, w1) + b1
a1 = relu(z1)

z2 = np.dot(a1, w2) + b2
a2 = softmax(z2)

print(a2)