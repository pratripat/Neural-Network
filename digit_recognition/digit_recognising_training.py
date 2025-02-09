from neural_network import Neural_Network
import random, os, json
from PIL import Image
import numpy as np
from load_data import load_mnist_dataset, shuffle_data, convert_to_image

random_seed = 400
random.seed(random_seed)

nn = Neural_Network([784, 256, 256, 10], ['relu', 'relu', 'softmax'], batch_size=20, least_count=0.0001, target_least_count=0.00000001, random_seed=random_seed)

inputs, targets, total_inputs = load_mnist_dataset()

# Training the neural network
nn.train(inputs, targets, total_inputs)

# TAKING AN INPUT FOR CHECKING THE OUTPUT
while True:
    break
    print('===========================================================')
    a = input("Test neural network? (y/N): ").lower()

    if a == 'n':
        break

    input_number = int(input("Enter a number: "))
    index = list(targets).index(input_number)
    ip = list(inputs)[index]

    inputs, targets = shuffle_data(inputs, targets)

    nn.forward(np.array([ip]))

    op = nn.output
    print("\nCalculated output: ", np.argmax(op))
    print(nn.output)

    convert_to_image(ip).show()


# SAVING THE PARAMETERS AFTER CALCULATIONS FOR LATER USE
# ==============================================
f = open('params.json', 'w')
json.dump([[layer.weights.tolist(), layer.biases.tolist()] for layer in nn.layers], f)
f.close()
