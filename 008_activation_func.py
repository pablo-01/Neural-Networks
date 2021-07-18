import numpy as np
import nnfs
from nnfs.datasets import spiral_data


nnfs.init()

# sample exercise data
# 100 feature sets of 3 classes
X, y = spiral_data(100, 3)


# Dense layer
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        # shape, size of sample n_inputs,
        # 0.10 for normalisation 
        self.weights = 0.10 * np.random.rand(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


# Activation function
class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


# size of iputs (number of features in each sample);
# number of neurons (any number)
layer1 = Layer_Dense(2, 5)
activation1 = Activation_ReLU()

# size of inputs is the same as the number
# of neurons in the previous layer

layer1.forward(X)
#print(layer1.output)
activation1.forward(layer1.output)
print(activation1.output)









####################
# inputs = [0, 2, -1, 3.3, -2.7, 1.1, 2,2, -100]

# output = []

# # reLU function
# for i in inputs:
#     if i > 0:
#         output.append(i)
#     elif i < 0:
#         output.append(0)

# or

# output.append(max(0, i))

# print(output)