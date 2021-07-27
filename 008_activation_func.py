import numpy as np
import nnfs
from nnfs.datasets import spiral_data


nnfs.init()

# Dense layer
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        # shape, size of sample n_inputs,
        # 0.10 for normalisation 
        self.weights = 0.10 * np.random.rand(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


# Activation function relu
class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

# Softmax activation function
class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities


# common loss
class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss

# categorical cross entropy loss
# inherit from Loss class
class Loss_CategoricalCrossEntropy(Loss):
    # y_pred is the output of the network
    # y_true is the true label
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        # clip the values to avoid log(0) - infinity
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)

        # check for one-hot encoding or not
        # scalar class values
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        # one-hot encoded vectors (2d array)
        # in one-hot encoding everything is 0 exept the target
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped*y_true, axis=1)

        # negative log likelihood
        negative_log_likelihood = -np.log(correct_confidences)
        # returns vector of values 
        return negative_log_likelihood 




# data
X, y = spiral_data(samples=100, classes=3)

# (inputs, outputs)
dense1 = Layer_Dense(2, 3)
activation1 = Activation_ReLU()

# (inputs, outputs): output from the previous layer is the input for the next layer
dense2 = Layer_Dense(3, 3)
activation2 = Activation_Softmax()

# begin passing data
dense1.forward(X)
# activate
activation1.forward(dense1.output)


# pass data through the next layer
dense2.forward(activation1.output)
activation2.forward(dense2.output)

# probs
# print first 5
print(activation2.output[:5])

# define loss fuction
loos_function = Loss_CategoricalCrossEntropy()

# calculate loss
loss = loos_function.calculate(activation2.output, y)

print("Loss: ", loss)










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