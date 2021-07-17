import numpy as np

# initial weights random
np.random.seed(0)


# inputs (input data to NN)
X = [[1.0, 2.0, 3.0, 2.5],
     [2.0, 5.0, -1.0, 2.0],
     [-1.5 , 2.7, 3.3, -0.8]]

# Dense layer
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        # shape, size of sample n_inputs,
        # 0.10 for normalisation 
        self.weights = 0.10 * np.random.rand(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
# size of iputs (number of features in each sample);
# number of neurons (any number)
layer1 = Layer_Dense(4, 5)

# size of inputs is the same as the number 
# of neurons in the previous layer
layer2 = Layer_Dense(5, 2)

layer1.forward(X)
#print(layer1.output)
layer2.forward(layer1.output)
print(layer2.output)







