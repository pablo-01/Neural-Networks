import numpy as np

###
# modelling 3 neurons (layer) with batch 
# (group of inputs/samples) of 3, each 4 inputs
###

 
# inputs from 4 neurons;
# 3 sets/groups of inputs
# each input has 4 features
inputs = [[1.0, 2.0, 3.0, 2.5],
          [2.0, 5.0, -1.0, 2.0],
          [-1.5 , 2.7, 3.3, -0.8]]

# making a list for the weights
# esentially list of lists
# matrix (of vectors)
#for layer 1
weights = [[0.2, 0.8, -0.5, 1.0], 
           [0.5, -0.91, 0.26, -0.5], 
           [-0.26, -0.27, 0.17, 0.87]]


# list for biases
biases = [2, 3, 0.5]

# for layer 2
weights2 = [[0.1, -0.14, 0.5], 
           [-0.5, 0.12, -0.33], 
           [-0.44, 0.73, -0.13]]

# list for biases
biases2 = [-1, 2, -0.5]





# to match shape of weights and inputs
# we need to transpose
# but first we need to convert to numpy array
layer1_outputs = np.dot(inputs, np.array(weights).T) + biases

layer2_outputs = np.dot(layer1_outputs, np.array(weights2).T) + biases2

print(layer2_outputs)





