import numpy as np

###
# modelling 3 neurons (layer) with 4 inputs
# fruther simplyfication by using numpy dot product
###


# inputs from 4 neurons; 
# vector
inputs = [1.0, 2.0, 3.0, 2.5]

# making a list for the weights
# esentially list of lists
# matrix (of vectors)
weights = [[0.2, 0.8, -0.5, 1.0], 
           [0.5, -0.91, 0.26, -0.5], 
           [-0.26, -0.27, 0.17, 0.87]]


# list for biases
# vector
biases = [2, 3, 0.5]

# dot_product 
layer_output = np.dot(weights, inputs) + biases

print("Layer output: ")
print(layer_output)



