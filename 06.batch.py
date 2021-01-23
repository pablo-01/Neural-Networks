import numpy as np

###
# modelling 3 neurons (layer) with batch (group of inputs/samples) of 3 each 4 inputs 
# TODO
###


# inputs from 4 neurons;
# 3 sets/groups of inputs 
# vector
inputs = [[1.0, 2.0, 3.0, 2.5],
          [2.0, 5.0, -1.0, 2.0],
          [-1.5 , 2.7, 3.3, -0.8]]

# making a list for the weights
# esentially list of lists
# matrix (of vectors)
weights = [[0.2, 0.8, -0.5, 1.0], 
           [0.5, -0.91, 0.26, -0.5], 
           [-0.26, -0.27, 0.17, 0.87]]


# list for biases
# vector
biases = [2, 3, 0.5]





