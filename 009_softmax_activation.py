import math
import numpy as np
import nnfs

## Softmax Activation Function


# input: output of data: 
layer_outputs = [[4.8, 1.21, 2.385],
                 [8.9, -1.8, 0.2],
                 [1.41, 1.051, 0.026]]

# E = 2.71828182846
# exponentiate to get rid of negative numbers
# while keeping the the meaning of the values
E = math.e

exp_values = np.exp(layer_outputs)

#print(exp_values)

# for output in layer_outputs:
#     exp_values.append(E ** output)

# axis 0 - the sum of columns
# axis 1 - the sum of rows (which is what we want)
# keepdims = True - makes it the same dimansion
# normalize the values
norm_values = exp_values / np.sum(exp_values, axis=1, keepdims=True)

print(norm_values)

'''
exponential values can grow very large very quickly
so we need to do some sort of overflow protection
v = u - max(u)
take all the values in output layer prior to exponentiation
and subtract the max value in the output layer (max u)  
from each value in that layer
this makes the range of the posibilites becomes between 0 and 1 after exponentionation
This is cintinued in file 008_softmax_activation.py
'''

