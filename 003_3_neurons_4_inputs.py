###
# modelling 3 neurons (layer) with 4 inputs
###

###
## since we bilid 3 neurons, we need 3 unique weight sets
# so each neuron will have 4 weights (becasue of 4 inputs)
# also 3 neurons == 3 bias
###


# inputs from 4 neurons
inputs = [1.0, 2.0, 3.0, 2.5]



#weights
# each neuron has its weight set
weights1 = [0.2, 0.8, -0.5, 1.0]
weights2 = [0.5, -0.91, 0.26, -0.5]
weights3 = [-0.26, -0.27, 0.17, 0.87]


# every unique neuron has a unique bias
bias1 = 2.0
bias2 = 3.0
bias3 = 0.5


#calculation
# x1*w1 + x2+w2 + ... + xN*wN + bias


###
# output of 3 neurons
# calculation for each neuron
# Each neuron in a layer takes exactly the same input — the input given to the layer 
# (which can be either the training data or the output from the previous layer)
###


# output
output = ([
    # Neuron 1
    inputs[0] * weights1[0] + 
    inputs[1] * weights1[1] + 
    inputs[2] * weights1[2] + 
    inputs[3] * weights1[3] + 
    bias1,

    # Neuron 2
    inputs[0] * weights2[0] + 
    inputs[1] * weights2[1] + 
    inputs[2] * weights2[2] + 
    inputs[3] * weights2[3] + 
    bias2,

    # Neuron 3
    inputs[0] * weights3[0] + 
    inputs[1] * weights3[1] + 
    inputs[2] * weights3[2] + 
    inputs[3] * weights3[3] + 
    bias3 ])

##
# This is called a fully connected neural network — 
# every neuron in the current layer has connections to 
# every neuron from the previous layer.
##

print("3 neurons with 4 inputs")
print(output, "\n")

