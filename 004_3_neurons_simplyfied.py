###
# modelling 3 neurons (layer) with 4 inputs
# code simplyfied
###

###
## since we build 3 neurons, we need 3 unique weight sets
# so each neuron will have 4 weights (becasue of 4 inputs)
# also 3 neurons == 3 biases
###


# inputs from 4 neurons
inputs = [1.0, 2.0, 3.0, 2.5]

# making a list for the weights
# esentially list of lists 
weights = [[0.2, 0.8, -0.5, 1.0], 
           [0.5, -0.91, 0.26, -0.5], 
           [-0.26, -0.27, 0.17, 0.87]]

# list for biases
biases = [2, 3, 0.5]





#######################
# zip example
# 
# a = ("John", "Charles", "Mike")
# b = ("Jenny", "Christy", "Monica")
# x = zip(a, b)
# print(tuple(x))
#
# OUTPUT:
# 
# (('John', 'Jenny'), ('Charles', 'Christy'), ('Mike', 'Monica'))
#
#######################


# Output of current layer
layer_outputs = []

# For each neuron
for neuron_weights, neuron_bias in zip(weights, biases):
    # Zero output of given neuron    
    neuron_output = 0    
    # For each input and weight to the neuron 
    for n_input, weight in zip(inputs, neuron_weights):
        #calculate input*weight        
        # and add to the neuron's output variable
        neuron_output += n_input*weight    
    # Add bias to calculated neuron output    
    neuron_output += neuron_bias    
    # Put neuron's result to the layer's output list    
    layer_outputs.append(neuron_output)
print("3 neurons with 4 inputs")
print(layer_outputs)
