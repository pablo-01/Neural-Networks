# single neuron within a Neural Network (NN) 
# receiving values from 3 neurons from previous layer


# in NN every neuron has unique connection 
# to every single previous neuron

###
# let say there are 3 neurons that are feeding
# into this 1 neuron, 
# and those neurons are outputting some values
# the outputs from those 3 neurons
# are the inputs of the neuron in next layer
###


#inputs
inputs = [1.0, 2.0, 3.0]

###
# every input also has 
# unique weight associated with it
# since there are 3 inputs
# we'll have 3 weights
###


# weights
weights = [0.2, 0.8, -0.5]

# every unique neuron has a unique bias
bias = 2.0


###
# first step for a neuron is to compute the output
# sum(weight * input) + bias , 
# or 
# x1*w1 + x2+w2 + ... + xN*wN + bias
###

# output of the neuron

# Neuron 1
output = (inputs[0] * weights[0] + 
          inputs[1] * weights[1] + 
          inputs[2] * weights[2] +
          bias)

print("One neuroun 3 inputs")
print(output, "\n")















# this is later passed into activation function
# activationFunction(x1*w1 + x2+w2 + ... + xN*wN + bias)  