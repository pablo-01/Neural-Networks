####
# one neuron with 4 inputs
###



# inputs from 4 neurons
inputs = [1.0, 2.0, 3.0, 2.5]



# weights
weights = [0.2, 0.8, -0.5, 1.0]

# every unique neuron has a unique bias
bias = 2.0


# x1*w1 + x2+w2 + ... + xN*wN + bias


# output of the neuron
output = (inputs[0] * weights[0] + 
          inputs[1] * weights[1] + 
          inputs[2] * weights[2] +
          inputs[3] * weights[3] +
          bias)


print("One neuron 4 inputs")
print(output, "\n")