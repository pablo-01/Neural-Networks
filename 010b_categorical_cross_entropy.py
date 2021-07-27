import math

''' 
an example of output that we may get from softmax activation function 
from output layer from NN
this it sued here to calculate the loss
'''
softmax_output = [0.7, 0.1, 0.2]

# target output
target_output = [1, 0, 0]

# calculate the loss
loss = -(math.log(softmax_output[0])*target_output[0] 
            + math.log(softmax_output[1])*target_output[1] 
            + math.log(softmax_output[2])*target_output[2])


print(loss)

loss = -math.log(softmax_output[0])
print(loss)
