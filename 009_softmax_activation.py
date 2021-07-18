import math
import numpy as np



layer_outputs = [4.8, 1.21, 2.385]

# E = 2.71828182846
# exponentiate to get rid of negative numbers
# while keeping the the meaning of the values
E = math.e

exp_values = np.exp(layer_outputs)

# for output in layer_outputs:
#     exp_values.append(E ** output)



norm_values = exp_values / np.sum(exp_values)
# # normalize the values
# norm_base = sum(exp_values)
# norm_values = []

# for value in exp_values:
#     norm_values.append(value / norm_base)




print(norm_values)
print(sum(norm_values))