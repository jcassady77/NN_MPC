#Reference: https://iamtrask.github.io/2015/07/12/basic-python-network/

import numpy as np
# compute sigmoid nonlinearity
def sigmoid(x):
    output = 1/(1+np.exp(-x))
    return output

# convert output of sigmoid function to its derivative
def sigmoid_output_to_derivative(output):
    return output*(1-output)


ins = np.loadtxt(open("inputStep1.csv", "rb"), delimiter=",", skiprows=1)
outs = np.loadtxt(open("outputStep1.csv", "rb"), delimiter=",", skiprows=1)
outs = outs.transpose()

print("ins")
#print((ins))
print("outs")
#print((outs.transpose()))

# input dataset
X = np.array(ins)
#X = np.array([  [0,1],
#                [0,1],
#                [1,0],
#                [1,0] ])

# output dataset    
y = np.array(outs)      
#y = np.array([[0,1],[0,1],[0,1],[0,1]])


# seed random numbers to make calculation
# deterministic (just a good practice)
np.random.seed(1)

# initialize weights randomly with mean 0
weights1 = 2*np.random.random((2,2)) - 1
weights2 = 2*np.random.random((2,10)) - 1


for iter in range(10):
    for n in range(98):
        # forward propagation
        layer_0 = X[n*10+0:n*10+10]
        layer_1 = sigmoid(np.dot(layer_0,weights1))
        layer_2 = sigmoid(np.dot(layer_1,weights2))
        
        # how much did we miss?
        layer_2_error = layer_2 - y[n*10+0:n*10+10]
        print('error')
        print(np.average(layer_2_error))
        # multiply how much we missed by the
        # slope of the sigmoid at the values in l1
        layer_2_delta = layer_2_error * sigmoid_output_to_derivative(layer_2)

        layer_1_error = layer_2_delta.dot(weights2.T)

        layer_1_delta = layer_1_error * sigmoid_output_to_derivative(layer_1)

        dWeights1 = np.dot(layer_0.T,layer_1_delta)
        dWeights2 = np.dot(layer_1.T,layer_2_delta)
        # update weights
        weights1 -= dWeights1
        weights2 -= dWeights2

print("Output After Training:")


