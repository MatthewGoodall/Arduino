import numpy as np
import serial 

def sigmoid(x, deriv=False):
    if(deriv==True):
        return (x*(1-x))

    return 1/(1+np.exp(-x))

#input data
X_data = np.array([[0,0,1],
            [0,1,1],
            [1,0,1],
            [1,1,1]])

#output data
y_data = np.array([[0],
             [0],
             [1],
             [1]])

np.random.seed(1)

#synapses
synapse0 = 2*np.random.random((3,4)) - 1  # 3x4 matrix of weights ((2 inputs + 1 bias) x 4 nodes in the hidden layer)
synapse1 = 2*np.random.random((4,1)) - 1  # 4x1 matrix of weights. (4 nodes x 1 output) - no bias term in the hidden layer.

for j in range(60000):

    # Calculate forward through the network.
    layer_0 = X_data
    layer_1 = sigmoid(np.dot(layer_0, synapse0))
    layer_2 = sigmoid(np.dot(layer_1, synapse1))

    # Back propagation of errors using the chain rule.
    layer_2_error = y_data - layer_2
    if(j % 10000) == 0:
        print("Error: " + str(np.mean(np.abs(layer_2_error))))

    layer_2_delta = layer_2_error*sigmoid(layer_2, deriv=True)

    layer_1_error = layer_2_delta.dot(synapse1.T)

    layer_1_delta = layer_1_error * sigmoid(layer_1,deriv=True)

    #update weights (no learning rate term)
    synapse1 += layer_1.T.dot(layer_2_delta)
    synapse0 += layer_0.T.dot(layer_1_delta)

print("Output after training")
print(layer_2)
