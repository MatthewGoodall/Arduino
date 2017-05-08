import serial, time
import math
import re
port = "COM3"
try:
    arduino = serial.Serial(port,9600)
    time.sleep(2)
    print("Connection to " + port + " established succesfully!\n")
except Exception as e:
    print(e)

import numpy as np
import struct

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
done = False

for j in range(60000):

    # Calculate forward through the network.
    layer_0 = X_data
    layer_1 = sigmoid(np.dot(layer_0, synapse0))
    layer_2 = sigmoid(np.dot(layer_1, synapse1))

    # Back propagation of errors using the chain rule.
    layer_2_error = y_data - layer_2

    layer_2_delta = layer_2_error*sigmoid(layer_2, deriv=True)

    layer_1_error = layer_2_delta.dot(synapse1.T)

    layer_1_delta = layer_1_error * sigmoid(layer_1,deriv=True)

    #update weights (no learning rate term)
    synapse1 += layer_1.T.dot(layer_2_delta)
    synapse0 += layer_0.T.dot(layer_1_delta)


    if(j % 10000) == 0:
        if j == 60000:
            done = True
print(layer_2)
while True:
    
    for x in range(4):
        data = np.rint(layer_2)
        data = str(data[x]).strip('[].')
        data = int(data)
        print(data)
        data = arduino.write(struct.pack('>B', data))
        time.sleep(2)
        print(struct.pack('>B', data))
