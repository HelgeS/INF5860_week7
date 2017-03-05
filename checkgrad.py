import scipy.io as sio
import numpy as np
import math
import matplotlib.pyplot as plt

from nncost import nnCostFunction

def debugInitializeWeights(fan_out,fan_in):
    # initialize the weights with fan_in incoming connections and fan_out outgoing connections in a fixed strateby to help debub
    W = np.zeros((fan_out,1+fan_in))
    #Initialize W as sin so that is is always the same
    nelem = fan_out*(1+fan_in)
    nind = np.arange(nelem+1)
    sinvec = np.sin(nind[1:nelem+1])
    W = sinvec.T.reshape(fan_out,1+fan_in)/10
    tt=1
    inum = 0
    for r in range(0,fan_in+1):
        W[:,r] = sinvec[inum:inum+fan_out]
        inum = inum+fan_out

    W = W/10
    return W


def checkNNGradients(lv):
    # Create a small network to check the backpropagation gradient.
    input_layer_size = 3
    hidden_layer_size = 5
    num_labels = 3
    m=5

    
    # Lets generate some "random" test data
    Theta1 = debugInitializeWeights(hidden_layer_size, input_layer_size)
    Theta2 = debugInitializeWeights(num_labels, hidden_layer_size)
    # Reuse debugInitalizeWeights to generate X
    X = debugInitializeWeights(m, input_layer_size - 1)
    r1 = np.arange(1, input_layer_size+1)
    X[0,:]=r1
    X[1,:]=r1+1
    X[2,:]= r1+2
    X[3,:]=r1+3
    X[4,:]=r1+4
    yv = np.arange(1, m+1)
    yvmod = yv % num_labels

    y = yvmod + 1
    cost,Theta1_grad, Theta2_grad = nnCostFunction(Theta1, Theta2, input_layer_size, hidden_layer_size, num_labels, X, y,lv )
    # CHeck Theta1 and Theta2 numerically

    #Check Theta1 numerically
    numgrad1 = np.zeros(Theta1.shape)
    pval = 1e-4
    Theta1check = np.zeros(Theta1.shape)
    Theta1check[:,:] = Theta1[:,:]
    Theta2check = np.zeros(Theta2.shape)
    Theta2check[:,:] = Theta2[:,:]
    lv0 = 0
    print 'Gradient checking for Theta1'

    # For each Theta1-element, compare the numerical and backprop gradients
    .......... add your code here




    # Check Theta2 numerically
    numgrad2 = np.zeros(Theta2.shape)
    pval = 1e-4
    ........... and for Theta2
