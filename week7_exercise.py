# This file helps you implement backpropagation on a small data set of 5 samples.

import scipy.io as sio
import numpy as np
import math
import matplotlib.pyplot as plt
from checkgrad import checkNNGradients
from nncost import sigmoid_func, sigmoidGradient, nnCostFunction




def randInitializeWeights(L_in, L_out):
    # This function initialize weights of a layer with  L_in incoming  weights and L_out outgoing weights
    # you should return W,   a matrix of size (L_out,1+L_in)
    # Fill in random small number in W

    return W



lrat = 1

################################################
# Part 1: implement and test sigmoid gradient
#############################################
# Fill in sigmoidGradient in nncost.py
t1 = -50
t2 = 0
t3 = 100
s1 = sigmoidGradient(t1)
s2 = sigmoidGradient(t2)
s3 = sigmoidGradient(t3)

print 'sigmoidgrad -50 - should be near 0:', s1
print 'sigmoidgrad -0 - should be 0.25: ', s2
print 'sigmoidgrad 100 - should be near 0 :', s3

#""""""""""""""""""""""""""""""""""""""""""""""""""""""
# Part 2: Initialize network parameters
########################################################
## Implement randInitialWeights to complete this part


initial_Theta1 = randInitializeWeights(input_layer_size,hidden_layer_size)
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels)
# CHeck that the weights are different from zero, and not equal

wmean = np.mean(initial_Theta1)
wstd = np.std(initial_Theta1)
print 'The weight mean should be close to zero, with standard deviation different from zero'
print 'Mean(initial_Theta1): ', wmean
print 'std(initial_Theta1): ', wstd



######################################################################
#          Part 3; implement backpropagation
#####################################################################
# Before computing the gradients in nnCostfunction, make sure that it returns the correct cost with
# and without regularization
# You implemented this last week, if not, do it now. The numbers below are for MNIST data in week7data1.mat
# Without regularization, J should be 0.2876
# With regularization (lambda=1), J should be 0.38377

#J,Jgrad =  nnCostFunction(Theta1, Theta2, input_layer_size, hidden_layer_size, num_labels, X, y,lrat)

lv = 0
############################################
### WHen implementing, work on a simple small dataset crated in checkNNGradients
checkNNGradients(lv)




J,dgrad1,dgrad2 =  nnCostFunction(Theta1, Theta2, input_layer_size, hidden_layer_size, num_labels, X, y,lrat)

print 'Checking gradient after regularization'

lrat = 3
checkNNGradients(lv)

# Also output the costfunction debugging values
debug_J, dgrad1, dgrad2 =  nnCostFunction(Theta1, Theta2, input_layer_size, hidden_layer_size, num_labels, X, y,lrat)
print 'Cost at fixed debugging parameters w lambda=10, this should be 0.576051', debug_J








############### Do not proceed until your code pass the gradient check.
# Now we will train on MNIST images
# Set up the number of nodes
input_layer_size = 400
hidden_layer_size = 25
num_labels = 10

# Load  the data
#THis contains the MNIST images of handwritten digits
datastruct = sio.loadmat('week7data1.mat')

X = datastruct['X']
y=datastruct['y']
nsamp,nfeat = X.shape








#############################################################
##########Part 4:                           """""""""""""""""""
################################################################
## Now we will train the net on MNIST data, and avoid checkingGradient

# Do a very simple gradient descent
maxiter = 100 # You might want to increase this
Theta1 = initial_Theta1
Theta2 = initial_Theta2
lreg=1
learningrate = 0.1 ######### Do you need to increase it to speed up the process?
Jval = np.zeros(maxiter)
for it in range(0,maxiter):

    ## Inlcude code to compute the cost function and gradients
    ## Update Theta1 and Theta2 according to gradient descent

    # Monitor if the cost J decreases

# PRedict the class labels and compute the accuracy on the training data set



