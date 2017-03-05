import scipy.io as sio
import numpy as np
import math
import matplotlib.pyplot as plt


def sigmoid_func(z):
    return float(1) / (1 + math.e**(-z))


def sigmoidGradient(z):

    # COmpute the gradient of the sigmoid function
    return sgrad


def nnCostFunction(Theta1, Theta2,  input_layer_size, hidden_layer_size, numLabels, X, y, lval):

    # First compute the cost J without regularization
    # Then include regularization
    # THis was part of week6 exercises


    ####"""""""""""""""""""""""""""""""""""""""""##########
    #I Part II, implement backpropagation to compute the gradients Theta1_grad and Theta2_grad.
    # They should be the partial derivatives of the cost function with respect to Theta1 and Theta2.
    # After implementing part 2, you can check your implementation by running checkNNGradients
    # Note the vector y input to the function is a vector of labels with values from 1...K.
    # You must map this vector into a binary vector of 1's and 0's to be used with the cost function.
    # We recommend implementing backpropagation using a for loop over the training samples if you are implementing
    # it for the first

    # Start implementing your code here

    #








    return J, Theta1_grad, Theta2_grad
