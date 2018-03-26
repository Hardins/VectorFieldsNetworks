# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 16:36:49 2018

@author: Kovau
"""

import numpy as np  
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import random
from sklearn import cluster, datasets
matplotlib.use('Agg')
plt.ioff()


########################    ACTIVATION FUNCTIONS   ########################
def log_reg(x):  
    return 1/(1+np.exp(-x))  
    
def log_reg_d(x):
    return x*(1-x)
########################    ACTIVATION FUNCTIONS   ########################

########################    COST FUNCTIONS   ########################
def binary_entropy(x, y, N):#, reg, lambda_r):
    return -np.sum(y*np.log(x)+(1-y)*np.log(1-x)) #+ reg(lambda_r, synapse, N)
    
def binary_entropy_d(x, y, N)  :
    return -(np.nan_to_num(y/x) - np.nan_to_num((1-y)/(1-x)))
########################    COST FUNCTIONS   ########################

########################    PREDICTION FUNCTION   ########################
def predict(X, V, C, N, synapse1, bias1):
    """
    Given the trained synapses, calculates the function prediction for a number
    of examples. User must be careful to set the proper non linearity, as
    hyperbolic tangent is the default.
    """
    dim = len(X[0])
    layer1 = np.zeros([N, 2])
    layer2 = np.zeros([N, 1])  
    
    for i in xrange (len(V)):
        #layer1 += np.multiply(V[i],np.prod(np.exp(- np.multiply(X- C[i], X- C[i])), axis =1))
        aux= np.prod(np.exp(- np.multiply(X- C[i], X- C[i])),axis=1)
        layer1 += np.multiply(V[i], np.transpose(np.tile(aux, [dim,1])))
        
    layer1 += X
    
    #layer1[:,0] = X[:,0] + V[0][0]*np.exp(-(X[:,0] - C[0][0])**2)*np.exp(-(X[:,1] - C[0][1])**2)   + V[1][0]*np.exp(-(X[:,0] - C[1][0])**2)*np.exp(-(X[:,1] - C[1][1])**2)
    #layer1[:,1] = X[:,1] + V[0][1]*np.exp(-(X[:,0] - C[0][0])**2)*np.exp(-(X[:,1] - C[0][1])**2)   + V[1][1]*np.exp(-(X[:,0] - C[1][0])**2)*np.exp(-(X[:,1] - C[1][1])**2)
    layer2= 1./(1.+np.exp(-(synapse1[0]*layer1[:,[0]] + synapse1[1]*layer1[:,[1]] +bias1)))
    return layer1, layer2


def accuracy(x, y, N):
    for i in xrange(N):
        if x[i,0]<0.5:
            x[i,0] = .0
        else:
            x[i,0] = 1.
    return     1.- np.sum(np.abs(y- x))/N       

########################    PREDICTION FUNCTION   ########################

########################    OPTIMIZATION METHODS   ########################
def SGD(layer0,\
 layer1,\
 layer2,\
 C,\
 V,\
 synapse1,\
 bias1,\
 y,\
 epochs,\
 eta,\
 mini_batch_size,\
 nonlin,\
 cost):
    """
    Stochastic Gradient Descent Method. Starting from a given point
    minimum is reached through a iteration process in the direction of greatest
    descent.
    Inputs:
        Layers(0,1,2) correspond to the input, hidden and output layer,
        respectively;
        [numyp multidimensional array]

        Synapses(1) correspond to weights connecting layers(1-2);
        [numyp multidimensional array]
            
        y is the desired function value in layer 2;
        [numyp multidimensional array]
            
        epochs determines how many iterations are to be used;
        [integer]
        
        eta is the step size for each iteration;
        [float]
        
        mini_batch_size sets the size of each iteration batch, if equals to
        False, full batch'll be used. Same applies for mini batch size higher
        than the entire sample;
        [integer / bool]
        
        nonlin is a list containing the activation function and its
        derivative;
        [list of objects]
        
        cost is a list containing the cost function and its derivative;
        [list of objects]
    
            
    Outputs:
            Layers and synapses/bias values are returned after convergence/iteration
            process.
    """  
    size = len(layer2)
    dim = len(layer0[0])
    if mini_batch_size==False:
        mini_batch_size = size      

    loss = np.zeros(epochs)
    acc = np.zeros(epochs)
    V1=np.copy(V)
    for i in xrange(epochs):
            
        print i
        permutation = np.random.permutation(mini_batch_size) 
        batch_X = layer0#[permutation]
        batch_y = y#[permutation]
        
        #Foward Propagation. Updates neurons values using current synapses.         
        layer1= np.zeros([len(batch_X),dim])
        for j in xrange (len(V)):
            aux= np.prod(np.exp(- np.multiply(batch_X- C[j], batch_X- C[j])),axis=1)
            layer1 += np.multiply(V[j], np.transpose(np.tile(aux, [dim,1])))
        layer1 += batch_X
    
        layer2= 1./(1.+np.exp(-(synapse1[0]*layer1[:,[0]] + synapse1[1]*layer1[:,[1]] +bias1)))  
        
        loss[i] = binary_entropy(layer2, batch_y, size)
        acc[i] = accuracy(layer2, batch_y, size)
        #Updates synapses and bias based on descent direction and step size.



        dv, dc = np.zeros([len(V),dim]), np.zeros([len(V),dim])
        for j in xrange(len(V)):
            gaussian = np.multiply( np.transpose(layer2 - batch_y), np.prod(np.exp(- np.multiply(batch_X- C[j], batch_X- C[j])), axis=1))
            dv[j] = np.mean(gaussian) * np.transpose(synapse1)
            dc[j] = 2*np.mean(np.transpose(np.tile(gaussian,[dim,1]))*np.dot(np.transpose(synapse1),V[j])*(batch_X- C[j]),axis=0)

            
#        gaussian1 = np.exp(-(batch_X[:,0] - C[0][0])**2)*np.exp(-(batch_X[:,1] - C[0][1])**2)
#        gaussian2 = np.exp(-(batch_X[:,0] - C[1][0])**2)*np.exp(-(batch_X[:,1] - C[1][1])**2)

#        d1_c0 = 2.*np.transpose(layer2- batch_y)*gaussian1*(batch_X[:,0] - C[0][0])*(synapse1[0]*V[0][0] + synapse1[1]*V[0][1])
#        d1_c1 = 2.*np.transpose(layer2- batch_y)*gaussian1*(batch_X[:,1] - C[0][1])*(synapse1[0]*V[0][0] + synapse1[1]*V[0][1])
#        d2_c0 = 2.*np.transpose(layer2- batch_y)*gaussian2*(batch_X[:,0] - C[1][0])*(synapse1[0]*V[1][0] + synapse1[1]*V[1][1])
#        d2_c1 = 2.*np.transpose(layer2- batch_y)*gaussian2*(batch_X[:,1] - C[1][1])*(synapse1[0]*V[1][0] + synapse1[1]*V[1][1])
        
        

#        dv0=np.mean(np.transpose(layer2- batch_y)*synapse1[0]*gaussian1)
#        dv1=np.mean(np.transpose(layer2- batch_y)*synapse1[1]*gaussian1)
#        dz0=np.mean(np.transpose(layer2- batch_y)*synapse1[0]*gaussian2)
#        dz1=np.mean(np.transpose(layer2- batch_y)*synapse1[1]*gaussian2)
    


#        C[0][0] -= eta*np.mean(d1_c0)
#        C[0][1] -= eta*np.mean(d1_c1)
#        C[1][0] -= eta*np.mean(d2_c0)
#        C[1][1] -= eta*np.mean(d2_c1)
        C -= eta*dc
        
        synapse1[0] -= eta*np.mean(layer1[:,[0]]*(layer2- batch_y))        
        synapse1[1] -= eta*np.mean(layer1[:,[1]]*(layer2- batch_y))
        bias1       -= eta*np.mean(layer2- batch_y)

        
        #V[0][0] = (1-eta*0.000)* V[0][0] - eta*dv0
        #V[0][1] = (1-eta*0.000)* V[0][1] - eta*dv1
        #V[1][0] = (1-eta*0.000)* V[1][0] - eta*dz0
        #V[1][1] = (1-eta*0.000)* V[1][1] - eta*dz1        
        V = (1 - eta*0.000)* V - eta*dv


    return layer1, layer2, C, V, synapse1, batch_y, loss, acc   
########################    OPTIMIZATION METHODS   ########################
       
    




def NN_Naive(training,\
 output,\
 C,\
 V,\
 neurons=1,\
 synapse1= False,\
 bias1 =False,\
 mini_batch_size= False,\
 epochs=60000,\
 eta= .15,\
 seed=False,\
 nonlin = [log_reg, log_reg_d],\
 opt=SGD,\
 cost=[binary_entropy, binary_entropy_d],\
 ):
    """
    Simple implementation of Neural Networks.Currently it is able to deal with
    an arbitrary number of training points andinputs/outputs of any dimension.
    Makes use of 3 layers:
            -one input layer (trainig points x input dimension)
            -one hidden layer (trainig points x number of neurons)
            -one output layer (trainig points x output dimension)
    Inputs:
        training is the numpy array containing all training points;
        [numyp multidimensional array]
            
        output is the numpy array containing the ground truth;
        [numyp multidimensional array]
        
        neurons is the number of neurons in the hidden layer;
        [integer]
        
        mini_batch_size sets the size of each iteration batch, if equals to
        False, full batch'll be used. Same applies for mini batch size higher
        than the entire sample;
        [integer / bool]
            
        epochs sets the number of epochs in the gradient descent process;
        [integer]
        
        eta is the step size for each iteration (a.k.a. learning rate);
        [float]
        
        seed sets the random seed to be used in the stochastic process. If
        False seed will not be specified and results can vary with the same
        inputs;
        [float/ bool]
        
        nonlin is a list containing the activation function and its
        derivative;
        [list of objects]
        
        cost is a list containing the cost function and its derivative;
        [list of objects]
        
        annealing is a function name from the annealing opotions;
        [object]
        
        k is the hyper paramter usedi n the annealng process;
        [float]

        regularization is a list containing the regularization function and
        its derivative;      
        [list of objects]
        
        lambda_r is the regularization hyper paramter.
        [float]
        
    Outputs:
        Synapses and Bias values.
    """

    if seed!=False:
        np.random.seed(seed)    
    
    dim = len(training[0])  #Input dimension
    out = len(output[0])    #Output dimension
    N = len(training)       #Number of training examples
     
    layer0 = training               #Input Layer
    layer1 = np.zeros([N,neurons])  #Hidden Layer
    layer2 = np.zeros([N,out])      #Output Layer
    
    if type(bias1) ==bool:
        bias1 = np.ones(out)              #Set of bias
    
    if type(synapse1) ==bool:
        synapse1=2*np.random.random((neurons,out))-1    #Set of weights
        
    #Call for optimization method and saving new layers/synapses values
    layer1, layer2, C, V, synapse1, pred, loss, acc = opt(layer0, layer1, layer2, C, V, synapse1, bias1, output, epochs,eta, mini_batch_size, nonlin, cost)

    return C, V, synapse1, bias1, layer2 ,pred, loss, acc

