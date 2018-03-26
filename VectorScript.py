# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 17:23:17 2018

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
import VectorNetwork as gv

##############################PLOTS##############################
def loss_epochs(loss, eta, epochs, dataset, n, std = False):
    
    f = plt.figure()
    epochs = np.arange(len(loss))
    plt.plot(epochs , loss, '-b', label = "Loss")
    if type(std)!=bool:
        plt.plot(epochs, loss - std, '--r', label = "Standard Deviation")
        plt.plot(epochs, loss + std, '--r')
    #plt.title("Train Cost vs Epochs \n %s Dataset - eta = %.2f - %d Samples"%(dataset, eta,n))
    #plt.legend()
    plt.xticks(fontsize=20, rotation=0)
    plt.yticks(fontsize=20, rotation=90)
    plt.ylim(100,200)
    plt.savefig("Cost%s%f.png"%(dataset,eta))
    plt.close()
def acc_epochs(acc, eta, epochs, dataset, n, std= False):
    
    f = plt.figure()
    epochs = np.arange(len(acc))
    plt.plot(epochs , acc, '-b',label = "Accuracy")
    if type(std)!=bool:
        plt.plot(epochs, acc - std,'--r',label = "Standard Deviation")
        plt.plot(epochs, acc + std,'--r')
    #plt.title("Train Accuracy vs Epochs \n %s Dataset - eta = %.2f - %d Samples"%(dataset, eta, n))
    #plt.legend()
    plt.xticks(fontsize=20, rotation=0)
    plt.yticks(fontsize=20, rotation=90)
    plt.ylim(0,1)
    plt.savefig("Accuracy%s%f.png"%(dataset,eta))
    plt.close()





def orignal_boundary(X, y, malha, y_, c, eta, epochs, dataset):
    size = len(X)
    f = plt.figure()
    for i in xrange(size):
        if y[i]==0:
            plt.plot(X[i,0], X[i,1], 'wo', label = 'Class 1')
        else:
            plt.plot(X[i,0], X[i,1], 'k^', label = 'Class 2')

    teste = plt.tricontourf(malha[:,0], malha[:,1], y_[:, 0], 200, cmap = cm.gist_heat)
    #f.colorbar(teste, shrink=0.5, aspect=5)       
    plt.xticks(fontsize=20, rotation=0)
    plt.yticks(fontsize=20, rotation=90)
    #plt.plot(c1[0], c1[1], 'go', markersize= 10)
    #plt.plot(c2[0], c2[1], 'yo', markersize= 10)
    #plt.title("Orignal Space Boundary \n %s Dataset - eta = %f"%(dataset, eta))
    plt.savefig("Original Boundary  - %s - %f .jpg"%(dataset,eta))
    plt.close()



def transformed_boundary(X, y, malha, y_, c, eta, epochs, dataset):
    size = len(X)
    f = plt.figure()
    for i in xrange(size):
        if y[i]==0:
            plt.plot(X[i,0], X[i,1], 'wo', label = 'Class 1')
        else:
            plt.plot(X[i,0], X[i,1], 'k^', label = 'Class 2')

    teste = plt.tricontourf(malha[:,0], malha[:,1], y_[:, 0], 200, cmap = cm.gist_heat)
    #f.colorbar(teste, shrink=0.5, aspect=5, fontsize=20)      
    plt.xticks(fontsize=20, rotation=0)
    plt.yticks(fontsize=20, rotation=90)
    #plt.plot(c1[0], c1[1], 'go', markersize= 10)
    #plt.plot(c2[0], c2[1], 'yo', markersize= 10)
    #plt.title("Transformed Space Boundary \n %s Dataset - eta = %f"%(dataset, eta))
    plt.savefig("Transformed Boundary NOREG - %s - %f .jpg"%(dataset,eta))
    plt.close()

##############################PLOTS##############################

def plot_points(X, y,X0, y0, C1, C2, contour = False):
    f = plt.figure()    
    if contour == True:
        
        teste=plt.tricontourf(X0[:,0], X0[:,1],y0[:,0],200,cmap=cm.gist_heat) # choose 200 contour levels, just to show how good its interpolation is
        f.colorbar(teste, shrink=0.5, aspect=5)        
    else:
        for i in xrange(len(y)):
  
            if y0[i]>0.5:
                plt.plot(X0[i,0],X0[i,1],'bx', lw=0, markersize= 10)    
            else:
                plt.plot(X0[i,0],X0[i,1],'rx', lw=0, markersize= 10)   


    for i in xrange(len(y)):
        if y[i]>0.5:
            plt.plot(X[i,0],X[i,1],'bo', lw=0, markersize= 10)    
     
        else:
            plt.plot(X[i,0],X[i,1],'ro', lw=0, markersize= 10)

    plt.plot(C1[0],C1[1],'gx', lw=0, markersize=10)  
    plt.plot(C2[0],C2[1],'go', lw=0, markersize=10)  
    
    
##############################BOUNDARY MESH##############################

def sin_boundary():
    start = -3.5
    stop =  3.5
    size = 20
    it = np.linspace(start, stop, size)
    X = np.zeros([size*size ,2])
    for i in xrange(size):
        for j in xrange(size):
            X[i*size + j,0] = it[j]
            X[i*size + j,1] = it[i]
     
    return X


def moon_boundary():
    start = -1.5
    stop =  2.5
    size = 20
    it = np.linspace(start, stop, size)
    X = np.zeros([size*size ,2])
    y = np.zeros([size*size,1])
    for i in xrange(size):
        for j in xrange(size):
            X[i*size + j,0] = it[j]
            X[i*size + j,1] = it[i]
    return X


def circle_boundary():
    start = -1.5
    stop =  1.5
    size = 20
    it = np.linspace(start, stop, size)
    X = np.zeros([size*size ,2])
    y = np.zeros([size*size,1])
    for i in xrange(size):
        for j in xrange(size):
            X[i*size + j,0] = it[j]
            X[i*size + j,1] = it[i]
    return X
##############################BOUNDARY MESH##############################
def make_sin(size):
    start = -np.pi
    stop =  np.pi
    X = np.random.random_sample([200,2])*(stop- start) + start
    y = np.zeros([size,1])
    for i in xrange(size):
        if X[i,1]<np.sin(X[i, 0]):
            y[i,0] = 1.
        else:
            y[i,0]= 0.

    return X, y








#X, y = datasets.make_moons(n_samples=N, noise= 0.1)
#y = np.transpose(np.array([y]))
#start, stop = -1.5, 2.5
#dataset = "Moons"
#malha = moon_boundary()
##
#X, y = datasets.make_circles(n_samples=N, factor = 0.2, noise = 0.1)
#y = np.transpose(np.array([y]))
#start, stop = -1.5, 1.5 
#dataset = "Circles"
#malha = circle_boundary()
#
#
#X, y = make_sin(N)
#start, stop = -3.5, 3.5 
#dataset = "Sin"
#sin_boundary()
#malha = sin_boundary()
ds = ['banknote.txt', 'cryo', 'imunotherapy.txt', 'ionosphere.data.txt', 'pima-indians-diabetes.data.txt']

load= np.genfromtxt(ds[0], delimiter= ',')
X= load[:,0:len(load[0])-1]
X = (X - X.min(axis=0))/(X.max(axis=0) - X.min(axis=0))
y = np.transpose(np.array([load[:,len(load[0])-1]]))
X = X*1.
y = y*1.
        

dataset= "test"
start , stop = 0. , 1.


n = 1
dim = len(X[0])
centros = [0.5,1.,1.5]
N= len(X)
epochs = 300
eta = np.array([0.3])
seed = 0
np.random.seed(seed)




 #Set of gaussian weight
 #Set of vectors
V = np.random.random_sample([centros[1]*dim, dim])
loss_mean= np.zeros([n, epochs])
acc_mean = np.zeros([n, epochs])
for j in eta:
    print j
    for i in xrange(n):
        C= np.random.random_sample([centros[1]*dim, dim])*(stop- start) + start
        centro, vetor, s1, b1, l2, pred, loss, acc = gv.NN_Naive(X, y, C, V, dim, epochs = epochs, seed = seed, eta = j)
        loss_mean[i] = loss
        acc_mean[i] = acc
        
    #acc_std = np.std(acc_mean, axis = 0)    
   # loss_std = np.std(loss_mean, axis = 0)    
    #loss = np.mean(loss_mean, axis = 0) 
    #acc = np.mean(acc_mean, axis = 0) 

    #loss_epochs(loss, j, epochs, dataset, n, std = loss_std)
    
    #acc_epochs(acc, j, epochs, dataset, n, std = acc_std)  
  
#    malha_t, y_ = gv.predict(malha, vetor, centro, 400, s1, b1)    
#    orignal_boundary(X, y, malha, y_, centro, j, epochs, dataset)
#    
#
#
#    X_t, y_t = gv.predict(X, vetor, centro, N, s1, b1)        
#    transformed_boundary(X_t, y, malha_t, y_, centro, j, epochs, dataset)
    
print acc




#layer1,layer2 = gv.predict(X, V, Z, centros1, centros2, layer1, layer2, s1, b1)














