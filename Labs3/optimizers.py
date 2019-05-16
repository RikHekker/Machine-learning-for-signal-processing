# -*- coding: utf-8 -*-
"""
Created on Thu May 16 15:41:52 2019

@author: s141010
"""

from backprop36 import backprop
import scipy.io 
import numpy as np
import matplotlib.pyplot as plt

data=scipy.io.loadmat('Assignment3.mat')

w2=data["w2_init"].astype(float)
W1=data["W1_init"].astype(float)
b2=data["b2_init"].astype(float)
b1=data["b1_init"].astype(float)
X = data["X"].astype(float)
y = data["y"].astype(float)

batch_size=10;
no_batches = int(X.shape[1]/batch)
y_der= []
J_arr = []
delta = [np.zeros(10)]
def relu(x):
    initialRotation =  np.matmul(w1,x) + b1[np.newaxis,:].T
    maxed =  np.maximum(initialRotation,np.zeros_like(initialRotation))
    return maxed
def sigmoid(x):
    return 1/(1+np.exp(-x))
mu=0.001

for b in range(no_batches):
    start = b*batch_size
    stride = range(start,start+batch_size)
    Xbatch = X[:,stride]
    ybatch = y[:,stride]
    z_out, J, dJ_db1, dJ_dW1, dJ_db2, dJ_dw2 = backprop(Xbatch,ybatch,W1,b1,w2,b2)
    b1-= mu * dJ_db1
    b2-= mu* dJ_db2
    W1-= mu * dJ_dW1
    w2-= mu * dJ_dw2
    J_arr+=[J]
#    a1 = relu(X[:,stride])
#    a2 = sigmoid((np.matmul(w2.T,a1) + b2)[0,0,:])
#    y_der += [a2]
#    loss +=[ -(y[stride]*np.log(y_der[-1])+(1+y[stride])*np.log(1-y_der[-1])).mean(axis=1)]
#    delta += [delta[-1] * loss] 

plt.figure()
plt.plot(J_arr)