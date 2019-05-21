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

batch_size=100;
no_batches = int(X.shape[1]/batch_size)
y_der= []
J_arr = []
no_iter = 50000

def relu(x):
    initialRotation =  np.matmul(W1,x) + b1[np.newaxis,:].T
    maxed =  np.maximum(initialRotation,np.zeros_like(initialRotation))
    return maxed
def sigmoid(x):
    return 1/(1+np.exp(-x))
mu=0.001

for i in range(no_iter):
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

false_idx = np.argwhere(np.round(z_out[0]) == 0)
true_idx = np.argwhere(np.round(z_out[0]) == 1)
    
plt.figure()
plt.scatter(Xbatch[0,false_idx],Xbatch[1,false_idx])
plt.scatter(Xbatch[0,true_idx],Xbatch[1,true_idx])
plt.grid(1)




w2=data["w2_init"].astype(float)
W1=data["W1_init"].astype(float)
b2=data["b2_init"].astype(float)
b1=data["b1_init"].astype(float)
X = data["X"].astype(float)
y = data["y"].astype(float)
J_arr = []
rho = 0.9

dJ_db1_list= [np.zeros_like(b1)]
dJ_db2_list= [np.zeros_like(b2)]
dJ_dW1_list= [np.zeros_like(W1)]
dJ_dw2_list= [np.zeros_like(w2)]

no_iter = 50000



for i in range(no_iter):
    for b in range(no_batches):
        start = b*batch_size
        stride = range(start,start+batch_size)
        Xbatch = X[:,stride]
        ybatch = y[:,stride]
        
        z_out, J, dJ_db1, dJ_dW1, dJ_db2, dJ_dw2 = backprop(Xbatch,ybatch,W1,b1,w2,b2)
        dJ_db1_list += [rho * dJ_db1_list[-1] + dJ_db1]
        dJ_db2_list += [rho * dJ_db2_list[-1] + dJ_db2]
        dJ_dW1_list += [rho * dJ_dW1_list[-1] + dJ_dW1]
        dJ_dw2_list += [rho * dJ_dw2_list[-1] + dJ_dw2]
        
        b1-= mu * dJ_db1_list[-1]
        b2-= mu* dJ_db2_list[-1]
        W1-= mu * dJ_dW1_list[-1]
        w2-= mu * dJ_dw2_list[-1]
        J_arr+=[J]


plt.figure()
plt.plot(J_arr)

false_idx = np.argwhere(np.round(z_out[0]) == 0)
true_idx = np.argwhere(np.round(z_out[0]) == 1)
    
plt.figure()
plt.scatter(Xbatch[0,false_idx],Xbatch[1,false_idx])
plt.scatter(Xbatch[0,true_idx],Xbatch[1,true_idx])
plt.grid(1)

## Adagrad

w2=data["w2_init"].astype(float)
W1=data["W1_init"].astype(float)
b2=data["b2_init"].astype(float)
b1=data["b1_init"].astype(float)
X = data["X"].astype(float)
y = data["y"].astype(float)
J_arr= []

rk_W1 = [0]
rk_w2 = [0]
rk_b1 = [0]
rk_b2 = [0]

no_iter = 50000

delta = 1e-10

for i in range(no_iter):
    for b in range(no_batches):
        start = b*batch_size
        stride = range(start,start+batch_size)
        Xbatch = X[:,stride]
        ybatch = y[:,stride]
        
        z_out, J, dJ_db1, dJ_dW1, dJ_db2, dJ_dw2 = backprop(Xbatch,ybatch,W1,b1,w2,b2)
        
        
        
        rk_W1 += [dJ_dW1*dJ_dW1 +rk_W1[-1]]
        rk_w2 += [dJ_dw2*dJ_dw2 +rk_w2[-1]]
        rk_b1 += [dJ_db1*dJ_db1 + rk_b1[-1]]
        rk_b2 += [dJ_db2*dJ_db2 + rk_b2[-1]]
        
        b1-=  mu / (delta + np.sqrt(rk_b1[-1])) * dJ_db1 
        b2-= mu / (delta + np.sqrt(rk_b2[-1])) * dJ_db2
        W1-= mu / (delta + np.sqrt(rk_W1[-1])) * dJ_dW1
        w2-= mu / (delta + np.sqrt(rk_w2[-1])) * dJ_dw2
        J_arr+=[J]


plt.figure()
plt.plot(J_arr)

false_idx = np.argwhere(np.round(z_out[0]) == 0)
true_idx = np.argwhere(np.round(z_out[0]) == 1)
    
plt.figure()
plt.scatter(Xbatch[0,false_idx],Xbatch[1,false_idx])
plt.scatter(Xbatch[0,true_idx],Xbatch[1,true_idx])
plt.grid(1)

## RMSPROP


w2=data["w2_init"].astype(float)
W1=data["W1_init"].astype(float)
b2=data["b2_init"].astype(float)
b1=data["b1_init"].astype(float)
X = data["X"].astype(float)
y = data["y"].astype(float)
J_arr= []
rho = 0.999
rk_W1 = [0]
rk_w2 = [0]
rk_b1 = [0]
rk_b2 = [0]

no_iter = 50000

delta = 1e-10

for i in range(no_iter):
    for b in range(no_batches):
        start = b*batch_size
        stride = range(start,start+batch_size)
        Xbatch = X[:,stride]
        ybatch = y[:,stride]
        
        z_out, J, dJ_db1, dJ_dW1, dJ_db2, dJ_dw2 = backprop(Xbatch,ybatch,W1,b1,w2,b2)
        
        
        
        rk_W1 += [dJ_dW1*dJ_dW1 *(1-rho) +rho*rk_W1[-1]]
        rk_w2 += [dJ_dw2*dJ_dw2*(1-rho) +rho* rk_w2[-1]]
        rk_b1 += [dJ_db1*dJ_db1*(1-rho) +rho* rk_b1[-1]]
        rk_b2 += [dJ_db2*dJ_db2*(1-rho) +rho* rk_b2[-1]]
        
        b1-=  mu / (delta + np.sqrt(rk_b1[-1])) * dJ_db1 
        b2-= mu / (delta + np.sqrt(rk_b2[-1])) * dJ_db2
        W1-= mu / (delta + np.sqrt(rk_W1[-1])) * dJ_dW1
        w2-= mu / (delta + np.sqrt(rk_w2[-1])) * dJ_dw2
        J_arr+=[J]


plt.figure()
plt.plot(J_arr)

false_idx = np.argwhere(np.round(z_out[0]) == 0)
true_idx = np.argwhere(np.round(z_out[0]) == 1)
    
plt.figure()
plt.scatter(Xbatch[0,false_idx],Xbatch[1,false_idx])
plt.scatter(Xbatch[0,true_idx],Xbatch[1,true_idx])
plt.grid(1)

##

## RMSPROP


w2=data["w2_init"].astype(float)
W1=data["W1_init"].astype(float)
b2=data["b2_init"].astype(float)
b1=data["b1_init"].astype(float)
X = data["X"].astype(float)
y = data["y"].astype(float)
J_arr= []
rho = 0.999
rk_W1 = [0]
rk_w2 = [0]
rk_b1 = [0]
rk_b2 = [0]

sk_W1 = [0]
sk_w2 = [0]
sk_b1 = [0]
sk_b2 = [0]


no_iter = 50000

eps = 1e-10
rho_1 = 0.9
rho_2 = 0.999
mu = 0.0001

def adam_update(delta,sk,rk,k,param):
#    print(delta)
#    print((1-rho_1) * delta)
#    print((1-rho_2) * delta*delta)
    sk+=[rho_1 * sk[-1] + (1-rho_1) * delta]
    rk+=[rho_2 * rk[-1] + (1-rho_2) * delta*delta]
#    print(sk)
#    print(rk)
    sk_ = sk[-1]/(1-rho_1**k)
    rk_ = rk[-1]/(1-rho_2**k)
#    print(sk_)
#    print(rk_)
    return param -  mu / (eps + np.sqrt(rk_)) * sk_
    
    
    
for k in range(no_iter):
    for b in range(no_batches):
        start = b*batch_size
        stride = range(start,start+batch_size)
        Xbatch = X[:,stride]
        ybatch = y[:,stride]
        
        z_out, J, dJ_db1, dJ_dW1, dJ_db2, dJ_dw2 = backprop(Xbatch,ybatch,W1,b1,w2,b2)
        print(dJ_db1)
        
        
        b1=  adam_update(dJ_db1,sk_b1,rk_b1,k,b1)
        b2=  adam_update(dJ_db2,sk_b2,rk_b2,k,b2)
        W1=  adam_update(dJ_dW1,sk_W1,rk_W1,k,W1)
        w2=  adam_update(dJ_dw2,sk_w2,rk_w2,k,w2)
        J_arr+=[J]


plt.figure()
plt.plot(J_arr)

    
false_idx = np.argwhere(np.round(z_out[0]) == 0)
true_idx = np.argwhere(np.round(z_out[0]) == 1)
    
plt.figure()
plt.scatter(Xbatch[0,false_idx],Xbatch[1,false_idx])
plt.scatter(Xbatch[0,true_idx],Xbatch[1,true_idx])
plt.grid(1)