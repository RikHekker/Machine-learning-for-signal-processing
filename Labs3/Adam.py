
from backprop36 import backprop
import scipy.io 
import numpy as np
import matplotlib.pyplot as plt

data=scipy.io.loadmat('Assignment3.mat')


batch_size=100;
no_batches = int(X.shape[1]/batch_size)
y_der= []
J_arr = []
no_iter = 1000


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
#    print("delta : " +  str(delta))
#    print((1-rho_1) * delta)
#    print((1-rho_2) * delta*delta)
    sk+=[rho_1 * sk[-1] + (1-rho_1) * delta]
    rk+=[rho_2 * rk[-1] + (1-rho_2) * delta*delta]
#    print(1-rho_1**k)
#    print(rk)
    sk_ = sk[-1]/(1-rho_1**k)
    rk_ = rk[-1]/(1-rho_2**k)
#    print(sk_)
#    print(rk_)
    return param -  mu / (eps + np.sqrt(rk_)) * sk_
    
    
    
for k in range(1,no_iter):
    for b in range(no_batches):
        start = b*batch_size
        stride = range(start,start+batch_size)
        Xbatch = X[:,stride]
        ybatch = y[:,stride]
        
        z_out, J, dJ_db1, dJ_dW1, dJ_db2, dJ_dw2 = backprop(Xbatch,ybatch,W1,b1,w2,b2)
#        print(dJ_db1)
        
        
        b1=  adam_update(dJ_db1,sk_b1,rk_b1,k,b1)
        b2=  adam_update(dJ_db2,sk_b2,rk_b2,k,b2)
        W1=  adam_update(dJ_dW1,sk_W1,rk_W1,k,W1)
        w2=  adam_update(dJ_dw2,sk_w2,rk_w2,k,w2)
        J_arr+=[J]


def plot_results(z_out,J_arr):

    plt.figure()
    plt.plot(J_arr)
    
    false_idx = np.argwhere(np.round(z_out[0]) == 0)
    true_idx = np.argwhere(np.round(z_out[0]) == 1)
    
    plt.figure()
    plt.scatter(Xbatch[0,false_idx],Xbatch[1,false_idx])
    plt.scatter(Xbatch[0,true_idx],Xbatch[1,true_idx])
    plt.grid(1)

plot_results(z_out,J_arr)

# -*- coding: utf-8 -*-

