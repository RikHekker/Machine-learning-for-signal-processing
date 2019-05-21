
from backprop36 import backprop

import scipy.io 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')


data=scipy.io.loadmat('Assignment3.mat')

rho_1 = 0.9
rho_2 = 0.999
eps = 1e-2
mu = 0.001


def adam_update(delta,sk,rk,k,lam,param):
    sk+=[rho_1 * sk[-1] + (1-rho_1) * delta]
    rk+=[rho_2 * rk[-1] + (1-rho_2) * delta*delta]
    sk_ = sk[-1]/(1-rho_1**k)
    rk_ = rk[-1]/(1-rho_2**k)
    return param -  mu / (eps + np.sqrt(rk_)) * sk_

def adam_update_l2(delta,sk,rk,k,lam,param):
    #print(k)
    sk+=[rho_1 * sk[-1] + (1-rho_1) * delta]
    rk+=[rho_2 * rk[-1] + (1-rho_2) * delta*delta]
    sk_ = sk[-1]/(1-rho_1**k)
    rk_ = rk[-1]/(1-rho_2**k)
    reg = lam * param #0.5 * np.matmul(param.T,param)  
    deel2 = param -  (mu / (eps + np.sqrt(rk_)) * sk_ + reg)
    #print(deel2.shape)
    return deel2 

def adam_update_l1(delta,sk,rk,k,lam,param):
    #print(k)
    sk+=[rho_1 * sk[-1] + (1-rho_1) * delta]
    rk+=[rho_2 * rk[-1] + (1-rho_2) * delta*delta]
    sk_ = sk[-1]/(1-rho_1**k)
    rk_ = rk[-1]/(1-rho_2**k)
    reg = lam * param / (eps +np.linalg.norm(param))
    deel2 = param -  (mu / (eps + np.sqrt(rk_)) * sk_ + reg)
    return deel2 


def converge(update,lam):
    batch_size=100;
    
    J_arr = []
    no_iter = 1000
    
    w2=[data["w2_init"].astype(float)]
    W1=[data["W1_init"].astype(float)]
    b2=[data["b2_init"].astype(float)]
    b1=[data["b1_init"].astype(float)]
    X = data["X"].astype(float)
    y = data["y"].astype(float)
    no_batches = int(X.shape[1]/batch_size)
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
    
    
    for k in range(1,no_iter):
        for b in range(no_batches):
            start = b*batch_size
            stride = range(start,start+batch_size)
            Xbatch = X[:,stride]
            ybatch = y[:,stride]
            
            z_out, J, dJ_db1, dJ_dW1, dJ_db2, dJ_dw2 = backprop(Xbatch,ybatch,W1[-1],b1[-1],w2[-1],b2[-1])
            
            b1 += [update(dJ_db1,sk_b1,rk_b1,k,lam,b1[-1])]
            b2 +=  [update(dJ_db2,sk_b2,rk_b2,k,lam,b2[-1])]
            W1 +=  [update(dJ_dW1,sk_W1,rk_W1,k,lam,W1[-1])]
            w2 +=  [update(dJ_dw2,sk_w2,rk_w2,k,lam,w2[-1])]
            
            
            J_arr+=[J]
    return z_out,J_arr, b1,b2,W1,w2,Xbatch

def plot_results(z_out,J_arr,name, b1,b2,W1,w2,Xbatch):

    plt.figure()
    plt.plot(J_arr)
    plt.xlabel("#Iterations")
    plt.ylabel("Loss")

    plt.savefig("regularizationplots\\"+name + "_Loss.jpg")
    
    false_idx = np.argwhere(np.round(z_out[0]) == 0)
    true_idx = np.argwhere(np.round(z_out[0]) == 1)
    
    plt.figure()
    plt.scatter(Xbatch[0,false_idx],Xbatch[1,false_idx])
    plt.scatter(Xbatch[0,true_idx],Xbatch[1,true_idx])
    plt.grid(1)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.savefig("regularizationplots\\"+name + "_scatter.jpg")
    
    b1 = np.array(b1)
	
    plt.figure()
    plt.plot(b1[:,0,0])
    plt.plot(b1[:,1,0])
    plt.xlabel("#Iterations")
    plt.ylabel("Value")
    plt.grid(1)
	

    plt.savefig("regularizationplots\\"+name+"b1.jpg")
    
    b2 = np.array(b2)
    plt.figure()
    plt.plot(b2[:,0,0])
    plt.savefig("regularizationplots\\"+name+"b2.jpg")
    plt.xlabel("#Iterations")
    plt.ylabel("Value")
    plt.grid(1)

    
    W1 = np.array(W1)
    plt.figure()
    plt.plot(W1[:,0,0])
    plt.plot(W1[:,0,1])
    plt.plot(W1[:,1,0])
    plt.plot(W1[:,1,1])
    plt.xlabel("#Iterations")
    plt.ylabel("Value")
    plt.grid(1)


    plt.savefig("regularizationplots\\"+name+"W1.jpg")
    
    w2 = np.array(w2)
    plt.figure()
    plt.plot(w2[:,0,0])
    plt.plot(w2[:,1,0])
    plt.xlabel("#Iterations")
    plt.ylabel("Value")
    plt.grid(1)

    plt.savefig("regularizationplots\\"+name+"w2.jpg")


z_out,J_arr, b1,b2,W1,w2,Xbatch = converge(adam_update,0)
name = "Basic_adam"
plot_results(z_out,J_arr,name, b1,b2,W1,w2,Xbatch)



for lam in [0.001, 0.01, 0.1]:
    for update in [adam_update_l1,adam_update_l2]:
        z_out,J_arr, b1,b2,W1,w2,Xbatch = converge(update,lam)
        name = "lambda_{0}".format(lam).replace(".","") + "_update_"+ str(update.__name__)
        
        plot_results(z_out,J_arr,name, b1,b2,W1,w2,Xbatch)

# -*- coding: utf-8 -*-

