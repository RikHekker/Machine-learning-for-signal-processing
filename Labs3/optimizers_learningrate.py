# -*- coding: utf-8 -*-


"""
Created on Thu May 16 15:41:52 2019

@author: s141010
"""


from backprop36 import backprop

import scipy.io 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')
#%% Stochastic gradient descent

data=scipy.io.loadmat('Assignment3.mat')
def load_data():
	w2=[data["w2_init"].astype(float)]
	W1=[data["W1_init"].astype(float)]
	b2=[data["b2_init"].astype(float)]
	b1=[data["b1_init"].astype(float)]
	X = data["X"].astype(float)
	y = data["y"].astype(float)
	J_arr = []

	return w2,W1,b1,b2,X,y,J_arr

def plot_results(J_arr_arr,name,learning_rates):
    plt.figure()

    for idx,J_arr in enumerate(J_arr_arr):
        plt.plot(J_arr,label = "mu ={0}".format(learning_rates[idx]))
    plt.xlabel("#Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("learningrateplots\\"+name + "_Losspermu.jpg")


batch_size=100

no_iter = 1000

learning_rates =[0.001,0.005,0.01,0.03,0.1]
J_arr_arr = []
for mu in  learning_rates:
   w2,W1,b1,b2,X,y,J_arr = load_data()
   no_batches = int(X.shape[1]/batch_size)
     
   for i in range(no_iter):
        for b in range(no_batches):
                start = b*batch_size
                stride = range(start,start+batch_size)
                Xbatch = X[:,stride]
                ybatch = y[:,stride]
                z_out, J, dJ_db1, dJ_dW1, dJ_db2, dJ_dw2 = backprop(Xbatch,ybatch,W1[-1],b1[-1],w2[-1],b2[-1])
                b1+=[b1[-1]- mu * dJ_db1]
                b2+=[b2[-1]- mu* dJ_db2]
                W1+=[W1[-1]- mu * dJ_dW1]
                w2+=[w2[-1] -mu * dJ_dw2]
                J_arr+=[J]
   print("Complete")
   J_arr_arr += [J_arr]

plot_results(J_arr_arr,"SGD",learning_rates)
#plt.show()


#%% # Momentum %%
J_arr_arr = []
for mu in  learning_rates:
    w2,W1,b1,b2,x,y,J_arr = load_data()
    no_batches = int(X.shape[1]/batch_size)
    
    rho = 0.9
    
    dJ_db1_list= [np.zeros_like(b1)]
    dJ_db2_list= [np.zeros_like(b2)]
    dJ_dW1_list= [np.zeros_like(W1)]
    dJ_dw2_list= [np.zeros_like(w2)]
    
    
    for i in range(no_iter):
        for b in range(no_batches):
            start = b*batch_size
            stride = range(start,start+batch_size)
            Xbatch = X[:,stride]
            ybatch = y[:,stride]
                    
            z_out, J, dJ_db1, dJ_dW1, dJ_db2, dJ_dw2 = backprop(Xbatch,ybatch,W1[-1],b1[-1],w2[-1],b2[-1])
            dJ_db1_list += [rho * dJ_db1_list[-1] + dJ_db1]
            dJ_db2_list += [rho * dJ_db2_list[-1] + dJ_db2]
            dJ_dW1_list += [rho * dJ_dW1_list[-1] + dJ_dW1]
            dJ_dw2_list += [rho * dJ_dw2_list[-1] + dJ_dw2]
            
            b1+=[b1[-1]- mu * dJ_db1_list[-1][0]]
            b2+=[b2[-1]- mu* dJ_db2_list[-1][0]]
            
            W1+=[W1[-1]- mu * dJ_dW1_list[-1][0]]
            w2+=[w2[-1]- mu * dJ_dw2_list[-1][0]]
            J_arr+=[J]
    print("Complete")
    J_arr_arr += [J_arr]

plot_results(J_arr_arr,"SGDm",learning_rates)

#%% # Adagrad

J_arr_arr = []
for mu in  learning_rates:
    w2,W1,b1,b2,x,y,J_arr = load_data()
    no_batches = int(X.shape[1]/batch_size)


    rk_W1 = [0]
    rk_w2 = [0]
    rk_b1 = [0]
    rk_b2 = [0]
    
    
    delta = 1e-10
    
    for i in range(no_iter):
        for b in range(no_batches):
            start = b*batch_size
            stride = range(start,start+batch_size)
            Xbatch = X[:,stride]
            ybatch = y[:,stride]
            
            z_out, J, dJ_db1, dJ_dW1, dJ_db2, dJ_dw2 = backprop(Xbatch,ybatch,W1[-1],b1[-1],w2[-1],b2[-1])
            
            
            
            rk_W1 += [dJ_dW1*dJ_dW1 +rk_W1[-1]]
            rk_w2 += [dJ_dw2*dJ_dw2 +rk_w2[-1]]
            rk_b1 += [dJ_db1*dJ_db1 + rk_b1[-1]]
            rk_b2 += [dJ_db2*dJ_db2 + rk_b2[-1]]
            
            b1+=[b1[-1] - mu / (delta + np.sqrt(rk_b1[-1])) * dJ_db1 ]
            b2+=[b2[-1] - mu / (delta + np.sqrt(rk_b2[-1])) * dJ_db2]
            W1+=[W1[-1] - mu / (delta + np.sqrt(rk_W1[-1])) * dJ_dW1]
            w2+=[w2[-1] - mu / (delta + np.sqrt(rk_w2[-1])) * dJ_dw2]
            J_arr+=[J]
    print("Complete")
    J_arr_arr += [J_arr]

plot_results(J_arr_arr,"adaGrad",learning_rates)

#%% # RMSPROP

J_arr_arr = []
for mu in  learning_rates:
    w2,W1,b1,b2,x,y,J_arr = load_data()
    no_batches = int(X.shape[1]/batch_size)

    rho = 0.999
    rk_W1 = [0]
    rk_w2 = [0]
    rk_b1 = [0]
    rk_b2 = [0]
    
    delta = 1e-10
    
    for i in range(no_iter):
        for b in range(no_batches):
            start = b*batch_size
            stride = range(start,start+batch_size)
            Xbatch = X[:,stride]
            ybatch = y[:,stride]
            
            z_out, J, dJ_db1, dJ_dW1, dJ_db2, dJ_dw2 = backprop(Xbatch,ybatch,W1[-1],b1[-1],w2[-1],b2[-1])
            
            
            
            rk_W1 += [dJ_dW1*dJ_dW1 *(1-rho) +rho*rk_W1[-1]]
            rk_w2 += [dJ_dw2*dJ_dw2*(1-rho) +rho* rk_w2[-1]]
            rk_b1 += [dJ_db1*dJ_db1*(1-rho) +rho* rk_b1[-1]]
            rk_b2 += [dJ_db2*dJ_db2*(1-rho) +rho* rk_b2[-1]]
            
            b1+=[b1[-1]-  mu / (delta + np.sqrt(rk_b1[-1])) * dJ_db1 ]
            b2+=[b2[-1]- mu / (delta + np.sqrt(rk_b2[-1])) * dJ_db2]
            W1+=[W1[-1]- mu / (delta + np.sqrt(rk_W1[-1])) * dJ_dW1]
            w2+=[w2[-1]- mu / (delta + np.sqrt(rk_w2[-1])) * dJ_dw2]
            J_arr+=[J]
    print("Complete")
    J_arr_arr += [J_arr]

plot_results(J_arr_arr,"RMSprop",learning_rates)

##

#%% # adam

J_arr_arr = []
for mu in  learning_rates:
    w2,W1,b1,b2,x,y,J_arr = load_data()
    no_batches = int(X.shape[1]/batch_size)
    rho = 0.999
    rk_W1 = [0]
    rk_w2 = [0]
    rk_b1 = [0]
    rk_b2 = [0]
    
    sk_W1 = [0]
    sk_w2 = [0]
    sk_b1 = [0]
    sk_b2 = [0]
    
    
    
    
    eps = 1e-10
    rho_1 = 0.9
    rho_2 = 0.999
    mu = 0.0001
    
    def adam_update(delta,sk,rk,k,param):
        sk+=[rho_1 * sk[-1] + (1-rho_1) * delta]
        rk+=[rho_2 * rk[-1] + (1-rho_2) * delta*delta]
        sk_ = sk[-1]/(1-rho_1**k)
        rk_ = rk[-1]/(1-rho_2**k)
        return param -  mu / (eps + np.sqrt(rk_)) * sk_
        
    for k in range(1,no_iter):
        for b in range(no_batches):
            start = b*batch_size
            stride = range(start,start+batch_size)
            Xbatch = X[:,stride]
            ybatch = y[:,stride]
            
            z_out, J, dJ_db1, dJ_dW1, dJ_db2, dJ_dw2 = backprop(Xbatch,ybatch,W1[-1],b1[-1],w2[-1],b2[-1])
            #print(dJ_db1)
            
            
            b1+=[adam_update(dJ_db1,sk_b1,rk_b1,k,b1[-1])]
            b2+=[adam_update(dJ_db2,sk_b2,rk_b2,k,b2[-1])]
            W1+=[adam_update(dJ_dW1,sk_W1,rk_W1,k,W1[-1])]
            w2+=[adam_update(dJ_dw2,sk_w2,rk_w2,k,w2[-1])]
            J_arr+=[J]
    print("Complete")
    J_arr_arr += [J_arr]

plot_results(J_arr_arr,"adam",learning_rates)