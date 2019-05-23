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

w2,W1,b1,b2,X,y,J_arr = load_data()



batch_size=100
no_batches = int(X.shape[1]/batch_size)
no_iter = 50000
mu=0.0001

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

def plot_results(z_out,J_arr,name, b1,b2,W1,w2):

    plt.figure()
    plt.plot(J_arr)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig("plots\\"+name + "_Loss.jpg")
    
    false_idx = np.argwhere(np.round(z_out[0]) == 0)
    true_idx = np.argwhere(np.round(z_out[0]) == 1)
    
    plt.figure()
    plt.scatter(Xbatch[0,false_idx],Xbatch[1,false_idx],label = "0")
    plt.scatter(Xbatch[0,true_idx],Xbatch[1,true_idx],label = "1")
    plt.grid(1)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    plt.savefig("plots\\"+name + "_scatter.jpg")
    
    b1 = np.array(b1)
	
    plt.figure()
    plt.plot(b1[:,0,0],label = "b1_1")
    plt.plot(b1[:,1,0],label = "b1_2")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(1)
	

    plt.savefig("plots\\"+name+"b1.jpg")
    
    b2 = np.array(b2)
    plt.figure()
    plt.plot(b2[:,0,0])
    plt.savefig("plots\\"+name+"b2.jpg")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.grid(1)

    
    W1 = np.array(W1)
    plt.figure()
    plt.plot(W1[:,0,0],label = "W1_11")
    plt.plot(W1[:,0,1],label = "W1_12")
    plt.plot(W1[:,1,0],label = "W1_21")
    plt.plot(W1[:,1,1],label = "W1_22")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(1)


    plt.savefig("plots\\"+name+"W1.jpg")
    
    w2 = np.array(w2)
    plt.figure()
    plt.plot(w2[:,0,0],label="w2_1")
    plt.plot(w2[:,1,0],label="w2_2")
    
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(1)

    plt.savefig("plots\\"+name+"w2.jpg")

plot_results(z_out,J_arr,"SGD",b1,b2,W1,w2)
#plt.show()
J_SGD = J_arr

#%% # Momentum %%
w2,W1,b1,b2,x,y,J_arr = load_data()

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


plot_results(z_out,J_arr,"SGDm",b1,b2,W1,w2)
J_SGDm = J_arr
#%% # Adagrad

w2,W1,b1,b2,x,y,J_arr = load_data()

J_arr= []

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
        
        b1+=[b1[-1]-  mu / (delta + np.sqrt(rk_b1[-1])) * dJ_db1 ]
        b2+=[b2[-1]- mu / (delta + np.sqrt(rk_b2[-1])) * dJ_db2]
        W1+=[W1[-1]- mu / (delta + np.sqrt(rk_W1[-1])) * dJ_dW1]
        w2+=[w2[-1]- mu / (delta + np.sqrt(rk_w2[-1])) * dJ_dw2]
        J_arr+=[J]

plot_results(z_out,J_arr,"adaGrad",b1,b2,W1,w2)

J_adagrad = J_arr
#%% # RMSPROP

w2,W1,b1,b2,x,y,J_arr = load_data()

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

plot_results(z_out,J_arr,"RMSprop",b1,b2,W1,w2)
J_RMS = J_arr
##

#%% # adam

w2,W1,b1,b2,x,y,J_arr = load_data()
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

plot_results(z_out,J_arr,"adam",b1,b2,W1,w2)

J_adam = J_arr

#%%
plt.figure()
plt.plot(J_SGD,label = "SGD")
plt.plot(J_SGDm,label = "SGDm")
plt.plot(J_adagrad,label = "AdaGrad")
plt.plot(J_RMS,label = "RMSProp")
plt.plot(J_adam,label = "Adam")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(1)
plt.savefig("plots\\compare_Loss.jpg")



