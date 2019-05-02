import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data=pd.read_csv('assignment1_data.csv')
x=data.A
y=data.B
alpha=1/20
w_o = np.array([0.2 ,1, -0.5])
w = [w_o]
w= [np.array([0,0,0])]
R_x = [[5, -1, -2],[-1 ,5 ,-1],[-2,-1,5]]
r_yx = [1,5.3,-3.9]


def plot_weights(w,y):
    R_x = [[5, -1, -2],[-1 ,5 ,-1],[-2,-1,5]]
    r_yx = [1,5.3,-3.9]
    w2 = -0.5
    resolution = 100
    r_yx = np.array(r_yx)
    Jmin = np.mean(y**2)-  np.matmul(np.matmul(r_yx.T,np.invert(R_x)),r_yx) 

    J = np.zeros((resolution,resolution))
    x = np.linspace(-1,2,resolution) 
    y = x
    X,Y = np.meshgrid(x,y)
    
    for w0_idx,w0 in enumerate(x):
        for w1_idx,w1 in enumerate(y):
            dw = np.array([w0,w1,w2])-w_o
            J[w0_idx,w1_idx] = Jmin + np.matmul(np.matmul(dw.T,R_x),dw)  
          
    plt.figure()
    plt.contour(X,Y,J.T,20)
    plt.plot(w[:,0],w[:,1])
    plt.xlabel('w_0')
    plt.ylabel('w_1')
    
    plt.show()



# Gradient descent
N = y.shape[0]
for k in range(N):
    w +=   [w[-1] + 2*alpha*(r_yx-np.matmul(R_x,w[-1]))]

w = np.array(w)
plot_weights(w,y)





## Newtons method
alpha = 0.5
w= [np.array([0,0,0])]
Rinv = np.linalg.inv(R_x)
for k in range(N):

    w += [ w[-1] + np.matmul( 2*alpha*Rinv,(r_yx-np.matmul(R_x,w[-1])))]

w = np.array(w)
plot_weights(w,y)


## LMS

alpha = 1e-3
w= [np.array([0,0,0])]
Rinv = np.linalg.inv(R_x)
y_pred = []
e = []
for k in range(1,N-1):
    inp = x[k-1:k+2]
    y_pred += [np.sum(inp * w[-1])]
    e += [y[k]-y_pred[-1]]         
    w += [ w[-1] + 2 * alpha * np.array(inp) * e[-1]]
    

w = np.array(w)
w = np.array([w[:,1],w[:,0],w[:,2]]).T
plot_weights(w,y)


## NLMS

alpha = 1e-3
w= [np.array([0,0,0])]
Rinv = np.linalg.inv(R_x)
y_pred = []
e = []
eps = 1e-5
for k in range(1,N-1):
    inp = np.array(x[k-1:k+2])
    y_pred += [np.sum(inp * w[-1])]
    e += [y[k]-y_pred[-1]] 
    sigma = np.matmul(inp.T,inp)/3 + eps         
    w += [ w[-1] + 2 * alpha/sigma * inp * e[-1]]
    

w = np.array(w)
w = np.array([w[:,1],w[:,0],w[:,2]]).T
plot_weights(w,y)


## RLS
w= [np.array([0,0,0])]
sigma = 1e-3
r_yx_pred = np.zeros(3)
R_xinv_pred = np.eye(3) * sigma
gamma = 1-1e-4
y_pred = []
e = []
for k in range(1,N-2):
    inp_next = np.array(x[k:k+3])
    # g_next_denominator is a scalar
    g_next_denominator = gamma**2 + inp_next.T*R_xinv_pred*inp_next

    g_next = R_xinv_pred*inp_next/g_next_denominator
    R_xinv_pred_next = gamma**-2 * (R_xinv_pred-np.dot(g_next,inp_next.T)*R_xinv_pred)
    r_yx_pred_next = gamma**2 * r_yx_pred + inp_next.T*y[k+1]
    w+=[np.dot(R_xinv_pred_next, r_yx_pred_next)]
    r_yx_pred = r_yx_pred_next
    R_xinv_pred = R_xinv_pred_next

w = np.array(w)  
w = np.array([w[:,1],w[:,0],w[:,2]]).T
plot_weights(w,y)

 


