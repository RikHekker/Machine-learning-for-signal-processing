import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data=pd.read_csv('assignment1_data.csv')
x=data.A
y=data.B
alpha=1/8
w_o = np.array([0.2 ,1, -0.5])
w = [w_o]
w= [np.array([0,0,0])]
R_x = [[5, -1, -2],[-1 ,5 ,-1],[-2,-1,5]]
r_yx = [1,5.3,-3.9]

N = y.shape[0]
y = []
for k in range(3,N):
    inp = x[k-3:k]
    y = np.sum(inp * w[-1])
    #R_x = np.matmul(inp * inp.T)
    #r_yx = np.matmul(inp.t,y)
    
    w +=   [w[-1] + 2*alpha*(r_yx-np.matmul(R_x,w[-1]))]

#w = np.array(w)
#plt.figure()
#plt.plot(w[:,0])
#plt.plot(w[:,1])
#plt.plot(w[:,2])
#
#plt.show()
r_yx = np.array(r_yx)
Jmin = np.mean(y**2)-  np.matmul(np.matmul(r_yx.T,np.invert(R_x)),r_yx) 
w2 = -0.5
resolution = 100
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