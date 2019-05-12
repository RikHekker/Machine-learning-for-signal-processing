import numpy as np# -*- coding: utf-8 -*-
## Q3 MSE
def calculate_MSE_weights(x,y):
    x_ = x.mean(axis=0)
    y_ = y.mean()
    xy__ = [(x[:,0]*y).mean(),(x[:,1]*y).mean()]
    w = (xy__ - x_*y_)/((x**2).mean(axis=0) - x_**2)
    b = y_ - np.matmul(w.T, x_) 
    return w,b

x = np.array([[0,0],[0.1,1],[1,0.2],[1,1]])
y = np.array([0,0.41,0.18,0.5])

w,b = calculate_MSE_weights(x,y)

y_pred = []
e = []
for i in range(4):
    y_pred += [np.matmul(w.T,x[i]) + b]
    e += [abs(y_pred[-1] - y[i])]

## Recalculating inputs

x = np.array([[0,0],[0.1,1],[1,0.2],[1,1]])
y = np.array([-0.416,0.3610,0.1222,0.473])

w,b = calculate_MSE_weights(x,y)


y_pred = []
e = []
for i in range(4):
    y_pred += [np.matmul(w.T,x[i]) + b]
    e += [abs(y_pred[-1] - y[i])]
    
    
    
    
    
    