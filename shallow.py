# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

def transform_linear(inp):
    rotated = np.matmul(W,inp) + b1[np.newaxis,:].T
    maxed= rotated.clip(0,np.max(rotated))
    return maxed

X = np.array([[0,0,1,1],[0,1,0,1]])

x0_arr = np.linspace(0,0.5,100)
x1_arr = 0.5 - x0_arr 
decision = np.array([x0_arr,x1_arr])


plt.figure()
plt.plot(X[0,0],X[1,0],'r o')
plt.plot(X[0,1],X[1,1],'g o')
plt.plot(X[0,2],X[1,2],'g o')
plt.plot(X[0,3],X[1,3],'r o')
plt.plot(decision[0],decision[1],'--')
plt.grid(1)


W = np.array([[1 ,1],[1,1]])
w2 = np.array([1, -2])
b1 = np.array([0,-1])
b2 = 0



h = transform_linear(X)
decision_tranformed = transform_linear(decision)
print(decision_tranformed)
plt.figure()
plt.plot(h[0,[0,3]],h[1,[0,3]],'r o')
plt.plot(h[0,[1,2]],h[1,[1,2]],'g o')
plt.plot(decision_tranformed[0],decision_tranformed[1],'--')

plt.grid(1)
plt.show()




