# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
y1 = [0, 1]
y2 = [1, 0]
x1 = [0, 1]

plt.figure()
plt.plot(x1,y1,'o')
plt.plot(x1,y2,'+')


X = np.array([[0,0,1,1],[0,1,0,1]])

W = np.array([[1 ,1],[1,1]])
w2 = np.array([1, -2])
b1 = np.array([0,-1])
b2 = 0

l = np.matmul(W,X) + b1[np.newaxis,:].T
h= l.clip(0,np.max(l))

plt.figure()
plt.plot(h[0,[0,3]],h[1,[0,3]],'o')
plt.plot(h[0,[1,2]],h[1,[1,2]],'x')
plt.grid(1)

y = np.matmul(w2,l) + b2

x1_arr = np.ones(100)*0.5
x0_arr = np.linspace(-1,3,100)

x = np.array([x0_arr,x1_arr])


l = np.matmul(W,x) + b1[np.newaxis,:].T
h= l.clip(0,np.max(l))
plt.plot(h[0],h[1])