# -*- coding: utf-8 -*-
"""
Created on Thu May 16 15:41:52 2019

@author: s141010
"""

#import backprop36 as backprop
import scipy.io 
import numpy as np

data=scipy.io.loadmat('Assignment3.mat')

w2=data["w2_init"]
w1=data["W1_init"]
b2=data["b2_init"]
b1=data["b1_init"]
batch=10;
f=w1*data["X"]+b1
y_der=np.transpose(w2)*np.maximum(np.zeros(2,100),f)+b2

mu=0.001
    