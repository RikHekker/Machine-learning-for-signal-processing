# -*- coding: utf-8 -*-
"""
Created on Wed May  8 18:46:22 2019


Load MNIST dataset and implement a deterministic autoencoder with only a few layers to do manifold learning

@author: rvulling
"""

import struct as st
from collections import defaultdict

import numpy as np
import math
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, Conv2D, LeakyReLU, AvgPool2D, UpSampling2D, ReLU, MaxPooling2D, Reshape
from tensorflow.keras.losses import MSE
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam

#from IPython.display import clear_output
import matplotlib.pyplot as plt

import sklearn
#%%
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import fetch_openml
#%%
epochs = 13
batch_size=10000


def build_batches(x, n):
    m = (x.shape[0] // n) * n
    return x[:m].reshape(-1, n, *x.shape[1:])

def get_mnist32_batches(batch_size, data_format='channels_last'):
    maxNum_data_train=10000 #reduce data size for computational load
    maxNum_data_test=1000
    channel_index = 1 if data_format == 'channels_first' else 3
    mnist = fetch_openml('mnist_784')
    data_x = mnist['data'].reshape(-1,28,28).astype(np.float32) / 255.
    print(data_x.shape)
    #Reduce dimensions of dataset to reduce computations times
    np.random.seed(42) #seed to ensure reproducible results
    randomIndices=np.random.permutation(np.size(data_x,0))
    indicesTrain=randomIndices[0:maxNum_data_train]
    indicesTest=randomIndices[np.size(data_x,0)-maxNum_data_test:np.size(data_x,0)]
    data_x_train=  data_x[indicesTrain,:,:] #Reduce dimensions of dataset to reduce computations times
    data_x_train = np.pad(data_x_train, ((0,0), (2,2), (2,2)), mode='constant')
    data_x_train = np.expand_dims(data_x_train, channel_index)
    data_x_test = data_x[indicesTest,:,:]
    data_x_test = np.pad(data_x_test, ((0,0), (2,2), (2,2)), mode='constant')
    data_x_test = np.expand_dims(data_x_test, channel_index)
    data_y = mnist['target']
    data_y_train=data_y[indicesTrain] #Reduce dimensions of dataset to reduce computations times
    data_y_test=data_y[indicesTest] #Reduce dimensions of dataset to reduce computations times
    indices = np.arange(len(data_x_train))
    #np.random.shuffle(indices)
    y_batches = build_batches(data_y_train[indices], batch_size)
    x_batches = build_batches(data_x_train[indices], batch_size)
    return x_batches, y_batches, data_x_train, data_y_train, data_x_test, data_y_test

x_batches, y_batches, data_x_train, data_y_train, data_x_test, data_y_test = get_mnist32_batches(batch_size)


#%% Model definition
def Encoder(input_shape):
    #ENCODER
    f=Sequential()
    f.add(Conv2D(16, (3, 3), activation='relu', padding='same',input_shape=(input_shape)))
    f.add(MaxPooling2D((2, 2), padding='same'))
    f.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
    f.add(MaxPooling2D((2, 2), padding='same'))
    f.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
    f.add(MaxPooling2D((2, 2), padding='same'))
    f.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
    f.add(MaxPooling2D((2, 2), padding='same'))
    f.add(Conv2D(1, (3, 3), activation='relu', padding='same'))
    f.add(MaxPooling2D((2, 1), padding='same'))
    #f.summary()
    """
    Complete this part
    """
    
    return f
    
def Decoder(input_shape):
    f=Sequential()
    f.add(UpSampling2D((2, 1),input_shape=(input_shape)))
    f.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
    f.add(UpSampling2D((2, 2)))
    f.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
    f.add(UpSampling2D((2, 2)))
    f.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
    f.add(UpSampling2D((2, 2)))
    f.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
    f.add(UpSampling2D((2, 2)))
    f.add(Conv2D(1, (3, 3), activation='relu', padding='same'))
    f.summary()
    """
    Complete this part
    """
    
    return f
    
        
    
    #%%Create models    
train = True
session = tf.Session()
with session.as_default():
    input_shape = x_batches.shape[2:]
    encoder=Encoder(input_shape)
    input_shape_decoder=encoder.output_shape[1:]
    decoder=Decoder(input_shape_decoder)
    print(decoder.output_shape)
    input_shape=x_batches.shape[2:]
    inputs=Input(input_shape)
    encoded=encoder(inputs)
    decoded=decoder(encoded)
    saver = tf.train.Saver()
    if not train:
        saver.restore(session,"data\\model.ckpt")
    
    model=tf.keras.Model(inputs=inputs,outputs=decoded)
    model.compile('adam', loss=lambda yt,yp: MSE(inputs, decoded))
    loss=[]
    if train:
        for i in range(epochs): 
            batch_idx = random.randint(0,x_batches.shape[0]-1)
            shuffle_idx = np.random.permutation(batch_size)
            history=model.fit(x_batches[batch_idx,shuffle_idx], y_batches[batch_idx,shuffle_idx])
            loss+=[history.history['loss']]
    
    saver.save(session,"data\\model.ckpt")    
    plt.plot(loss)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')

    pred=decoder(encoder(x_batches[0])).eval()
    
    plt.figure()
    for i in range(10):
        plt.subplot(2,5,i+1)
    

        plt.imshow((pred[i,:,:,0]),cmap='gray')
    plt.show()
    pred=encoder(data_x_train).eval()
    test_encoded = encoder(data_x_test).eval()
#%%
data_y_train = np.array( [int (item) for item in data_y_train])
data_y_test = np.array( [int (item) for item in data_y_test])

#%%

plt.figure()
for i in range(10):
    idx_digit = np.argwhere(data_y_train == i)
    pred_digit = pred[idx_digit[:,0]]
    print(pred_digit.shape)
    plt.scatter(pred_digit[:,:,0],pred_digit[:,:,1],label = "digit {}".format(i))        
plt.xlabel("x1")
plt.ylabel("x2")

plt.grid(1)
plt.legend()
plt.show()

#%%
nbrs=KNeighborsClassifier(n_neighbors=1).fit(pred[:,0,:,0],data_y_train)

pred_y_test=nbrs.predict(test_encoded[:,0,:,0])





#%%

percentage=[]
for i in range(10):
    idx_digit = np.argwhere(data_y_test == i)
    good=np.argwhere(pred_y_test[idx_digit[:,0]]==data_y_test[idx_digit[:,0]])
    percentage+=[len(good[:,0])/idx_digit.shape[0]]
    

    
 #%%
 #   
with session.as_default():
    x=np.linspace(0,0.6,15)
    X,Y = np.meshgrid(x,x)
    
    megaplaatje = np.zeros((32*15,32*15))
    
    plt.figure()
    for idx_y,y in enumerate(np.linspace(0,1.6,15)):
        for idx_x, x in enumerate(np.linspace(0,3,15)):
            inp = tf.convert_to_tensor(np.array([[[[x],[y]]]]),dtype = np.float32)
            plaatje = decoder(inp).eval()
            start_y = idx_y * 32
            stop_y = start_y + 32
            start_x = idx_x * 32
            stop_x = start_x + 32
            print("done {} out of {}".format(idx_x + idx_y * 15,15*15))
            
            megaplaatje[start_y:stop_y,start_x:stop_x] = plaatje[0,:,:,0]

plt.imshow(megaplaatje)
plt.show()