# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 14:39:10 2019

@author: admin
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import datetime
from tensorflow.python.framework import ops
import os

def create_model(x_train):
    return tf.keras.models.Sequential([  
        
        tf.keras.layers.Conv1D(input_shape=(x_train.shape[1],1),filters=32, kernel_size=(2), strides=(1), padding='SAME'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),      
        tf.keras.layers.MaxPool1D(pool_size=(2)),

        tf.keras.layers.Conv1D(input_shape=(x_train.shape[1]/2,32),filters=64, kernel_size=(2), strides=(1), padding='SAME'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPool1D(pool_size=(2)),

        tf.keras.layers.Flatten(),
        
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(5, activation='softmax')
        ])


def randomize(dataset, labels):

    permutation = np.random.permutation(labels.shape[0])
    dataset = dataset[permutation,:]
    labels = labels[permutation,:]
    return dataset, labels

def Hot_map(X):  
    xLabel = []
    yLabel = list(range(1,X.shape[0]+1))
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    ax.set_yticks(range(len(yLabel)))
    ax.set_yticklabels(yLabel)
    ax.set_xticks(range(len(xLabel)))
    ax.set_xticklabels(xLabel)
    ax.set_xlabel('Retention time')
    ax.set_ylabel('Compound')
    im = ax.imshow(X,interpolation='none', cmap=plt.cm.hot_r)
    position=fig.add_axes([0.92, 0.37, 0.02, 0.25])
    plt.colorbar(im,cax=position)
    plt.show()


def Load_data(data_path):
       
    datafile1 = data_path+'/X.npy'
    X_raw = np.load(datafile1)

   
    for i in range(X_raw.shape[0]):
        SC = np.random.randint(40,60)
        X_raw[i,SC:]=np.tile(X_raw[i,SC],(128-SC))
        
          
    datafile2 = data_path+'/Y.npy'
    Y_raw = np.load(datafile2) 
    
  

    X,Y = randomize(X_raw,Y_raw)
   
    X = X.reshape(X.shape[0],X.shape[1],1)
    Xtrain = X[0:int(0.9*X.shape[0])]
    Xtest = X[int(0.9*X.shape[0]):X.shape[0]]

    Ytrain = Y[0:int(0.9*X.shape[0])]
    Ytest = Y[int(0.9*X.shape[0]):X.shape[0]]
    return Xtrain,Xtest,Ytrain,Ytest
def mkdir(path):
    path = path.strip()
    path = path.rstrip("\\")
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        return True
    else:
        return False

if __name__ == '__main__':
    data_path = u'F:/massbank/PCA0107'
    
    savepath = 'C:/Users/admin/Desktop/DeepResolution2/model/kCNN'
    mkdir(savepath)
    
    Xtrain,Xtest,Ytrain,Ytest = Load_data(data_path)

    starttime = datetime.datetime.now()   

    tf.keras.backend.clear_session()
    ops.reset_default_graph()
        
    model = create_model(Xtrain)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                 loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=['accuracy'])
    model.summary()


    history = model.fit(Xtrain, Ytrain, batch_size=500, epochs=200, validation_split=0.1, verbose=1)
   
         
    fig = plt.figure(figsize = (6,5))
    ax = fig.add_subplot(111)       
    ax.set_ylabel('Accuracy',size=15)
    ax.set_xlabel('Epoch',size=15)       
    lns1 = plt.plot(history.history['accuracy'],label = 'Acc_training',color='r')
    lns2 = plt.plot(history.history['val_accuracy'],label = 'Acc_validation',color='g')
    ax2 = ax.twinx()  
    ax2.set_ylabel('Loss',size=15)
    lns3 = plt.plot(history.history['loss'],label = 'Loss_training',color='b')
    lns4 = plt.plot(history.history['val_loss'],label = 'Loss_validation',color='orange')
    lns = lns1+lns2+lns3+lns4
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc=7)
    plt.show()
    
    
    model.evaluate(Xtest, Ytest, verbose=2)

    model.save(savepath+'/model.h5')

    del model

    endtime = datetime.datetime.now()  
    print ('The time :',(endtime - starttime),".seconds")
    print('Trained Model Saved.')

