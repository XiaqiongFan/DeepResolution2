# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 20:18:22 2019

@author: admin
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import datetime

from tensorflow.python.framework import ops
import os

def unet(X):
    inputs = tf.keras.layers.Input((X.shape[1],X.shape[2],X.shape[3]))
    conv1 = tf.keras.layers.Conv2D(64, (1,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = tf.keras.layers.Conv2D(64, (1,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)    
    pool1 = tf.keras.layers.MaxPooling2D(pool_size=(1, 2))(conv1)
   
    conv2 = tf.keras.layers.Conv2D(128, (1,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = tf.keras.layers.Conv2D(128, (1,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = tf.keras.layers.MaxPooling2D(pool_size=(1, 2))(conv2)

    conv3 = tf.keras.layers.Conv2D(256, (1,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = tf.keras.layers.Conv2D(256, (1,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    drop3 = tf.keras.layers.Dropout(0.5)(conv3)
    pool3 = tf.keras.layers.MaxPooling2D(pool_size=(1, 2))(drop3)

    conv4 = tf.keras.layers.Conv2D(512, (1,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = tf.keras.layers.Conv2D(512, (1,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = tf.keras.layers.Dropout(0.5)(conv4)
    
    up5 = tf.keras.layers.Conv2D(256, (1,2), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(tf.keras.layers.UpSampling2D(size = (1,2))(drop4))
    merge5 = tf.keras.layers.concatenate([conv3,up5], axis = 3)
    conv5 = tf.keras.layers.Conv2D(256, (1,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge5)
    conv5 = tf.keras.layers.Conv2D(256, (1,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)

    up6 = tf.keras.layers.Conv2D(128, (1,2), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(tf.keras.layers.UpSampling2D(size = (1,2))(conv5))
    merge6 = tf.keras.layers.concatenate([conv2,up6], axis = 3)
    conv6 = tf.keras.layers.Conv2D(128, (1,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = tf.keras.layers.Conv2D(128, (1,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = tf.keras.layers.Conv2D(64, (1,2), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(tf.keras.layers.UpSampling2D(size = (1,2))(conv6))
    merge7 = tf.keras.layers.concatenate([conv1,up7], axis = 3)
    conv7 = tf.keras.layers.Conv2D(64, (1,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = tf.keras.layers.Conv2D(64, (1,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
    conv7 = tf.keras.layers.Conv2D(2, (1,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    conv8 = tf.keras.layers.Conv2D(2, (1,1), activation = 'sigmoid')(conv7)
    model = tf.keras.models.Model(inputs, conv8)
    
    return model

def Load_data(data_path,COM):
    
    datafile1 = data_path+'/'+str(COM)+'Xdata.npy'
    X_raw = np.load(datafile1)
    
    datafile2 = data_path+'/'+str(COM)+'Ydata.npy'
    Y_raw = np.load(datafile2) 
       
    for i in range(X_raw.shape[0]):
        for j in range(COM,X_raw.shape[1]):           
            X_raw[i,j,:,:] = np.flip(X_raw[i,j,:,:],0)
            Y_raw[i,j,:,:] = np.flip(Y_raw[i,j,:,:],0)
            
                          
    for i in range(X_raw.shape[0]):
        X_raw[i,:,:,:]=X_raw[i,:,:,:]/np.max(X_raw[i,:,:,:])

    Y2 = tf.keras.utils.to_categorical(Y_raw)
    X,Y = randomize(X_raw, Y2)
    
    Xtrain = X[0:int(0.9*X.shape[0]),:,:,:]
    Xtest = X[int(0.9*X.shape[0]):X.shape[0],:,:,:]

    Ytrain = Y[0:int(0.9*X.shape[0]),:,:,:]
    Ytest = Y[int(0.9*X.shape[0]):X.shape[0],:,:,:]
    return Xtrain,Xtest,Ytrain,Ytest
    

def randomize(dataset, labels):

    permutation = np.random.permutation(labels.shape[0])
    dataset = dataset[permutation,:,:,:]
    labels = labels[permutation,:,:,:]
    return dataset, labels

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

    data_path = u'F:/massbank/EFAefa'

    COMS = [1,2,3,4,5]
    
    LR = [7e-3,7e-3,7e-3,6e-3,7e-3]

    
    for COM in range(len(COMS)):
        COM=COMS[COM]
            
        savepath = 'C:/Users/admin/Desktop/DeepResolution2/model/UNet4R/'+str(COM)
        mkdir(savepath)
        
        tf.keras.backend.clear_session()
        ops.reset_default_graph()
        
        Xtrain,Xtest,Ytrain,Ytest = Load_data(data_path,COM)

        starttime = datetime.datetime.now()  
        
        model = unet(Xtrain)   

        model.compile(optimizer = tf.keras.optimizers.SGD(lr=LR[COM], momentum=0.99), loss = 'binary_crossentropy', metrics = ['accuracy'])

        model.summary()

        history = model.fit(Xtrain, Ytrain, batch_size=500, epochs=50, validation_split=0.1, verbose=1)

        
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
   
        model.save(savepath+'/model.h5')
  
        del Xtrain,Ytrain
    
        endtime = datetime.datetime.now()  
        print ('The time :',(endtime - starttime),".seconds")
        print('Trained Model Saved.')      
        


    

