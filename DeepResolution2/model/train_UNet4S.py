# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 20:18:22 2019

@author: admin
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import datetime
import os


def unet(X):
    inputs = tf.keras.layers.Input((X.shape[1],X.shape[2],X.shape[3]))
    conv1 = tf.keras.layers.BatchNormalization()(inputs)
    conv1 = tf.keras.layers.Conv2D(64, (1,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    conv1 = tf.keras.layers.BatchNormalization()(conv1)
    conv1 = tf.keras.layers.Conv2D(64, (1,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)    
    conv1 = tf.keras.layers.BatchNormalization()(conv1)
    pool1 = tf.keras.layers.MaxPooling2D(pool_size=(1, 2))(conv1)

    
    conv2 = tf.keras.layers.Conv2D(128, (1,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = tf.keras.layers.BatchNormalization()(conv2)
    conv2 = tf.keras.layers.Conv2D(128, (1,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    conv2 = tf.keras.layers.BatchNormalization()(conv2)
    pool2 = tf.keras.layers.MaxPooling2D(pool_size=(1, 2))(conv2)

    conv3 = tf.keras.layers.Conv2D(256, (1,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = tf.keras.layers.BatchNormalization()(conv3)
    conv3 = tf.keras.layers.Conv2D(256, (1,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    conv3 = tf.keras.layers.BatchNormalization()(conv3)
    pool3 = tf.keras.layers.MaxPooling2D(pool_size=(1, 2))(conv3)

    conv4 = tf.keras.layers.Conv2D(512, (1,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = tf.keras.layers.BatchNormalization()(conv4)
    conv4 = tf.keras.layers.Conv2D(512, (1,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    conv4 = tf.keras.layers.BatchNormalization()(conv4)
    pool4 = tf.keras.layers.MaxPooling2D(pool_size=(1, 2))(conv4)

    conv5 = tf.keras.layers.Conv2D(1024, (1,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = tf.keras.layers.BatchNormalization()(conv5)
    conv5 = tf.keras.layers.Conv2D(1024, (1,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    conv5 = tf.keras.layers.BatchNormalization()(conv5)
    
    up6 = tf.keras.layers.Conv2D(512, (1,2), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(tf.keras.layers.UpSampling2D(size = (1,2))(conv5))
    up6 = tf.keras.layers.BatchNormalization()(up6)
    merge6 = tf.keras.layers.concatenate([conv4,up6], axis = 3)
    conv6 = tf.keras.layers.Conv2D(512, (1,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = tf.keras.layers.BatchNormalization()(conv6)
    conv6 = tf.keras.layers.Conv2D(512, (1,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
    conv6 = tf.keras.layers.BatchNormalization()(conv6)

    up7 = tf.keras.layers.Conv2D(256, (1,2), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(tf.keras.layers.UpSampling2D(size = (1,2))(conv6))
    up7 = tf.keras.layers.BatchNormalization()(up7)
    merge7 = tf.keras.layers.concatenate([conv3,up7], axis = 3)
    conv7 = tf.keras.layers.Conv2D(256, (1,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = tf.keras.layers.BatchNormalization()(conv7)
    conv7 = tf.keras.layers.Conv2D(256, (1,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
    conv7 = tf.keras.layers.BatchNormalization()(conv7)

    up8 = tf.keras.layers.Conv2D(128, (1,2), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(tf.keras.layers.UpSampling2D(size = (1,2))(conv7))
    up8 = tf.keras.layers.BatchNormalization()(up8)
    merge8 = tf.keras.layers.concatenate([conv2,up8], axis = 3)
    conv8 = tf.keras.layers.Conv2D(128, (1,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = tf.keras.layers.BatchNormalization()(conv8)
    conv8 = tf.keras.layers.Conv2D(128, (1,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
    conv8 = tf.keras.layers.BatchNormalization()(conv8)

    up9 = tf.keras.layers.Conv2D(64, (1,2), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(tf.keras.layers.UpSampling2D(size = (1,2))(conv8))
    up9 = tf.keras.layers.BatchNormalization()(up9)
    merge9 = tf.keras.layers.concatenate([conv1,up9], axis = 3)
    conv9 = tf.keras.layers.Conv2D(64, (1,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = tf.keras.layers.BatchNormalization()(conv9)
    conv9 = tf.keras.layers.Conv2D(64, (1,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = tf.keras.layers.BatchNormalization()(conv9)
    conv9 = tf.keras.layers.Conv2D(2, (1,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)

    conv10 = tf.keras.layers.Conv2D(2, (1,1), activation = 'sigmoid')(conv9)
    model = tf.keras.models.Model(inputs, conv10)
    
    return model

def Load_data(data_path):
    
    datafile1 = data_path+'/DeepSegmentation/X.npy'
    X_raw = np.load(datafile1)
    plt.figure()
    plt.plot(X_raw[0,:,:,0].T)
    
    datafile2 = data_path+'/DeepSegmentation/Y.npy'
    Y_raw = np.load(datafile2) 
    plt.figure()
    plt.imshow(Y_raw[0,:,:,0])

    X,Y = randomize(X_raw, Y_raw)

    Xtrain = X[0:int(0.9*X.shape[0])]
    Xtest = X[int(0.9*X.shape[0]):X.shape[0]]

    Ytrain = Y[0:int(0.9*X.shape[0])]
    Ytest = Y[int(0.9*X.shape[0]):X.shape[0]]
    return Xtrain,Xtest,Ytrain,Ytest

def randomize(dataset, labels):

    permutation = np.random.permutation(labels.shape[0])
    dataset = dataset[permutation,:]
    labels = labels[permutation,:]
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
    from tensorflow.python.framework import ops    
    tf.keras.backend.clear_session()
    ops.reset_default_graph()
    data_path = u'F:/massbank'
    Xtrain,Xtest,Ytrain,Ytest = Load_data(data_path)

    savepath = 'C:/Users/admin/Desktop/DeepResolution2/model/UNet4S'
    mkdir(savepath)
    starttime = datetime.datetime.now()  
    
    model = unet(Xtrain)    

    model.compile(optimizer = tf.keras.optimizers.Adam(), loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    model.summary()
    
    history = model.fit(Xtrain, Ytrain, batch_size=256, epochs=30, validation_split=0.1, verbose=1)
           
    fig = plt.figure()
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


