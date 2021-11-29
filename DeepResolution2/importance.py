# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 23:24:09 2021

@author: admin
"""

import tensorflow as tf
import numpy as np
import csv
import os
import bisect
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score


def randomize(dataset, labels):
    
    permutation = np.random.permutation(labels.shape[0])
    dataset = dataset[permutation,:]
    labels = labels[permutation]
    return dataset, labels

def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
              / predictions.shape[0])

def vip(x, y, model):
    t = model.x_scores_
    w = model.x_weights_
    q = model.y_loadings_

    m, p = x.shape
    _, h = t.shape

    vips = np.zeros((p,))

    s = np.diag(t.T @ t @ q.T @ q).reshape(h, -1)
    total_s = np.sum(s)

    for i in range(p):
        weight = np.array([ (w[i,j] / np.linalg.norm(w[:,j]))**2 for j in range(h) ])
        vips[i] = np.sqrt(p*(s.T @ weight)/total_s)

    return vips



    
if __name__ == '__main__':
    import matplotlib.pyplot as plt    
    import seaborn as sns  

        
    DATA = pd.read_csv("./PlasmaResult.csv", encoding='gbk')

    X = np.array(DATA.values[:,1:].T,dtype=float)
    COMs = DATA.values[:,0]
    Y = np.zeros((136,2))
    Y[0:61,1]=1
    Y[61:136,0]=1


    myplsda=PLSRegression(3).fit(X=X,Y=Y)

    VIPs = vip(X, Y, myplsda)   

    COM = []
    VIP = []

    for vv in range(53):
        if VIPs[vv]>=1:
            
            VIP.append(VIPs[vv])
            COM.append(COMs[vv])
     

    sorted_vips = sorted(enumerate(VIP), key=lambda x: x[1])
    idx = [i[0] for i in sorted_vips]
    vips = [i[1] for i in sorted_vips]
    COMS=[]
    for jj in range(len(COM)):
        COMS.append(COM[idx[jj]])

    plt.figure(figsize=(6,6))
    plt.scatter(vips,range(len(COMS)),c = "g")
    plt.yticks(range(len(COMS)),COMS,size=16)
    plt.xticks(size=15)

    plt.grid(axis="y",linestyle='-.')