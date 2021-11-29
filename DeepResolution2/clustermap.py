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
    from mpl_toolkits.mplot3d import Axes3D
    import seaborn as sns  
    from matplotlib.pyplot import MultipleLocator

    CSV_file = csv.reader(open('PlasmaResult.csv', encoding='gbk'))
    X = [row for row in CSV_file]


        
    DATA = pd.read_csv("./PlasmaResult.csv", encoding='gbk')

    X = np.array(DATA.values[:,1:].T,dtype=float)
    COM = DATA.values[:,0]
    Y = np.zeros((136,2))
    Y[0:61,1]=1
    Y[61:136,0]=1

    #X,Y = randomize(X,Y)

    myplsda=PLSRegression(3).fit(X=X,Y=Y)

    VIPs = vip(X, Y, myplsda)   
    COM_vip = []
    VIP = []

    for vv in range(53):
        if VIPs[vv]>=1:
            
            VIP.append(vv)
            COM_vip.append(COM[vv])
    Xvip = X[:,VIP]
        
    sns.set(font_scale=1.4)
    df = pd.DataFrame(Xvip,index=None,columns = COM_vip)
    g = sns.clustermap(df.T,cmap="coolwarm",col_cluster=False,row_cluster=True,annot_kws={"size": 15},
                       cbar_kws=dict(orientation='horizontal'),figsize=(12,5),xticklabels=False)

    x0, _y0, _w, _h = g.cbar_pos
    g.ax_cbar.set_position([0.178, 0.88, 0.64, 0.02])
    g.ax_cbar.tick_params(axis='x', length=10)
    for spine in g.ax_cbar.spines:
        g.ax_cbar.spines[spine].set_color('crimson')
        g.ax_cbar.spines[spine].set_linewidth(2)

    plt.show()
