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
    Ypred = myplsda.predict(X)
  
    Ypred[Ypred>=0.5]=1
    Ypred[Ypred<0.5]=0

    scores = myplsda.x_scores_
    myplsda.x_scores_
    myplsda.x_loadings_
    myplsda.x_weights_
    scores = myplsda.x_scores_

    A=[]
    B=[]
    C=[]
    D=[]
    for i in range(X.shape[0]):
        if Ypred[i][0] == 0 and Y[i][0] ==0:
            A.append(i)
        elif Ypred[i][0] == 1 and Y[i][0] ==1:
            B.append(i)
        elif Ypred[i][0]  == 1 and Y[i][0] ==0:
            C.append(i)
        else:
            D.append(i)
              
    x0 = scores[A]
    x1 = scores[B]
    x2 = scores[C]
    x3 = scores[D]

    fig = plt.figure(figsize=(6,6))

    #ax = fig.gca(projection='3d')
    ax = fig.add_subplot(111, projection='3d')
    plt.gca().patch.set_facecolor('white')
    ax.w_xaxis.set_pane_color((0, 0, 0, 0))
    ax.w_yaxis.set_pane_color((0, 0, 0, 0))
    ax.w_zaxis.set_pane_color((0, 0, 0, 0))

    ax.view_init(elev=30)#azim=-50
    ax.set_yticks(range(-6,7,2))
    ax.set_yticklabels([])
    ax.set_xticks(range(-6,7,2))
    ax.set_xticklabels([])
    ax.set_zticks(range(-4,5,2))
    ax.set_zticklabels([])
    ax.set_ylim(-6,6)
    ax.set_xlim(-6,6)
    ax.set_zlim(-4,4)


    ax.scatter(x0[:, 0], x0[:, 1], x0[:, 2], c = "r", marker='o', label='TN')    
    ax.scatter(x1[:, 0], x1[:, 1], x1[:, 2], c = "b", marker='^', label='TP')    
    ax.scatter(x2[:, 0], x2[:, 1], x2[:, 2], c = "k", marker='+', label='FP')
    ax.scatter(x3[:, 0], x3[:, 1], x3[:, 2], c = "g", marker='*', label='FN')  

    ax.set_xlabel('PLS1',size=15)
    ax.set_ylabel('PLS2',size=15)
    ax.set_zlabel('PLS3',size=15)
    plt.legend(loc = 2,prop={'size':12})
    plt.show()

    print('TN=',x0.shape[0])
    print('TP=',x1.shape[0])
    print('FP=',x2.shape[0])
    print('FN=',x3.shape[0])
    print('sensitivity=',x1.shape[0]/(x1.shape[0]+x3.shape[0]))
    print('specificity=',x0.shape[0]/(x0.shape[0]+x2.shape[0]))
    print('acc=',accuracy(Ypred,Y)/100)
    
    

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
