# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 15:50:24 2020

@author: admin
"""
import tensorflow as tf
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt 
import os
import csv
from NetCDF import netcdf_reader

from sklearn.metrics import explained_variance_score
import datetime
from scipy.stats import pearsonr
from scipy.integrate import simps
from scipy.optimize import nnls
import tensorflow as tf
from tensorflow.python.framework import ops
from numpy import linalg as LA


def back_remove(xx):
    Ns = [[0,3],[xx.shape[0]-3,xx.shape[0]]]

    bak = np.zeros(xx.shape)
    for i in range(0, xx.shape[1]):
        tiab = []
        reg = []
        for j in range(0, len(Ns)):
            tt = range(Ns[j][0],Ns[j][1])
            tiab.extend(xx[tt, i])
            reg.extend(np.arange(Ns[j][0], Ns[j][1]))
        rm = reg - np.mean(reg)
        tm = tiab - np.mean(tiab)
        b = np.dot(np.dot(float(1)/np.dot(rm.T, rm), rm.T), tm)
        s = np.mean(tiab)-np.dot(np.mean(reg), b)
        b_est = s+b*np.arange(xx.shape[0])
        bak[:, i] = xx[:, i]-b_est   
    bias = xx-bak
    return bak, bias


def process_UNet4R(XX,COM): 
    X_bk, bias = back_remove(XX)

    S = np.zeros((2*COM,XX.shape[0]))

    for j in range(1,XX.shape[0]+1):   
        
        XJ = X_bk[0:j,:]  
        [u, s, v]=LA.svd(XJ,full_matrices=False) 
        if len(s)<COM:
            S[0:len(s),j-1] = s
        else:
            S[0:COM,j-1] = s[0:COM]
         
    for k in range(XX.shape[0]-1,-1,-1):                
        XIK = X_bk[k:XX.shape[0],:]  
        [u, s, v]=LA.svd(XIK,full_matrices=False) 

        if len(s)<COM:
            S[COM:int(COM+len(s)),k] = s
        else:
            S[COM:int(2*COM),k] = s[0:COM]
    S=S/np.max(S)
    return S


    
def UNet4R(work_path,m,ind_st,ind_en,model_size,COM):
    
    Xdata = m['d']  
    Xtest = np.zeros((len(ind_st),int(2*COM),model_size,1))
    for p in range(len(ind_st)):
        
        Xp = Xdata[ind_st[p]:ind_en[p],:] 
        Sp = process_UNet4R(Xp,COM)
       
        S = np.zeros(Sp.shape)
        S[0:COM,:] = Sp[0:COM,:]
        S[COM:int(2*COM),:] = Sp[COM:int(2*COM),:]

    
        for j in range(COM,int(2*COM)):           
            S[j,:] = np.flip(S[j,:],0)



        Xtest[p,:,0:S.shape[1],0] = S
        Xtest[p,:,S.shape[1]:model_size,0] = np.tile(S[:,S.shape[1]-1:S.shape[1]],(1,model_size-S.shape[1]))
        

    
    tf.keras.backend.clear_session()
    ops.reset_default_graph()
    restored_model = tf.keras.models.load_model(work_path+'/model/UNet4R/'+str(COM)+'/model.h5')    
    y_UNet4R = restored_model.predict(Xtest)

    return y_UNet4R,Xtest
        
