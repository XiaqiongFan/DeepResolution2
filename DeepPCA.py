# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 16:02:13 2020

@author: admin
"""


from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow.python.framework import ops
from sklearn import preprocessing
import matplotlib.pyplot as plt 
import os
import csv
from NetCDF import netcdf_reader
import datetime
from scipy.stats import pearsonr
from scipy.linalg import norm
from scipy.integrate import simps
from scipy.optimize import nnls
from numpy import linalg as LA
from scipy.signal import find_peaks,peak_widths
import math
def mkdir(path):
    path = path.strip()
    path = path.rstrip("\\")
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        return True
    else:
        return False

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

def process_DeepPCA(X,model_size): 

    X_b, bias = back_remove(X)

    TS = np.zeros((model_size))    
    [u, s, v]=LA.svd(X_b,full_matrices=False) 

    if s.shape[0]<model_size:   

        TS[0:s.shape[0]] = s
        TS[s.shape[0]:model_size] = np.tile(s[-1], int(model_size-s.shape[0])) 
    else:
        TS = s[0:model_size]

    return TS


def DeepPCA(work_path,data_file,ind_st,ind_en,model_size):
    
    X = data_file['d']
    X_PCA = np.zeros((len(ind_st),model_size,1))
    for k in range(len(ind_st)):
        
        xx = X[ind_st[k]:ind_en[k],:]
        
        xx = process_DeepPCA(xx,model_size)

        X_PCA[k,:,0]=xx
    
    restored_model = tf.keras.models.load_model(work_path+'/model/DeepPCA/model.h5') 
    y_DeepPCA = restored_model.predict_classes(X_PCA)
    y_DeepPCA = y_DeepPCA+np.ones(len(y_DeepPCA))
    y_DeepPCA = y_DeepPCA.astype(int)
    
   
    return y_DeepPCA


    
    
    
    
    
