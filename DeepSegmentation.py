# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 14:56:01 2020

@author: admin
"""

import numpy as np
import tensorflow as tf
import datetime
import matplotlib.pyplot as plt 
import math

def process_DeepSegmentation(chrom,RT,model_size):    
    num = chrom.shape[0]//model_size
    if chrom.shape[0]%model_size!=0:
        segs = np.zeros((int(num+1),chrom.shape[1],model_size))
        tics = np.zeros((int(num+1),model_size))
        RTs =  np.zeros((int(num+1),2))
        for i in range(num):
            segs[i,:,:]=chrom[int(i*model_size):int((i+1)*model_size),:].T
            tics[i,:]=np.sum(segs[i,:,:],0)
            RTs[i,0] = RT[int(i*model_size)]
            RTs[i,1] = RT[int((i+1)*model_size-1)]
        segs[num,:]=chrom[int(chrom.shape[0]-model_size):chrom.shape[0],:].T
        tics[num,:]=np.sum(segs[num,:],0)
        RTs[num,0] = RT[int(chrom.shape[0]-model_size)]
        RTs[num,1] = RT[int(chrom.shape[0]-1)]
    
    else:
        segs = np.zeros((num,model_size))
        RTs =  np.zeros((num,2))
        for i in range(num):
            segs[i,:,:]=chrom[i*model_size:(i+1)*model_size,:].T
            tics[i,:]=np.sum(segs[i,:,:],0)
            RTs[i,0] = RT[int(i*model_size)]
            RTs[i,1] = RT[int((i+1)*model_size-1)]


    for j in range(segs.shape[0]):
        for k in range(segs.shape[1]):
            segsI = segs[j,k,:]       
            if np.max(segsI)!=0:                
                segs[j,k,:]= segsI/np.max(segsI)        
    segs_unfold = np.zeros((segs.shape[0]*segs.shape[1],segs.shape[2]))
    for s in range(segs.shape[0]):
        segs_unfold[int(s*segs.shape[1]):int((s+1)*segs.shape[1]),:]=segs[s,:,:]
    segs_unfold = segs_unfold.reshape(segs_unfold.shape[0],1,segs_unfold.shape[1],1)
   
    return segs,segs_unfold,tics,RTs

def find_min(y_tic,A,B,model_size):
    while B-A>model_size:
        seg_range = list(y_tic[A:B])
        minind = seg_range.index(min(seg_range)) + A
    return minind
    
    
def Segmentation(y_DeepSeg,model_size,distance,threshold):
    
    y_tic = np.sum(y_DeepSeg,1)

    peak_st_ind=[]
    peak_en_ind=[]
    
    for scan in range(y_DeepSeg.shape[0]-1):
        if y_tic[scan]<threshold and y_tic[scan+1]>=threshold:
            peak_st_ind.append(scan)
        if y_tic[scan]>=threshold and y_tic[scan+1]<threshold:
            peak_en_ind.append(scan)

##半峰            
    if peak_st_ind[0]<peak_en_ind[0]:
        pass
    else:
        peak_en_ind = np.delete(peak_en_ind,0)

    if peak_st_ind[-1]<peak_en_ind[-1]:
        pass
    else:
        peak_st_ind = np.delete(peak_st_ind,-1)

##假峰    
    peak_st_ind1=[]
    peak_en_ind1=[]   
    for scan in range(len(peak_st_ind)):
        if peak_en_ind[scan] - peak_st_ind[scan]<=distance:
            pass
        else:
            peak_st_ind1.append(peak_st_ind[scan])
            peak_en_ind1.append(peak_en_ind[scan])

##连续峰  

    peak_st_ind=[]
    peak_en_ind=[]    
    for scan in range(1,len(peak_st_ind1)-1):
# =============================================================================
#         print(scan,peak_st_ind1[scan],peak_en_ind1[scan])
# =============================================================================
        
        if peak_st_ind1[int(scan+1)]-peak_en_ind1[scan]>distance and peak_st_ind1[scan]-peak_en_ind1[int(scan-1)]>distance:
            peak_st_ind.append(peak_st_ind1[scan])
            peak_en_ind.append(peak_en_ind1[scan])
                    
        elif peak_st_ind1[scan]-peak_en_ind1[int(scan-1)]<=distance and peak_st_ind1[int(scan+1)]-peak_en_ind1[scan]>distance:
            le=0
            while peak_st_ind1[int(scan-le)]-peak_en_ind1[int(scan-1-le)]<=distance and scan-1-le>=0:
                le=le+1
            peak_st_ind.append(peak_st_ind1[int(scan-le)])
            peak_en_ind.append(peak_en_ind1[scan])
            
        elif peak_st_ind1[scan]-peak_en_ind1[int(scan-1)]>distance and peak_st_ind1[int(scan+1)]-peak_en_ind1[scan]<=distance:
            ri=0
            while peak_st_ind1[int(scan+ri+1)]-peak_en_ind1[int(scan+ri)]<=distance and scan+ri+1<len(peak_st_ind1)-1:
                ri=ri+1
            peak_en_ind.append(peak_en_ind1[scan+ri])  
            peak_st_ind.append(peak_st_ind1[scan])
            
        else:
            le=0
            while peak_st_ind1[int(scan-le)]-peak_en_ind1[int(scan-1-le)]<=distance and scan-1-le>=0:
                le=le+1
            peak_st_ind.append(peak_st_ind1[int(scan-le)])
            ri=0
            while peak_st_ind1[int(scan+ri+1)]-peak_en_ind1[int(scan+le)]<=distance and scan+ri+1<len(peak_st_ind1)-1:
                ri=ri+1
            peak_en_ind.append(peak_en_ind1[scan+ri])  
    
     
    if peak_st_ind1[0] not in peak_st_ind:
        peak_st_ind.insert(0,peak_st_ind1[0])
        peak_en_ind.insert(0,peak_en_ind1[0])
    if peak_en_ind1[-1] not in peak_en_ind:
        peak_st_ind.append(peak_st_ind1[-1])
        peak_en_ind.append(peak_en_ind1[-1])
        

    peak_st_ind_sifting=[]
    peak_en_ind_sifting=[]
    for scan in range(len(peak_st_ind)):
        if peak_st_ind[scan] not in peak_st_ind_sifting:
            peak_st_ind_sifting.append(peak_st_ind[scan])
            peak_en_ind_sifting.append(peak_en_ind[scan])
        else:
            pass


##适应model_size    

    re_peak_st_ind=[]
    re_peak_en_ind=[]
    
    for s in range(len(peak_st_ind_sifting)):
        if peak_en_ind_sifting[s] - peak_st_ind_sifting[s]<=model_size-2*distance:
            re_peak_st_ind.append(max(0,peak_st_ind_sifting[s]-distance))
            re_peak_en_ind.append(min(peak_en_ind_sifting[s] + distance, y_DeepSeg.shape[0]))
        else:           
            sta = int(peak_st_ind_sifting[s] + distance)
            end = int(peak_en_ind_sifting[s] - distance)
        
            seg_range = list(y_tic[sta:end])
            minind = seg_range.index(min(seg_range)) + sta
            
            minindL = minind
            while minindL - sta>model_size - 2*distance:       
                seg_range = list(y_tic[sta:minindL - distance])
                minindL = seg_range.index(min(seg_range)) + sta

            minindR = minind
            while end - minindR>model_size - 2*distance:
                seg_range = list(y_tic[minindR + distance:end])
                minindR = seg_range.index(min(seg_range)) + minindR + distance

            re_peak_st_ind.append(max(0,sta - 2*distance))
            re_peak_en_ind.append(minindL)
            re_peak_st_ind.append(minindR)
            re_peak_en_ind.append(min(end + 2*distance, y_DeepSeg.shape[0]))

            sta = minindL + distance
            end = minindR - distance

            while end - sta >model_size - 2*distance:
                
                seg_range = list(y_tic[sta:end])
                minind = seg_range.index(min(seg_range)) + sta
                
                minindL = minind

                while minindL - sta>model_size - distance: 
                 
                    seg_range = list(y_tic[sta:minindL - distance])
                    minindL = seg_range.index(min(seg_range)) + sta

                minindR = minind
                while end - minindR>model_size - distance:
                    seg_range = list(y_tic[minindR + distance:end])
                    minindR = seg_range.index(min(seg_range)) + minindR + distance

                re_peak_st_ind.append(sta - distance)
                re_peak_en_ind.append(minindL)
                re_peak_st_ind.append(minindR)
                re_peak_en_ind.append(end + distance)
                
                sta = minindL + distance
                end = minindR - distance

            if minindL!=minindR:
                re_peak_st_ind.append(sta - distance)
                re_peak_en_ind.append(end + distance)     

    return re_peak_st_ind, re_peak_en_ind
                
    
def DeepSegmentation(work_path,data_file,model_size,distance,threshold):
    X = data_file['d']
    RT = data_file['rt']
    segs,segs_unfold,tics,RTs = process_DeepSegmentation(X,RT,model_size)   
    restored_model = tf.keras.models.load_model(work_path+'/model/DeepSegmentation/model.h5') 
    y_DeepSegPred  = restored_model.predict(segs_unfold)


    y_fold = np.zeros((segs.shape[0],segs.shape[1],segs.shape[2]))
    y_DeepSeg = np.zeros(X.shape)
    for i in range(segs.shape[0]):
        y_fold[i,:,:]=y_DeepSegPred[int(i*segs.shape[1]):int((i+1)*segs.shape[1]),0,:,0]
        if i<segs.shape[0]-1:               
            y_DeepSeg[i*model_size:(i+1)*model_size,:]=y_fold[i,:,:].T        
    y_DeepSeg[(segs.shape[0]-1)*model_size:X.shape[0],:]=y_fold[-1,:,segs.shape[0]*model_size-X.shape[0]:model_size].T

    y_DeepSeg[y_DeepSeg>=0.9]=1
    y_DeepSeg[y_DeepSeg<0.9]=0   

    ind_st,ind_en = Segmentation(y_DeepSeg,model_size,distance,threshold)


    return y_DeepSeg,ind_st,ind_en
