# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 19:08:22 2019

@author: admin
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt 
import os
import csv
import pandas as pd
from NetCDF import netcdf_reader
from MCR import back_remove,ittfa,fnnls,mcr_by_fr
from sklearn.metrics import explained_variance_score
import datetime
from scipy.stats import pearsonr
from scipy.integrate import simps
from scipy.optimize import nnls
from tensorflow.python.framework import ops  
from DeepSegmentation import DeepSegmentation
from DeepPCA import DeepPCA
from DeepEFA import DeepEFA
from MCR_ALS import mcr_als
from scipy.signal import fftconvolve,find_peaks,find_peaks_cwt

def mkdir(path):
    path = path.strip()
    path = path.rstrip("\\")
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        return True
    else:
        return False

def preprocess(X_raw,channle): 
    X = np.zeros(X_raw.shape)
    for r in range(X.shape[0]):
        X[r,:] = 10000*X_raw[r,:]/np.max(X_raw[r,:])    

    size = int(channle/2)
    Xs =  np.zeros((size,X.shape[1]))
         
    q = int(channle*X.shape[1])
    X_p = np.zeros((X.shape[0],q))
    Xnew = np.vstack((Xs,X,Xs))
    for k in range(X.shape[0]):
        Xk = Xnew[k:k+channle,:].reshape(1,q)
        X_p[k,:] = Xk
    return X_p


def get_regions(y_DeepEFA_M,COM):

    regions_le = []
    regions_ri = []
    for row in range(COM):       
        for col in range(int(y_DeepEFA_M.shape[1])):
            if np.mean(y_DeepEFA_M[row,col:y_DeepEFA_M.shape[1]])>0.95 and y_DeepEFA_M[row,col]>0.95:
                regions_le.append(int(col+1))
                break
    for row_r in range(int(2*COM-1),int(COM-1),-1):       
        for col_r in range(int(y_DeepEFA_M.shape[1]-1),1,-1):
            if np.mean(y_DeepEFA_M[row_r,0:col_r])>0.95 and y_DeepEFA_M[row_r,col_r]>0.95:
                regions_ri.append(col_r+1)
                break
    num=min(len(regions_le),len(regions_ri))
    if len(regions_le) == len(regions_ri):
        pass
    else:
        regions_le = regions_le[0:num]
        regions_ri = regions_ri[0:num]
    
    regions_ri1 = regions_ri
    regions_le1 = regions_le

    for p in range(len(regions_le)):       
        if regions_le[p]>=regions_ri[p]:               
            regions_ri1=np.delete(regions_ri1,0)
            regions_le1=np.delete(regions_le1,-1)
  
    regions = np.zeros((len(regions_le1),2),dtype = int)
    regions[:,0] = regions_le1
    regions[:,1] = regions_ri1

    re=[]        
    for p in range(regions.shape[0]-1):
        for q in range(p+1,regions.shape[0],1):
            
            if regions[q,0]==regions[p,0] and regions[q,1]==regions[p,1]:
                re.append(q)
            if regions[q,0]-regions[p,0]<=5:
                regions[q,0]=regions[q,0]+5
            if regions[q,1]-regions[p,1]<=5:
                regions[p,1]=regions[p,1]-5
            
    regions = np.delete(regions,re,0)   

    return regions

        
def plot_chrom(x,re_chrom,COM,method,path,rt_s,rt_e,R2,y_EFA):
    plt.figure(figsize=(8,9))
    
    rect1 = [0.14, 0.75, 0.77, 0.20]
    rect2 = [0.14, 0.35, 0.77, 0.35]
    rect3 = [0.14, 0.10, 0.77, 0.20]

    plt.axes(rect1)
    plt.plot(x)    
    plt.ylim(0, 1.1*np.max(x))
    plt.xlim(0, y_EFA.shape[1]-1)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('Scans',size=15)
    plt.ylabel('Intensity',size=15)

    plt.axes(rect2)
    plt.imshow(y_EFA)
    plt.xticks([])
    plt.yticks([])
    plt.xlim(0, y_EFA.shape[1]-1)
    plt.xlabel('Scans',size=15)
    
    plt.axes(rect3)

    plt.ylim(0, 1.1*np.max(re_chrom))
    plt.xticks([])
    plt.yticks([])


    color=['b','g','r','k','m']
    for i in range(COM):  
        plt.plot(re_chrom[i,:],linewidth=2,color =str(color[i]))
    plt.xlabel('Scans',size=15)
    plt.ylabel('Intensity',size=15)

    plt.savefig(path+'/RT_from'+str('%.3f'% rt_s)+'to'+str('%.3f'% rt_e)+'_by_'+str(method)+'.jpg')

          
def MCR(x,y_DeepEFA_M,PCA,method,STA,END,RT,result_path): 
    overlapping_num = []
    retention_time = []
    R_R2 = []

    Speak = []
    MCR_method = []
 
    x, bias = back_remove(x)

###################   ITTFA   #####################################################
      
    regions = get_regions(y_DeepEFA_M,PCA)
    if 0<regions.shape[0]<=5:
        COM = regions.shape[0]
        c = np.zeros((x.shape[0], COM))

        if COM == 1:
            ind_s = [np.argmax(np.abs(np.sum(x,1)))]
            if method == 'MCRALS':

                parameter = [['nnls', 'average'], ['nnls']]
                MCRALSModel = mcr_als(x, x[ind_s], parameter)
                Dec = MCRALSModel.run()
    
                c = Dec[1]
                S = Dec[2]

            elif method == 'ITTFA':

                for i in range(COM):  
                    cc = ittfa(x, ind_s[i], COM)
                    c[:, i] = cc[:,0]
                S = np.zeros((COM, x.shape[1]))
                for j in range(0, S.shape[1]):
                    a = fnnls(np.dot(c.T, c), np.dot(c.T, x[:, j]), tole='None')
                    S[:, j] = a['xx']
                    
            else:
                pass

        elif COM == 2:
            ch = np.abs(np.sum(x,1)) 
            
            ind_s1 = np.argmax(ch[regions[0,0]:min(regions[0,1],regions[1,0])])
            ind_s2 = np.argmax(ch[max(regions[0,1],regions[1,0]):regions[1,1]])
            ind_s = [int(ind_s1+regions[0,0]),int(max(regions[0,1],regions[1,0])+ind_s2)]                   
                                
            if method == 'MCRALS':           
                parameter = [['nnls', 'average'], ['nnls']]
                MCRALSModel = mcr_als(x, x[ind_s], parameter)
                Dec = MCRALSModel.run()
    
                c = Dec[1]
                S = Dec[2]
  
            elif method == 'ITTFA':    
                         
                for i in range(COM):  
                    cc = ittfa(x, ind_s[i], COM)
                    c[:, i] = cc[:,0]
                S = np.zeros((COM, x.shape[1]))
                for j in range(0, S.shape[1]):
                    a = fnnls(np.dot(c.T, c), np.dot(c.T, x[:, j]), tole='None')
                    S[:, j] = a['xx']
            else:
                pass
                             
        elif COM == 3:
            ch = np.abs(np.sum(x,1))            

            ind_s1 = np.argmax(ch[regions[0,0]:min(regions[0,1],regions[1,0])]) 
            ind_s3 = np.argmax(ch[max(regions[1,1],regions[2,0]):regions[2,1]])
            if max(regions[0,1],regions[1,0])<min(regions[1,1],regions[2,0]):
                
                p1 = max(regions[0,1],regions[1,0])
                p2 = min(regions[1,1],regions[2,0])
                ind_s2 = np.argmax(ch[p1:p2])  
                ind_s = [int(ind_s1+regions[0,0]),
                         int(ind_s2+min(max(regions[0,1],regions[1,0]),min(regions[1,1],regions[2,0]))),
                         int(ind_s3+max(regions[1,1],regions[2,0]))]
            else:
                
                ind_s2 = int(np.mean([regions[1,0],regions[1,1]]))   
                ind_s = [int(ind_s1+regions[0,0]),ind_s2,
                         int(ind_s3+max(regions[1,1],regions[2,0]))]
          
            if method == 'MCRALS':
       
                parameter = [['nnls', 'average'], ['nnls']]

                MCRALSModel = mcr_als(x, x[ind_s], parameter)
                Dec = MCRALSModel.run()
    
                c = Dec[1]
                S = Dec[2]
                
            elif method == 'ITTFA':
            
                for i in range(COM):  
                    cc = ittfa(x, ind_s[i], COM)
                    c[:, i] = cc[:,0]
                S = np.zeros((COM, x.shape[1]))
                for j in range(0, S.shape[1]):
                    a = fnnls(np.dot(c.T, c), np.dot(c.T, x[:, j]), tole='None')
                    S[:, j] = a['xx']
                    
            else:
                pass
 
        elif COM == 4:
            ch = np.abs(np.sum(x,1))
            ind_s1 = np.argmax(ch[regions[0,0]:min(regions[0,1],regions[1,0])]) 
            p1 = min(max(regions[0,1],regions[1,0]),min(regions[1,1],regions[2,0]))
            p2 = max(max(regions[0,1],regions[1,0]),min(regions[1,1],regions[2,0]))
            if p1==p2:
                p2=p2+1
            ind_s2 = np.argmax(ch[p1:p2])   
            p3 = min(max(regions[1,1],regions[2,0]),min(regions[2,1],regions[3,0]))
            p4 = max(max(regions[1,1],regions[2,0]),min(regions[2,1],regions[3,0]))
            if p3==p4:
                p4=p4+1
            ind_s3 = np.argmax(ch[p3:p4])        
            ind_s4 = np.argmax(ch[max(regions[2,1],regions[3,0]):regions[3,1]])
            ind_s = [int(ind_s1+regions[0,0]),
                     int(ind_s2+min(max(regions[0,1],regions[1,0]),min(regions[1,1],regions[2,0]))),
                     int(ind_s3+min(max(regions[1,1],regions[2,0]),min(regions[2,1],regions[3,0]))),
                     int(ind_s4+max(regions[2,1],regions[3,0]))]

            if method == 'MCRALS':
                parameter = [['nnls', 'average'], ['nnls']]
                MCRALSModel = mcr_als(x, x[ind_s], parameter)
                Dec = MCRALSModel.run()
                c = Dec[1]
                S = Dec[2]
                
            elif method == 'ITTFA':
                for i in range(COM):  
                    cc = ittfa(x, ind_s[i], COM)
                    c[:, i] = cc[:,0]
                S = np.zeros((COM, x.shape[1]))
                for j in range(0, S.shape[1]):
                    a = fnnls(np.dot(c.T, c), np.dot(c.T, x[:, j]), tole='None')
                    S[:, j] = a['xx']      
            else:
                pass

        elif COM == 5:
            ch = np.abs(np.sum(x,1))
            ind_s1 = np.argmax(ch[regions[0,0]:min(regions[0,1],regions[1,0])]) 
            p1 = min(max(regions[0,1],regions[1,0]),min(regions[1,1],regions[2,0]))
            p2 = max(max(regions[0,1],regions[1,0]),min(regions[1,1],regions[2,0]))
            if p1==p2:
                p2=p2+1
            ind_s2 = np.argmax(ch[p1:p2])   
            p3 = min(max(regions[1,1],regions[2,0]),min(regions[2,1],regions[3,0]))
            p4 = max(max(regions[1,1],regions[2,0]),min(regions[2,1],regions[3,0]))
            if p3==p4:
                p4=p4+1
            ind_s3 = np.argmax(ch[p3:p4])    
            p5 = min(max(regions[2,1],regions[3,0]),min(regions[3,1],regions[4,0]))
            p6 = max(max(regions[2,1],regions[3,0]),min(regions[3,1],regions[4,0]))
            if p5==p6:
                p6=p6+1
            ind_s4 = np.argmax(ch[p5:p6])  
            ind_s5 = np.argmax(ch[max(regions[3,1],regions[4,0]):regions[4,1]])
            ind_s = [int(ind_s1+regions[0,0]),
                     int(ind_s2+min(max(regions[0,1],regions[1,0]),min(regions[1,1],regions[2,0]))),
                     int(ind_s3+min(max(regions[1,1],regions[2,0]),min(regions[2,1],regions[3,0]))),
                     int(ind_s4+min(max(regions[2,1],regions[3,0]),min(regions[3,1],regions[4,0]))),
                     int(ind_s5+max(regions[3,1],regions[4,0]))]

            if method == 'MCRALS':
                parameter = [['nnls', 'average'], ['nnls']]
                MCRALSModel = mcr_als(x, x[ind_s], parameter)
                Dec = MCRALSModel.run()
                c = Dec[1]
                S = Dec[2]
            elif method == 'ITTFA':
                for i in range(COM):  
                    cc = ittfa(x, ind_s[i], COM)
                    c[:, i] = cc[:,0]
                S = np.zeros((COM, x.shape[1]))
                for j in range(0, S.shape[1]):
                    a = fnnls(np.dot(c.T, c), np.dot(c.T, x[:, j]), tole='None')
                    S[:, j] = a['xx']   
            else:
                pass
            

        re_chrom = np.zeros((COM, x.shape[0]))
        for k in range(COM): 
            ck = c[:,k].reshape(x.shape[0],1)
            sk = S[k,:].reshape(1,x.shape[1])
            re_chrom[k,:]=np.sum(np.dot(ck,sk),1)
        re_x = np.dot(c,S)
        R2 = explained_variance_score(x, re_x, multioutput='variance_weighted')
            
        for num in range(COM):
            overlapping_num.append(COM) 
            retention_time.append(RT[int(np.argmax(re_chrom[num,:])+STA)])
            R_R2.append(R2)
            Snum = simps(re_chrom[num,:])
            Speak.append(Snum) 
            MCR_method.append(method)
        #plot_chrom(x,re_chrom,COM,method,result_path,RT[STA],RT[END],R2,y_DeepEFA_M) 

####FRR

        if R2<0.999 and COM>=3:
            method = 'HELP_FR'
            xx,re_chrom,R2 = mcr_by_fr(x,regions,COM)

            if R2>=0.9:
                for num in range(COM):
                    overlapping_num.append(COM) 
                    retention_time.append(RT[int(np.argmax(re_chrom[num,:])+STA)])
                    R_R2.append(R2)
    
                    Snum = simps(re_chrom[num,:])
                    Speak.append(Snum) 
                    MCR_method.append(method)

                #plot_chrom(x,re_chrom,COM,method,result_path,RT[STA],RT[END],R2,y_DeepEFA_M) 
            
    else:
        print('Peak_rt:'+str('%.3f'% RT[STA])+'to'+str('%.3f'% RT[END])+': The overlapping peak contains more than five components.')
   
    return{'RT':retention_time,'OVER':overlapping_num,
           'R2':R_R2,'S':Speak,'method':MCR_method}


    
if __name__ == '__main__':
    #Load the data file, model path and components information 
    starttime_t = datetime.datetime.now() 
    COMs = [1,2,3,4,5]
    
    work_path = 'C:/Users/admin/Desktop/DeepResolution2_upload'
    model_size=128
    
    data_file = work_path+'/data/zhi40-5vs1.CDF'

    result_path = work_path+'/result/'+data_file.split('/')[-1]
    mkdir(result_path)
    
    ncr = netcdf_reader(data_file, bmmap=False)
    m = ncr.mat(1,3599, 1)

    ####DeepSegmentation
    tf.keras.backend.clear_session()
    ops.reset_default_graph()
    
    starttime = datetime.datetime.now()  
    y_DeepSeg,ind_st_DeepSeg,ind_en_DeepSeg = DeepSegmentation(work_path,m,model_size,distance=5,threshold=10)  
        
    endtime = datetime.datetime.now()  
    print ('The DeepSegmentation time :',(endtime - starttime),".seconds") 
  
  
    ####DeepPCA
    starttime = datetime.datetime.now()  
    PCA = DeepPCA(work_path,m,ind_st_DeepSeg,ind_en_DeepSeg,model_size)

    endtime = datetime.datetime.now()  
    print ('The DeepPCA time :',(endtime - starttime),".seconds") 
    
    ####DeepEFA
    starttime_EFA = datetime.datetime.now() 
    
    RT_s  = []
    Over_s = []
    R2_s = []
    S_s = []
    Method_s = []
    method = 'MCRALS'
    
    for COMi in range(len(COMs)):
        COM=COMs[COMi]
 
        PC = [idx for idx,i in enumerate(PCA) if i==COM] 
        if len(PC)!=0:
    
            ind_st_DeepSegC = np.array(ind_st_DeepSeg,dtype='int')[PC]
            ind_en_DeepSegC = np.array(ind_en_DeepSeg,dtype='int')[PC]

            
            y_DeepEFA_pre,S_EFA = DeepEFA(work_path,m,ind_st_DeepSegC,ind_en_DeepSegC,model_size,COM)
            

            for M in range(y_DeepEFA_pre.shape[0]):
                XM = m['d'][ind_st_DeepSegC[M]:ind_en_DeepSegC[M],:] 
                
                y_DeepEFA_M = np.zeros((int(2*COM),XM.shape[0]))
                S = np.zeros((int(2*COM),XM.shape[0]))
                for j in range(COM,int(2*COM)):           
                    y_DeepEFA_pre[M,j,:,1] = np.flip(y_DeepEFA_pre[M,j,:,1],0)
                    S_EFA[M,j,:,0] = np.flip(S_EFA[M,j,:,0],0)
    
                y_DeepEFA_M[0:COM,0:XM.shape[0]] = y_DeepEFA_pre[M,0:COM,0:XM.shape[0],1]
                y_DeepEFA_M[COM:int(2*COM),0:XM.shape[0]] = y_DeepEFA_pre[M,COM:int(2*COM),int(model_size-XM.shape[0]):model_size,1]  

                S[0:COM,0:XM.shape[0]] = S_EFA[M,0:COM,0:XM.shape[0],0]
                S[COM:int(2*COM),0:XM.shape[0]] = S_EFA[M,COM:int(2*COM),int(model_size-XM.shape[0]):model_size,0]

                MCR_result = MCR(XM,y_DeepEFA_M,COM,method,ind_st_DeepSegC[M],ind_en_DeepSegC[M],m['rt'],result_path)
                
                for ov in range(len(MCR_result['R2'])):
                    if MCR_result['R2'][ov]>=0:                    
                        RT_s.append(MCR_result['RT'][ov])                        
                        Over_s.append(MCR_result['OVER'][ov])
                        R2_s.append(MCR_result['R2'][ov])
                        S_s.append(MCR_result['S'][ov])
                        Method_s.append(MCR_result['method'][ov])
    
    dataframe = pd.DataFrame({'retention_time':RT_s,'overlapping_num':Over_s,
                              'R2':R2_s,'Speak':S_s,'MCR_method':Method_s})

    dataframe.to_csv(str(result_path)+'/result.csv',index=False,sep=',')
    endtime_EFA = datetime.datetime.now()  
    print ('The DeepEFA time :',(endtime_EFA - starttime_EFA),".seconds") 

    endtime_t = datetime.datetime.now()  
    print ('The total time is',(endtime_t - starttime_t),".seconds")    
      
    

    
    


      