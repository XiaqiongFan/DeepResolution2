# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 19:08:22 2019

@author: admin
"""

import numpy as np
from matplotlib.pyplot import show, plot, text
from NetCDF import netcdf_reader,plot_ms,plot_tic
import matplotlib.pyplot as plt
from sklearn.metrics import explained_variance_score
from scipy.linalg import norm

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


def FR(x, s, o, z, com):
    xs = x[s,:]   
    xs[xs<0]=0
    xz = x[z,:]
    xo = x[o,:]
    xc = np.vstack((xs, xz))
    mc = np.vstack((xs, np.zeros(xz.shape)))

    u, s0, v = np.linalg.svd(xc)
    t = np.dot(u[:,0:com],np.diag(s0[0:com]))
    r = np.dot(np.dot(np.linalg.pinv(np.dot(t.T, t)), t.T), np.sum(mc, 1))
    u1, s1, v1 = np.linalg.svd(x)
    t1 = np.dot(u1[:, 0:com], np.diag(s1[0:com]))
    c = np.dot(t1, r)

    c1, ind = contrain_FR(c, s, o)
    c1[c1<0]=0
    spec = x[s[ind],:]

    if c1[s[ind]] == 0:
        pu = 1e-6
    else:
        pu = c1[s[ind]]
        
    cc = c1/pu

    res_x = np.dot(np.array(cc, ndmin=2).T, np.array(spec, ndmin=2))
    left_x = x - res_x
    spec = spec.reshape(1,spec.shape[0])
    return cc, spec,res_x

def contrain_FR(c, s, o):
    ind_s = np.argmax(np.abs(c[s]))

    if c[s][ind_s] < 0:
        c = -c
    
    if s[0]<o[0]:
        if c[s[-2]]<c[s[-1]]:
            ind1 = s[-1]
            ind2 = o[np.argmax(c[o])]
        else:
            ind1 = s[np.argmax(c[s])]
            ind2 = o[0]
    else:
        if c[s[1]] < c[s[0]]:
            ind1 = o[np.argmax(c[o])]
            ind2 = s[0]
        else:
            ind1 = o[-1]
            ind2 = s[np.argmax(c[s])]

    for i, indd in enumerate(np.arange(ind1, 0, -1)):
        if c[indd-1] >= c[indd]:
            c[0:indd] = 0
            break
        if c[indd-1] < 0:
            c[0:indd] = 0
            break

    for i, indd in enumerate(np.arange(ind2, len(c)-1, 1)):
        if c[indd+1] >= c[indd]:
            c[indd+1:len(c)] = 0
            break
        if c[indd+1] < 0:
            c[indd+1:len(c)] = 0
            break
    return c, ind_s

def mcr_by_fr(x,regions,COM):

    if COM == 3:
        p1 = regions[0,1]
        p2 = regions[1,0]
        p3 = regions[1,1]
        p4 = regions[2,0]
        
        if p1==p2:
            p2=p2+1
        s = list(range(0,min(int(p1),int(p2))))
        o = list(range(min(int(p1),int(p2)),max(int(p1),int(p2))))
        z = list(range(max(int(p1),int(p2)),x.shape[0])) 
        
        cc1, ss1, xx1 = FR(x, s, o, z, COM)  
    
        if p3==p4:
            p4=p4+1
        s = list(range(int(max(int(p3),int(p4))+1),x.shape[0]))
        o = list(range(min(int(p3),int(p4)),max(int(p3),int(p4))))
        z = list(range(0,min(int(p3),int(p4))))                 
        
        cc3, ss3, xx3 = FR(x, s, o, z, COM)
    
        xx2 = x-xx1-xx3
        
        ind_s = np.argmax(np.abs(np.sum(xx2,1)))
    
        cc2 = ittfa(xx2, ind_s, 1)
    
        ss2 = np.zeros((1, x.shape[1]))
        for j in range(0, ss2.shape[1]):
            a = fnnls(np.dot(cc2.T, cc2), np.dot(cc2.T, xx2[:, j]), tole='None')
            ss2[:, j] = a['xx']
    
        xx2 = np.dot(cc2,ss2)
    
        re_x = xx1+xx2+xx3
        
        re_chrom  = np.zeros((COM, x.shape[0]))
        re_chrom[0,:] = np.sum(xx1,1)
        re_chrom[1,:] = np.sum(xx2,1)
        re_chrom[2,:] = np.sum(xx3,1)
        
        R2 = explained_variance_score(x, re_x, multioutput='variance_weighted')
        xx = [xx1,xx2,xx3]
        S = np.concatenate((ss1,ss2,ss3),0)
    elif COM == 4:
        p1 = regions[0,1]
        p2 = regions[1,0]
        p3 = regions[1,1]
        p4 = regions[2,0]
        p5 = regions[2,1]
        p6 = regions[3,0]
        
        if p1==p2:
            p2=p2+1
        s = list(range(0,min(int(p1),int(p2))))
        o = list(range(min(int(p1),int(p2)),max(int(p1),int(p2))))
        z = list(range(max(int(p1),int(p2)),x.shape[0])) 
        
        cc1, ss1, xx1 = FR(x, s, o, z, COM)  
        
        xx_3 = x-xx1
    
        if p3==p4:
            p4=p4+1
            
            
        s = list(range(min(int(p1),int(p2)),min(int(p3),int(p4))))
        o = list(range(min(int(p3),int(p4)),max(int(p3),int(p4))))
        z = list(range(max(int(p3),int(p4)),x.shape[0])) 
        cc2, ss2, xx2 = FR(xx_3, s, o, z, int(COM-1))  
        
        if p5==p6:
            p6=p6+1
        s = list(range(int(max(int(p5),int(p6))+1),x.shape[0]))
        o = list(range(min(int(p5),int(p6)),max(int(p5),int(p6))))
        z = list(range(0,min(int(p5),int(p6))))                         
        cc4, ss4, xx4 = FR(xx_3, s, o, z, int(COM-1))
    
        xx3 = x-xx1-xx2-xx4
        
        ind_s = np.argmax(np.abs(np.sum(xx3,1)))
    
        cc3 = ittfa(xx3, ind_s, 1)
    
        ss3 = np.zeros((1, x.shape[1]))
        for j in range(0, ss3.shape[1]):
            a = fnnls(np.dot(cc3.T, cc3), np.dot(cc3.T, xx3[:, j]), tole='None')
            ss3[:, j] = a['xx']
    
        xx3 = np.dot(cc3,ss3)
    
        re_x = xx1+xx2+xx3+xx4
        
        re_chrom  = np.zeros((COM, x.shape[0]))
        re_chrom[0,:] = np.sum(xx1,1)
        re_chrom[1,:] = np.sum(xx2,1)
        re_chrom[2,:] = np.sum(xx3,1)
        re_chrom[3,:] = np.sum(xx4,1)
        
        R2 = explained_variance_score(x, re_x, multioutput='variance_weighted')
        xx = [xx1,xx2,xx3,xx4]
        S = np.concatenate((ss1,ss2,ss3,ss4),0)
    elif COM == 5:
        p1 = regions[0,1]
        p2 = regions[1,0]
        p3 = regions[1,1]
        p4 = regions[2,0]
        p5 = regions[2,1]
        p6 = regions[3,0]
        p7 = regions[3,1]
        p8 = regions[4,0]
        
        if p1==p2:
            p2=p2+1
        s = list(range(0,min(int(p1),int(p2))))
        o = list(range(min(int(p1),int(p2)),max(int(p1),int(p2))))
        z = list(range(max(int(p1),int(p2)),x.shape[0])) 
        
        cc1, ss1, xx1 = FR(x, s, o, z, COM)  
        
        if p7==p8:
            p8=p8+1
        s = list(range(max(int(p7),int(p8)),x.shape[0]))
        o = list(range(min(int(p7),int(p8)),max(int(p7),int(p8))))
        z = list(range(0,min(int(p7),int(p8))))                 
        
        cc5, ss5, xx5 = FR(x, s, o, z, COM)
    
        xx_3 = x-xx1-xx5

        if p3==p4:
            p4=p4+1     
        s = list(range(min(int(p1),int(p2)),min(int(p3),int(p4))))
        o = list(range(min(int(p3),int(p4)),max(int(p3),int(p4))))
        z = list(range(max(int(p3),int(p4)),x.shape[0])) 
        cc2, ss2, xx2 = FR(xx_3, s, o, z, int(COM-2))  
        
        if p5==p6:
            p6=p6+1        
        s = list(range(max(int(p5),int(p6)),max(int(p7),int(p8))))
        o = list(range(min(int(p5),int(p6)),max(int(p5),int(p6))))
        z = list(range(0,min(int(p5),int(p6))))                         
        cc4, ss4, xx4 = FR(xx_3, s, o, z, int(COM-2))
    
        xx3 = x-xx1-xx2-xx4-xx5
        
        ind_s = np.argmax(np.abs(np.sum(xx3,1)))
    
        cc3 = ittfa(xx3, ind_s, 1)
    
        ss3 = np.zeros((1, x.shape[1]))
        for j in range(0, ss3.shape[1]):
            a = fnnls(np.dot(cc3.T, cc3), np.dot(cc3.T, xx3[:, j]), tole='None')
            ss3[:, j] = a['xx']
    
        xx3 = np.dot(cc3,ss3)
    
        re_x = xx1+xx2+xx3+xx4+xx5
        
        re_chrom  = np.zeros((COM, x.shape[0]))
        re_chrom[0,:] = np.sum(xx1,1)
        re_chrom[1,:] = np.sum(xx2,1)
        re_chrom[2,:] = np.sum(xx3,1)
        re_chrom[3,:] = np.sum(xx4,1)
        re_chrom[4,:] = np.sum(xx5,1)
        
        R2 = explained_variance_score(x, re_x, multioutput='variance_weighted')
        xx = [xx1,xx2,xx3,xx4,xx5]
        S = np.concatenate((ss1,ss2,ss3,ss4,ss5),0)
    return xx,re_chrom,R2,S
    
    
def ittfa(d, needle, pcs):
    u, s, v = np.linalg.svd(d)
    t = np.dot(u[:,0:pcs], np.diag(s[0:pcs]))
    row = d.shape[0]
    cin = np.zeros((row, 1))
    cin[needle-1] = 1
    out = cin
    for i in range(0, 100):
        vec = out
        out = np.dot(np.dot(np.dot(t, np.linalg.pinv(np.dot(t.T, t))), t.T), out)
        out[out < 0] = 0
        out = unimod(out, 1.1, 2)
        if norm(out)!=0:     
            out = out/norm(out)
        kes = norm(out-vec)
        if kes < 1e-6 or iter == 99:
            break
    return out

def unimod(c, rmod, cmod, imax=None):
    ns = c.shape[1]
    if imax == None:
        imax = np.argmax(c, axis=0)
    for j in range(0, ns):
        rmax = c[imax[j], j]
        k = imax[j]
        while k > 0:
            k = k-1
            if c[k, j] <= rmax:
                rmax = c[k, j]
            else:
                rmax2 = rmax*rmod
                if c[k, j] > rmax2:
                    if cmod == 0:
                        c[k, j] = 0 #1e-30
                    if cmod == 1:
                        c[k, j] = c[k+1, j]
                    if cmod == 2:
                        if rmax > 0:
                            c[k, j] = (c[k, j]+c[k+1, j])/2
                            c[k+1, j] = c[k, j]
                            k = k+2
                        else:
                            c[k, j] = 0
                    rmax = c[k, j]
        rmax = c[imax[j], j]
        k = imax[j]

        while k < c.shape[0]-1:
            k = k+1
            if k==53:
                k=53
            if c[k, j] <= rmax:
                rmax = c[k, j]
            else:
                rmax2 = rmax*rmod
                if c[k, j] > rmax2:
                    if cmod == 0:
                        c[k, j] = 1e-30
                    if cmod == 1:
                        c[k, j] = c[k-1, j]
                    if cmod == 2:
                        if rmax > 0:
                            c[k, j] = (c[k, j]+c[k-1, j])/2
                            c[k-1, j] = c[k, j]
                            k = k-2
                        else:
                            c[k, j] = 0
                    rmax = c[k, j]
    return c

def fnnls(x, y, tole):
    xtx = np.dot(x, x.T)
    xty = np.dot(x, y.T)
    if tole == 'None':
        tol = 10*np.spacing(1)*norm(xtx, 1)*max(xtx.shape)
    mn = xtx.shape
    P = np.zeros(mn[1])
    Z = np.array(range(1, mn[1]+1), dtype='int64')
    xx = np.zeros(mn[1])
    ZZ = Z-1
    w = xty-np.dot(xtx, xx)
    iter = 0
    itmax = 30*mn[1]
    z = np.zeros(mn[1])
    while np.any(Z) and np.any(w[ZZ] > tol):
        t = ZZ[np.argmax(w[ZZ])]
        P[t] = t+1
        Z[t] = 0
        PP = np.nonzero(P)[0]
        ZZ = np.nonzero(Z)[0]
        nzz = np.shape(ZZ)
        if len(PP) == 1:
            z[PP] = xty[PP]/xtx[PP, PP]
        elif len(PP) > 1:
            if np.linalg.det(xtx[np.ix_(PP, PP)]) ==0:
                small = 1e-6*np.identity(xtx[np.ix_(PP, PP)].shape[0])
                z[PP] = np.dot(xty[PP], np.linalg.inv(xtx[np.ix_(PP, PP)]+small))
            else:
                z[PP] = np.dot(xty[PP], np.linalg.inv(xtx[np.ix_(PP, PP)]))
        z[ZZ] = np.zeros(nzz)
        while np.any(z[PP] <= tol) and iter < itmax:
            iter += 1
            qq = np.nonzero((tuple(z <= tol) and tuple(P != 0)))
            alpha = np.min(xx[qq] / (xx[qq] - z[qq]))
            xx = xx + alpha*(z - xx)
            ij = np.nonzero(tuple(np.abs(xx) < tol) and tuple(P != 0))
            Z[ij[0]] = ij[0]+1
            P[ij[0]] = np.zeros(max(np.shape(ij[0])))
            PP = np.nonzero(P)[0]
            ZZ = np.nonzero(Z)[0]
            nzz = np.shape(ZZ)
            if len(PP) == 1:
                z[PP] = xty[PP]/xtx[PP, PP]
            elif len(PP) > 1:
                z[PP] = np.dot(xty[PP], np.linalg.inv(xtx[np.ix_(PP, PP)]))
            z[ZZ] = np.zeros(nzz)
        xx = np.copy(z)
        w = xty - np.dot(xtx, xx)
    return{'xx': xx, 'w': w}
