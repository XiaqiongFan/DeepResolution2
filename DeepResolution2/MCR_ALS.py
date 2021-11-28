import numpy as np
from numpy import ix_
from pylab import plot, show, figure,subplot
from scipy.optimize import nnls

class mcr_als:
    def __init__(self, data, pures, parameter):
        self.x = data
        self.pures = pures
        self.para = parameter

    def pcarep(self, xi, nf):
        u, s, v = np.linalg.svd(xi)
        x = np.dot(np.dot(u[:, 0:nf], np.diag(s[0:nf])), v[0:nf, :])
        res = xi - x
        sst1 = np.power(res, 2).sum()
        sst2 = np.power(xi, 2).sum()
        sigma = sst1/sst2*100
        return u, s, v, x, sigma

    def fnnls(self, xtx, xty, tole):
        if tole == 'None':
            tol = 10*np.spacing(1)*np.linalg.norm(xtx, 1)*max(xtx.shape)
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
                z[PP] = np.dot(xty[PP], np.linalg.pinv(xtx[ix_(PP, PP)]))
            z[ZZ] = np.zeros(nzz)
            while np.any(z[PP] <= tol) and iter < itmax:
                iter += 1
                qq = np.nonzero((tuple(z <= tol) and tuple(P != 0)))
                alpha = np.min(xx[qq] / (xx[qq] - z[qq] + np.spacing(1)))
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
                    z[PP] = np.dot(xty[PP], np.linalg.pinv(xtx[ix_(PP, PP)]))
                z[ZZ] = np.zeros(nzz)
            xx = np.copy(z)
            w = xty - np.dot(xtx, xx)
        return{'xx': xx, 'w': w}

    def unimod(self, c, rmod, cmod):
        ns = c.shape[1]
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
                            c[k, j] = 1e-30
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

    def run(self):
        S = np.array([])
        rtpos = []
        pc = min(self.x.shape)
        r2opt = []

        tolsigma = 0.1
        niter = 0
        idev = 0
        dn = self.x
        u, s, v, d, sd = self.pcarep(dn, pc)
        sstn = np.sum(np.power(dn, 2))
        sst = np.sum(np.power(d, 2))
        sigma2 = np.power(sstn, 0.5)
        nit = 30
        crs = self.pures
        while niter < nit:
            niter += 1

            if d.shape[0] == crs.shape[0]:
                conc = crs
                spec = np.dot(np.dot(np.linalg.pinv(np.dot(conc.T, conc)), conc.T), d)
            elif d.shape[1] == crs.shape[1]:
                spec = crs
                conc = np.dot(np.dot(d, spec.T), np.linalg.pinv(np.dot(spec, spec.T)))
            else:
                print('please imput right initial estimation')

            conc2 = np.zeros(conc.shape)
            if self.para[0][0] == "fnnls":
                for j in range(0, conc.shape[0]):
                    a = self.fnnls(np.dot(spec, spec.T), np.dot(spec, d[j, :].T), tole='None')
                    conc2[j, :] = a['xx']
            elif self.para[0][0] == "nnls":
                for j in range(0, conc.shape[0]):
                    a, rnomal = nnls(np.dot(spec, spec.T), np.dot(spec, d[j, :].T))
                    conc2[j, :] = a
            # elif self.para[0][0] == "none":

            conc = conc2

            if self.para[0][1] == "average":
                mod = [1.1, 1]
                conc2 = self.unimod(conc, mod[0], mod[1])
                conc = conc2

            spec2 = np.zeros(spec.shape)
            if self.para[1][0] == "fnnls":
                for j in range(0, spec.shape[1]):
                    a = self.fnnls(np.dot(conc.T, conc), np.dot(conc.T, d[:, j]), tole='None')
                    spec2[:, j] = a['xx']
            elif self.para[1][0] == "nnls":
                for j in range(0, spec.shape[1]):
                    a, rnomal = nnls(np.dot(conc.T, conc), np.dot(conc.T, d[:, j]))
                    spec2[:, j] = a
            spec = spec2

            # for i in range(0, spec.shape[1]):
            #     plot(np.dot(conc[:, i:i + 1], spec[i:i + 1, :]))
            # show()

            res = d - np.dot(conc, spec)
            resn = dn - np.dot(conc, spec)
            u = np.sum((np.power(res, 2)))
            un = np.sum((np.power(resn, 2)))
            sigma = np.power(u / (d.shape[0] * d.shape[1]), 0.5)
            change = (sigma2 - sigma) / sigma

            if change < 0.0: idev += 1
            else: idev = 0
            change = np.dot(100, change)
            lof_pca = np.power((u /sst), 0.5 ) *100
            lof_exp = np.power((un /sstn), 0.5 ) *100
            r2 = (sstn -un ) /sstn
            if change > 0 or niter == 1:
                #print("change:" + str(change) + "; sigma:" +str(sigma)+"; sigma2:"+str(sigma2))
                sigma2 = sigma
                copt = conc
                sopt = spec
                itopt = niter +1
                sdopt = np.array([lof_pca, lof_exp])
                ropt = res
                r2opt = r2
            if abs(change) < tolsigma:
                sigma2 = sigma
                copt = conc
                sopt = spec
                itopt = niter +1
                sdopt = np.array([lof_pca, lof_exp])
                ropt = res
                r2opt = r2
                #print('CONVERGENCE IS ACHIEVED, STOP!!!')
                break
            if idev >= 10:
                #print('FIT NOT IMPROVING FOR 20 TMES CONSECUTIVELY (DIVERGENCE?), STOP!!!')
                break
            crs = spec
            #print("MCR-ALS model continue in: " + str(niter) + "th")
        R2 = r2opt
        SD = sdopt
        RES = res
        maxpos = np.argmax(copt, axis=0)
        index = np.argsort(maxpos)
        # apexpos = np.sort(np.argmax(copt, axis=0))
        # index = np.argsort(np.argmax(copt, axis=0))
        C = copt[:, index]
        S = sopt[index,:]
        # for i in range(0, C.shape[1]):
        #     plot(np.dot(C[:, i:i + 1], S[i:i + 1, :]))
        # show()
        # plot(np.dot(C, S))
        # show()
        return [np.sort(maxpos), C, S, SD, R2, RES, itopt]

def unimod(c, rmod, cmod):
    ns = c.shape[1]
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
                        c[k, j] = 1e-30
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

def unimod1(cc, startpoint, rmod, cmod):
    c = np.copy(cc)
    ns = c.shape[1]
    imax = [startpoint] #np.argmax(c, axis=0)
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
                        c[k, j] = 1e-30
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



def pure(d, nr, f):
    rc = d.shape
    p = np.zeros((nr, rc[1]))
    s = np.zeros((nr, rc[1]))
    imp = list()
    s[0, :] = np.std(d, axis=0, ddof=1)
    m = np.mean(d, axis=0)
    ll = s[0, :]**2 + m**2
    f = f/100.00*m.max()
    pp = s[0, :]/(m+f)
    imp.append(np.argmax(pp))
    l = np.power(s[0, :]**2+(m+f)**2, 0.5)
    dl = np.zeros(d.shape)
    for j in range(0, rc[1]):
        dl[:, j] = d[:, j]/l[j]
    c = np.dot(dl.T, dl)/rc[0]
    # plot(c)
    # show()
    w = np.zeros((nr, rc[1]))
    w[0, :] = ll / (l**2)
    p[0, :] = w[0, :]*pp
    s[0, :] = w[0, :]*s[0, :]
    for i in range(1, nr):
        for j in range(0, rc[1]):
            dm = wmat(c, imp[0:i], i, j)
            w[i, j] = np.linalg.det(dm)
            p[i, j] = p[0, j]*w[i, j]
            s[i, j] = s[0, j]*w[i, j]
        imp.append(np.argmax(p[i, :]))
    # plot(p.T)
    # show()
    sn = d[:, imp]
    sp = sn.T
    for bit in range(0, sp.shape[0]):
        sr = np.linalg.norm(sp[bit, :])
        sp[bit, :] = sp[bit, :]/(sr+1e-10)
    return{'SP': sp, 'IMP': imp}

def wmat(c, imp, irank, jvar):
    dm = np.zeros((irank+1, irank+1))
    dm[0, 0] = c[jvar, jvar]
    for k in range(1, irank+1):
        kvar = imp[k-1]
        dm[0, k] = c[jvar, kvar]
        dm[k, 0] = c[kvar, jvar]
        for kk in range(1, irank+1):
            kkvar = imp[kk-1]
            dm[k, kk] = c[kvar, kkvar]
    return dm

def FinderLocalMax(pp, width=5):
    LocalMaxPos = []
    LocalMaxValue = []
    halfNum = int(width/2)
    for i in range(0, len(pp)):
        # find peak top point
        top = True
        for j in range(i-halfNum, i+halfNum):
            if j>=0 and j<i:
                if pp[j] > pp[j+1]:
                    top = False
                    break
            if j+1<=len(pp)-1 and j >= i:
                if pp[j] < pp[j+1]:
                    top = False
                    break
        if top == True:
            LocalMaxPos.append(i)
            LocalMaxValue.append(pp[i])
    return np.array(LocalMaxPos), np.array(LocalMaxValue)

def SelectPureChromatogram(Pures, thres=0.05):
    Dets = []
    PC_number = 1
    for i in range(1, Pures.shape[0]+1):
        Y = np.dot(Pures[0:i, :], Pures[0:i, :].T)
        Dets.append(np.linalg.det(Y))

    # plot(Dets)
    # show()
    for j in range(0, len(Dets)-1):
        if Dets[j] <= thres:
            PC_number = j
            break
    # plot(np.arange(1,len(Dets)+1),Dets)
    # plot(np.arange(1,len(Dets)+1), 0.2*np.ones(len(Dets)))
    # plot(np.arange(0,len(Dets))[j],np.array(Dets)[j-1], 'ro')
    # show()
    # DifDet = []
    # for j in range(0, len(Dets)-1):
    #     DifDet.append(abs(Dets[j]-Dets[j+1]))
    # maxDifDetPos = np.argmax(DifDet)
    # PC_number = maxDifDetPos+2
    PureChromatograms = Pures[0:PC_number,:]
    return PureChromatograms, PC_number

def iskss(X, t, siglev=3.0):
    # S, t = keyspectra(X)

    if len(t)==1:
        S = X[t[0]:t[0]+1,:]
        C = np.dot(X, np.linalg.pinv(S))
        Res = X - np.dot(C, S)
        return C, S.T, Res

    (mx, nx) = X.shape
    pcnum = len(t)
    c = np.zeros((mx, pcnum))
    s_i_ks = X[t[0],:]
    s_i_ks = s_i_ks/np.linalg.norm(s_i_ks)
    for i in range(t[1]+1, mx):
        X_sub = X[i:mx,:]
        U,s,V = np.linalg.svd(X_sub)
        S_tmp = np.vstack((s_i_ks, V[0:pcnum,:]))
        C_tmp = np.dot(X, np.linalg.pinv(S_tmp))
        # plot(C_tmp[:,0])
        # show()
        # if i == t[1]+1+4:
        #     show()
        c_bk = C_tmp[i:mx,0]
        c_val = min(c_bk)
        if np.abs(c_val) < siglev*np.std(c_bk):
            c[:, 0] = C_tmp[:, 0]
            break

    s_i_ks = X[t[pcnum-1],:]
    s_i_ks = s_i_ks/np.linalg.norm(s_i_ks)
    for i in range(t[pcnum-2], 1, -1):
        X_sub = X[0:i, :]
        U,s,V = np.linalg.svd(X_sub)
        S_tmp = np.vstack((V[0:pcnum-1], s_i_ks))
        C_tmp = np.dot(X, np.linalg.pinv(S_tmp))
        # plot(C_tmp[:,-1])
        # show()
        c_bk = C_tmp[0:i, -1]
        c_val = min(c_bk)
        if np.abs(c_val) < siglev*np.std(c_bk):
            c[:, pcnum-1] = C_tmp[:, pcnum-1]
            break

    for i in range(1, pcnum-1):
        s_i_ks = X[t[i],:]
        s_i_ks = s_i_ks/np.linalg.norm(s_i_ks)
        X_sub = X[t[i+1]:mx,:]
        U,s,V = np.linalg.svd(X_sub)
        V_ks_right = V[0:pcnum-i-1,:]
        locat = 0
        for j in range(t[i-1], 1, -1):
            X_sub = X[0:j,:]
            U,s,V = np.linalg.svd(X_sub)
            S_tmp = np.vstack((V[0:i, :], V_ks_right, s_i_ks))
            C_tmp = np.dot(X, np.linalg.pinv(S_tmp))
            # plot(C_tmp[:,-1])
            # show()

            c_bk = C_tmp[0:j, -1]
            c_val = min(c_bk)
            if np.abs(c_val) < siglev*np.std(c_bk):
                locat = j
                c[:, i] = C_tmp[:, -1]
                break

        if locat == 0:
            V_ks_left =[]
        else:
            X_sub = X[0:locat,:]
            U, s, V = np.linalg.svd(X_sub)
            V_ks_left = V[0:i, :]
        for j in range(t[i+1]+1,mx):
            X_sub = X[j:mx,:]
            U, s, V = np.linalg.svd(X_sub)
            if locat == 0:
                S_tmp = np.vstack((V[0:pcnum-i-1,:], s_i_ks))
            else:
                S_tmp = np.vstack((V_ks_left, V[0:pcnum-i-1,:], s_i_ks))
            C_tmp = np.dot(X, np.linalg.pinv(S_tmp))
            # plot(C_tmp[:,-1])
            # show()
            c_bk = C_tmp[j:mx,-1]
            c_val = min(c_bk)
            if np.abs(c_val) <siglev*np.std(c_bk):
                c[:, i] = C_tmp[:, -1]
                break

    C1 = c
    C = unimod(C1, 1.1, 0)
    C[C<0] =1e-10
    S = np.zeros((C.shape[1], X.shape[1]))
    for j in range(0, X.shape[1]):
        a, rnomal = nnls(np.dot(C.T, C), np.dot(C.T, X[:, j]))
        S[:, j] = a

    #S = np.dot(np.linalg.pinv(C), X)
    Res = X-np.dot(C, S)
    S = S.T

    return C, S, Res

if __name__ == "__main__":
    from NetCDF import netcdf_reader
    from pylab import plot, show, figure, scatter, xlabel, ylabel, hist, ylim, axhline, vlines, subplot
    import sys


    filename = 'C:/Users/admin/Desktop/DeepResolution/CDF/zhi10-5vs1.CDF'
    ncr = netcdf_reader(filename, bmmap=False)
    m = ncr.mat(0, ncr.scan_acquisition_time.shape[0] - 1, 1)

    m = m['d']
    plot(m)
    show()

    #Pures = SelectPureChromatogram(m, 0.0001, 10)

    # t = [6, 7, 8,13]#[0, 8, 12,14,16,20] #[9, 12, 15,20]
    # C, S, Res = iskss(m, t)
    # plot(C)
    # show()
    #
    # for i in range(0, C.shape[1]):
    #     plot(np.dot(C[:, i:i+1],S[:,i:i+1].T))
    # # plot(np.dot(C,S.T))
    # show()
    Pures = pure(m, 10, 10)
    #plot(Pures['SP'][0:3,:].T)
    # show()
    #Cpure,PC_number = SelectPureChromatogram(Pures['SP'], 0.2)
    #pure = filtering(Cpure, width=5)
    # plot(Cpure.T)
    # show()
    #
    # print(np.corrcoef(pure[:,0:2].T))

    # plot(m[:,Pure['IMP']])
    # show()
    # pures = m[:,Pure['IMP']]
    # print(Pure['IMP'])
    
    Cpure, PC_number = SelectPureChromatogram(Pures['SP'], 0.2)
    
    parameter = [['nnls', 'average'], ['nnls']]
    MCRALSModel = mcr_als(m, Cpure.T, parameter)
    #MCRALSModel = mcr_als(m, Pures, parameter)
    Dec = MCRALSModel.run()

    # plot(m, 'r')
    print(Dec[4])
    C = Dec[1]
    S = Dec[2]
    # plot(m-np.dot(C,S))
    # show()
    fig = figure()
    fig.add_subplot(111)
    colors = ['b', 'y', 'g', 'c', 'k', 'r', 'm']
    for i in range(0, C.shape[1]):
        plot(np.dot(C[:, i:i + 1], S[i:i + 1, :]), colors[i])  #
    show()


    #ss, t = keyspectra(m)
    # C,S,Res = iskss(m,[0,8,13,21])
    # plot(C)
    # show()
    #
    # colors = ['b', 'y', 'g', 'c', 'k', 'r', 'm']
    # for i in range(0, C.shape[1]):
    #     plot(np.dot(C[:, i:i + 1], S[i:i + 1, :]) , colors[i])  # , colors[i]
    # show()

