__author__ = 'Administrator'

import scipy.io.netcdf as nc
from numpy import  hstack
import numpy as np
import sys
from matplotlib.ticker import FormatStrFormatter
from matplotlib.pyplot import figure, show, plot, xlabel,ylabel,subplot,legend,title
from pylab import vlines
import matplotlib.pyplot as plt

class netcdf_reader:
    def __init__(self, fname, bmmap=True):
        self.filename = fname
        self.f = nc.netcdf_file(fname, 'r', mmap=bmmap)
        self.mass_values = self.f.variables['mass_values']
        self.intensity_values = self.f.variables['intensity_values']
        self.scan_index = self.f.variables['scan_index']
        self.total_intensity = self.f.variables['total_intensity']
        self.scan_acquisition_time = self.f.variables['scan_acquisition_time']
        self.mass_max = np.max(self.f.variables['mass_range_max'].data)
        self.mass_min = np.max(self.f.variables['mass_range_min'].data)
        if sys.byteorder == "little":
            self.nbo = "<"
        else:
            self.nbo = ">"

    def mz_point(self, n):
        scan_index_end = hstack((self.scan_index.data, np.array([len(self.intensity_values.data)], dtype=int)))

        ms = {}
        inds = range(scan_index_end[n], scan_index_end[n + 1])
        ms['mz'] = self.mass_values[inds]
        ms['val'] = self.intensity_values[inds]
        return ms

    def mz_rt(self, t):
        scan_index_end = hstack((self.scan_index.data, np.array([len(self.intensity_values.data)], dtype=int)))
        ms = {}
        tic_dict = self.tic()
        rt = tic_dict['rt']
        n = np.searchsorted(rt, t)
        inds = range(scan_index_end[n], scan_index_end[n + 1])
        ms['mz'] = self.mass_values[inds]
        ms['val'] = self.intensity_values[inds]
        return ms

    def tic(self):
        tic_dict = {'rt': self.scan_acquisition_time.data / 60.0, 'val': self.total_intensity.data}
        if tic_dict['val'].dtype.byteorder != self.nbo:
            tic_dict['val'] = tic_dict['val'].byteswap().newbyteorder()
        return tic_dict

    def mat_rt(self, rt_start, rt_end):
        indmin, indmax = np.searchsorted(self.tic()['rt'], (rt_start, rt_end))
        rt = self.tic()['rt'][indmin:indmax + 1]
        mass_max = np.max(self.f.variables['mass_range_max'].data)
        mass_min = np.max(self.f.variables['mass_range_min'].data)
        mz = np.linspace(mass_min, mass_max, num=mass_max - mass_min + 1)
        return {'mat': self.mat(indmin, indmax, 1)['mat'], 'rt': rt, 'mz': mz}

    def mat(self, imin, imax, bin):
        f = nc.netcdf_file(self.filename, 'r', mmap=False)
        t = f.variables['scan_acquisition_time'].data / 60.0
        mass_values = f.variables['mass_values']

        intensity_values = f.variables['intensity_values']

        scan_index = f.variables['scan_index']
        scan_index_end = np.hstack((scan_index.data, np.array([len(intensity_values.data)], dtype=int)))
        mass_max = np.max(f.variables['mass_range_max'].data)

        mass_min = np.max(f.variables['mass_range_min'].data)

        mz = np.linspace(mass_min, mass_max, num=int((mass_max - mass_min)/bin+1))
        if imin < 0 and imax > len(t):
            print('please print suitable index of retention time')
            exit()
        rt = t[imin:imax+1]
        c = int(imax-imin+1)
        r = int((mass_max-mass_min)/bin+1)
        mo = np.zeros((c, r))
        for j in range(imin, imax+1):
            msnext = scan_index_end[j+1]
            if msnext > np.shape(mass_values):
                mo[j:imax, :] = []
                break
            else:
                mz_val = mass_values[scan_index_end[j]:scan_index_end[j + 1]]
                sp = intensity_values[scan_index_end[j]:scan_index_end[j + 1]]
                ind = np.round((mz_val - mass_min) / bin)
                position = np.nonzero(mz_val > np.max(mz))
                ind = np.delete(ind, position)
                sp = np.delete(sp, position)
                ind2 = np.unique(ind)
                if np.shape(ind2) != np.shape(ind):
                    sp2 = np.zeros(ind2.shape[0])
                    for i in range(0, ind2.shape[0]-1):
                        tempind = np.nonzero(ind == ind2[i])
                        sp2[i] = np.sum(sp[tempind[0]])
                else:
                    sp2 = sp
                    ind2 = ind
                mo[j-imin, np.array(ind2, dtype=np.int32)] = sp2
        return{'d': mo, 'rt': rt, 'mz': mz}


def plot_ms(ms):
    fig = figure()
    ax = fig.add_subplot(111)
    ax.vlines(ms['mz'], np.zeros((len(ms['mz']),)), ms['val'], color='b', linestyles='solid')
    ax.xaxis.set_major_formatter(FormatStrFormatter('%3.4f'))
    show()


def plot_tic(tic_dict):
    figure()
    plot(tic_dict['rt'], tic_dict['val'])
    xlabel('Rentention time')
    ylabel('Intensity') 
    title('TIC')
    legend()
    show()



if __name__ == '__main__':

    
    filename = 'D:/my/DeepResolution/CDF/zhi10-5vs1.CDF'
    #filename = 'D:/GCMS/LHM/Deconvolution software validation/cdf/normal1/2C_1.cdf'
    #filename = 'D:/GCMS/tbb/metabolomics/080603cdf/正常/244-2.CDF'
    #filename = 'D:/GCMS/highGCMS/椒样薄荷/B/20120613_8UGJIAOYANGBOHE03.CDF'#10556
    filename = 'D:/GCMS/male infertility plasma GC-MS data/272.CDF'
    filename = 'F:/GCMS/LMJ/标准品CDF/hb-1.CDF'#7739
    ncr = netcdf_reader(filename, bmmap=False)
    m = ncr.mat(1,3599,1)
    RT = m['rt']
    Xtest = m['d'] 
    plt.figure()
    plt.plot(Xtest) 


    RT_sta = np.searchsorted(RT,7.81)
    RT_end = np.searchsorted(RT,7.93)
    plt.figure()
    plot(Xtest[RT_sta:RT_end,:])
    
    
    tic = ncr.tic()  
    plot_tic(tic)

    ll=ncr.mz_rt(10.77)
    plot_ms(ll)
    #print(np.mean(abs(ll['val'])))
    


     
    
    #np.save('MP_test.npy',Test)
    #np.save('MP_test_rt.npy',RT)
    
    
    
    
    


