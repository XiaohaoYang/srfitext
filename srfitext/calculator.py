import numpy as np

from diffpy.srreal.pdfcalculator import makePDFBaseline, PDFBaseline, PDFCalculator, DebyePDFCalculator
from diffpy.Structure import loadStructure
from diffpy.pdfgetx.cromermann import fxrayatq

class DPDFCalculator(object):
    '''
    calculate the differential PDF
    '''
    def __init__(self, calc, adele, stru, mode='ad', extlen=65536):
        self.calc = calc
        self.setAdStru(adele, stru, mode)
        self.extlen = extlen
        return
    
    def setAdStru(self, adele, stru, mode='ad'):
        '''
        if mode == ad -> return the ad pdf
        if mode == non-ad -> return the non-ad pdf
        '''
        self.adele = adele
        self.mode = mode
        self.periodic = stru.periodic
        s = loadStructure(stru.parent.filename)
        self.elements = list(set(s.element))
        # init c
        c = {}
        for ele in self.elements:
            c[ele] = 0.0
        tc = 0.0
        for ele, occ in zip(s.element, s.occupancy):
            c[ele] = c[ele] + occ
            tc = tc + occ
        for ele in self.elements:
            c[ele] = c[ele] / tc 
        self.c = c
        # init f
        self.f = {}
        ftotal = 0
        for ele in self.elements:
            self.f[ele] = fxrayatq(ele, [0])
            ftotal += fxrayatq(ele, [0]) * self.c[ele]
        self.f['total'] = ftotal
        self.adw = self.c[adele] * self.f[adele] / self.f['total']
        return
    
    def __call__(self, srrealstru):
        '''
        calculate the dPDF, for PDFCalc, use a fft filter to remove the weird background
        for DebyePDFCalc, directly return the background
        '''
        self.calc.rmax = 50.0
        if self.mode == 'non-ad':
            self.calc.setTypeMask('all', 'all', True)
            r, gr_total = self._getGr(srrealstru)
        
        self.calc.setTypeMask('all', 'all', False)
        self.calc.setTypeMask(self.adele, 'all', True)
        r, gr_ad_all = self._getGr(srrealstru)
        
        self.calc.setTypeMask('all', 'all', False)
        self.calc.setTypeMask(self.adele, self.adele, True)
        r, gr_ad_ad = self._getGr(srrealstru)
        gr_ad = (gr_ad_all + gr_ad_all) / 2
        gr_ad = gr_ad / self.adw
        
        if self.mode == 'non-ad':
            gr_non_ad = (gr_total - gr_ad * self.adw) / (1 - self.adw)
            return r, gr_non_ad
        else:
            return r, gr_ad
        
    def _getGr(self, srrealstru):
        '''
        calculate fq using PDF calculator, assuming f' and f" are 0
        '''
        rmax = self.calc.rmax
        rlen = len(self.calc.rgrid)
        # self.calc.rmax = rmax * 5
        
        self.calc(srrealstru)
        gr = self.calc.pdf
        
        if self.periodic:
            yy = np.concatenate([gr, np.zeros(self.extlen - len(gr))])
            rmax = self.calc.rstep * self.extlen
            fq = np.imag(np.fft.ifft(yy))
            qstep = 2 * np.pi / rmax
            ii = int(0.2 / qstep) + 1
            fq[:ii] = 0
            fq[-ii:] = 0
            gr1 = np.imag(np.fft.ifft(fq)) * (rmax * rmax / np.pi)
            gr1 = gr1[:rlen]
            
            # self.calc.rmax = rmax
            return self.calc.rgrid, gr1
        else:
            # self.calc.rmax = rmax
            return self.calc.rgrid, self.calc.pdf
    
