import numpy as np
import itertools

from diffpy.srreal.pdfcalculator import makePDFBaseline, PDFBaseline, PDFCalculator, DebyePDFCalculator
from diffpy.Structure import loadStructure
from diffpy.pdfgetx.cromermann import fxrayatq

class DPDFCalculator(object):
    '''
    calculate the differential PDF
    '''
    def __init__(self, calc, adele, stru, dPDFmode='ad', extlen=65536):
        self.calc = calc
        self.setAdStru(adele, stru, dPDFmode)
        self.extlen = extlen
        return
    
    def setAdStru(self, adele, stru, dPDFmode='ad'):
        '''
        if dPDFmode == ad -> return the ad pdf
        if dPDFmode == non_ad -> return the non_ad pdf
        '''
        self.adele = adele
        self.dPDFmode = dPDFmode
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
        if self.dPDFmode == 'total':
            self.calc.setTypeMask('all', 'all', True)
            return self.calc(srrealstru)
        
        self.calc.rmax = 50.0
        a = self.adele
        gr_ad = np.zeros_like(self.calc.rgrid)
        
        if self.dPDFmode == 'non_ad':
            self.calc.setTypeMask('all', 'all', True)
            r, gr_total = self._getGr(srrealstru) 
        
        for b in self.elements:
            self.calc.setTypeMask('all', 'all', False)
            self.calc.setTypeMask(a, b, True)
            r, gr = self._getGr(srrealstru)
            weight = self.c[a] * self.c[b] * self.f[a][0] * self.f[b][0] / (self.f['total'][0] ** 2)
            gr = gr / weight
            if a != b:
                gr = gr / 2.0 
            gr_ad += gr * self.c[b] * self.f[b][0] / self.f['total'][0]
        
        if self.dPDFmode == 'non_ad':
            weight = self.c[a] * self.f[a][0] / self.f['total'][0]
            gr_nonad = (gr_total - gr_ad * weight) / (1 - weight)
            return r, gr_nonad
        elif self.dPDFmode == 'ad':
            return r, gr_ad
        else:
            raise ValueError('dPDFmode should be ad, non-ad, or total')
        
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
    
