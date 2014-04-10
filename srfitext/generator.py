#!/usr/bin/env python
import numpy as np

from diffpy.srfit.pdf import PDFGenerator, DebyePDFGenerator
from diffpy.srfit.pdf.basepdfgenerator import BasePDFGenerator
from diffpy.srfit.fitbase.parameter import ParameterAdapter, Parameter
from diffpy.srfit.fitbase import ProfileGenerator
from diffpy.srreal.pdfcalculator import makePDFBaseline, PDFBaseline, PDFCalculator, DebyePDFCalculator
from diffpy.srfit.structure import struToParameterSet

from srfitext.struparset import StructureExtParSet, ObjCrystMoleculeParSetExt
from srfitext.structure import StructureExt
from pyobjcryst._pyobjcryst import Molecule

def fvarbkg(x, bkgslope):
    return bkgslope * x
makePDFBaseline("varbkg", fvarbkg, bkgslope=0)


class PDFGeneratorExt(BasePDFGenerator):
    '''
    PDF Generator
    '''
    def __init__(self, name="pdf", mode='pdf'):
        """Initialize the generator.

        """
        self.mode = mode
        BasePDFGenerator.__init__(self, name)
        if mode == 'pdf':
            self._setCalculator(PDFCalculator())
            self.periodic = True
        elif mode == 'debye':
            self._setCalculator(DebyePDFCalculator())
            self.periodic = False
        else:
            raise TypeError('PDF generator mode wrong, should be "pdf" or "debye"')
        return

    def _setCalculator(self, calc):
        """Set the SrReal calulator instance.

        Setting the calculator creates Parameters from the variable attributes
        of the SrReal calculator.

        """
        self._calc = calc
        for pname in self.__class__._parnames + ['qmax', 'qmin']:
            self.addParameter(
                ParameterAdapter(pname, self._calc, attr=pname)
                )
        self.processMetaData()
        return

    def setStructure(self, stru, name=None):
        """Set the structure that will be used to calculate the PDF.
        
        periodic is determined according to stru if not specfied
        if self.periodic is not same as self.periodic then raise error
        """

        # Create the ParameterSet
        name = stru.name if name == None else name
        if stru.periodic == self.periodic:
            periodic = self.periodic
        else:
            raise TypeError('stru.periodic != generator.periodic')

        if isinstance(stru, StructureExt):
            parset = StructureExtParSet(name, stru)
        elif isinstance(stru, Molecule):
            parset = ObjCrystMoleculeParSetExt(name, stru)
        else:
            parset = struToParameterSet(name, stru)
        # Set the phase
        self.setPhase(parset, periodic)

        for par in stru._params:
            self.addParameter(ParameterAdapter(par, stru, attr=par))
        return

    def setPhase(self, parset, periodic=None):
        """Set the phase that will be used to calculate the PDF.
        """
        if periodic == None:
            periodic = self.periodic

        return BasePDFGenerator.setPhase(self, parset, periodic)

    def setQmin(self, qmin):
        """Set the qmin value.
        """
        if self.periodic:
            self._calc.qmin = qmin
        self.meta["qmin"] = self.getQmin()
        return
    
    def setOptimized(self, optimized=True):
        if optimized:
            self._calc.evaluatortype = 'OPTIMIZED'
        else:
            self._calc.evaluatortype = 'DEFAULT'
        return

    def __call__(self, r):
        if r is not self._lastr:
            self._prepare(r)

        stru = self._phase.stru

        rcalc, y = self._calc(self._phase._getSrRealStructure())

        if np.isnan(y).any():
            y = np.zeros_like(r)
        else:
            y = np.interp(r, rcalc, y)
        return y


class GrGenerator(ProfileGenerator):
    '''genertor for scaling PDF
    '''
    def __init__(self, name="pdf"):
        """Initialize the generator."""
        ProfileGenerator.__init__(self, name)

        self._pdfprofile = None
        # self.xscale = 1.0
        # self.yscale = 1.0
        self.meta = {'xscale':1.0,
                     'yscale':1.0}
        self._lastr = None

        self._setCalculator()
        return

    _parnames = ['xscale', 'yscale']

    def _setCalculator(self):
        """Set the SrReal calulator instance.

        Setting the calculator creates Parameters from the variable attributes
        of the SrReal calculator.

        """
        for pname in self.__class__._parnames:
            self.addParameter(
                Parameter(pname, value=self.meta[pname])
                )
        self.processMetaData()
        return

    def setStructure(self, stru, name='phase', periodic=True):
        '''read _pdfprofile from stru (abnormal stru)
        '''
        self.stru = stru
        self._pdfprofile = stru._pdfprofile
        return

    def parallel(self, ncpu, mapfunc=None):
        """dummy method
        """
        return

    def processMetaData(self):
        """Process the metadata once it gets set."""
        ProfileGenerator.processMetaData(self)

        for name in self.__class__._parnames:
            val = self.meta.get(name)
            if val is not None:
                par = self.get(name)
                par.setValue(val)
        return

    def _validate(self):
        """Validate my state.

        This validates that the phase is not None.
        This performs ProfileGenerator validations.

        Raises AttributeError if validation fails.
        
        """
        # FIXME
        if self._pdfprofile is None:
            raise AttributeError("_pdfprofile is None")
        ProfileGenerator._validate(self)
        return

    def __call__(self, r):
        """Calculate the PDF.

        """
        y = np.interp(r, self._pdfprofile[0] * self.xscale.value, self._pdfprofile[1] * self.yscale.value)
        return y

