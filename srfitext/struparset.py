
import numpy as np
import itertools
import re
import os

from diffpy.srfit.fitbase.parameter import Parameter, ParameterProxy
from diffpy.srfit.fitbase.parameter import ParameterAdapter as ParameterAdapter
from diffpy.srfit.fitbase.parameterset import ParameterSet
from diffpy.srfit.structure.srrealparset import SrRealParSet
from diffpy.srfit.structure.objcrystparset import ObjCrystMoleculeParSet
from diffpy.srreal.structureadapter import nosymmetry

from srfitext.structure import StructureExt

def _getValue(obj, name):
    return getattr(obj, name)

class ParameterAdapterExt(ParameterAdapter):

    def setValue(self, value, lb=None, ub=None):
        """Set the value of the Parameter."""
        if isinstance(value, np.ndarray):
            if np.any(value != self.getValue()):
                self.setter(self.obj, value)
                self.notify()
        else:
            if value != self.getValue():
                self.setter(self.obj, value)
                self.notify()

        if lb is not None: self.bounds[0] = lb
        if ub is not None: self.bounds[1] = ub

        return self

class StructureExtParSet(SrRealParSet):

    def __init__(self, name, stru):
        SrRealParSet.__init__(self, name)
        self.stru = stru
        self.lat = stru.lat

        stru = self.stru
        occ = ParameterAdapterExt("occ", stru, getter=_getValue, setter=self._updateOcc, attr="occ")
        self.addParameter(occ)
        self.addParameter(ParameterProxy("occupancy", occ))
        uij = ParameterAdapterExt('uij', stru, getter=_getValue, setter=self._updateUij, attr='uij_c')
        self.addParameter(uij)
        self.addParameter(ParameterProxy('U', uij))
        xyz = ParameterAdapterExt('xyz', stru, getter=_getValue, setter=self._updateXYZ, attr='xyz_c')
        self.addParameter(xyz)
        uiso = ParameterAdapterExt('uiso', stru, getter=_getValue, setter=self._updateUiso, attr='uiso')
        self.addParameter(uiso)
        
        self._zoomscale = 1.0
        self.addParameter(ParameterAdapter('zoomscale', self, getter=_getValue, setter=self._updateZoomscale, attr='_zoomscale'))

        self._initSrRealStructure()
        return

    def getLattice(self):
        return self.lat

    @classmethod
    def canAdapt(self, stru):
        return isinstance(stru, StructureExt)

    def getScatterers(self):
        raise TypeError('StructureExtParSet does not support getScatteres')
        return None

    def _initSrRealStructure(self):
        self.n = self.stru.n
        self.srrstru = self.stru.convertPeriodicStru('xyz_c')
        return

    def _updateZoomscale(self, obj, name, value):
        for ii in xrange(self.n):
            self.srrstru[ii].xyz_cartn = self.stru.xyz_c[ii] * value
        self._zoomscale = value
        return

    def _updateXYZ(self, obj, name, value):
        ind = self.stru.xyz_c != value
        ind = np.sum(ind, axis=1) > 0
        i = np.nonzero(ind)[0]
        for ii in i:
            self.srrstru[ii].xyz_cartn = value[ii]
        self.stru.xyz_c = value
        return

    def _updateOcc(self, obj, name, value):
        ind = self.stru.occ != value
        i = np.nonzero(ind)[0]
        for ii in i:
            self.srrstru[ii].occupancy = value[ii]
        self.stru.occ = value
        return

    def _updateUij(self, obj, name, value):
        ind = self.stru.uij_c != value
        ind = np.sum(ind, axis=(1, 2))
        i = np.nonzero(ind)[0]
        for ii in i:
            self.srrstru[ii].uij_cartn = value[ii]
        self.stru.uij_c = value
        return

    def _updateUiso(self, obj, name, value):
        ind = self.stru.uiso != value
        i = np.nonzero(ind)[0]
        for ii in i:
            self.updateUisoi(ii, value[ii])
        return

    def updateXYZi(self, i, value):
        self.stru.xyz_c[i] = value
        self.srrstru[i].xyz_cartn = value
        return

    def updateOcci(self, i, value):
        self.stru.occ[i] = value
        self.srrstru[i].occupancy = value
        return

    def updateUiji(self, i, value):
        self.stru.uij_c[i] = value
        self.srrstru[i].uij_cartn = value
        return

    def updateUisoi(self, i, value):
        self.stru.uiso[i] = value
        self.stru.uij_c[i] = value * np.identity(3)
        self.srrstru[i].uij_cartn = value * np.identity(3)
        if self.stru.anisotropy[i]:
            self.stru.anisotropy[i] = False
            self.srrstru[i].anisotropy = False
        return

    def _getSrRealStructure(self):
        if self.stru.periodic:
            rv = self.srrstru
        else:
            rv = nosymmetry(self.srrstru)
        return rv

class ObjCrystMoleculeParSetExt(ObjCrystMoleculeParSet):
    
    def __init__(self, name, molecule, parent=None):
        """Initialize

        name    --  The name of the scatterer
        molecule    --  The pyobjcryst.Molecule instance
        parent  --  The ObjCrystCrystalParSet this belongs to (default None).

        """
        ObjCrystMoleculeParSet.__init__(self, name, molecule, parent)
        
        self._zoomscale = 1.0
        self.addParameter(ParameterAdapter('zoomscale', self, attr='_zoomscale'))
        return
    
    def _getSrRealStructure(self):
        """Get the structure object for use with SrReal calculators.

        Molecule objects are never periodic. Return the object and let the
        SrReal adapters do the proper thing.

        """
        from diffpy.srreal.srreal_ext import convertObjCrystMolecule
        stru = convertObjCrystMolecule(self.stru)
        if self._zoomscale != 1.0:
            for a in stru:
                a.xyz_cartn = self._zoomscale * a.xyz_cartn
        
        return stru

if __name__ == '__main__':
    stru = StructureExt()
    # stru.loadStrufile('CdS.cif', 'diffpy', periodic=True)
    stru.loadStrufile('cd35se20.xyz', 'diffpy', periodic=False)
    stru.parseDiffpyStru()

    struparaset = StructureExtParSet('test', stru)
    # a = np.zeros_like(struparaset.xyz.value)
    # struparaset.xyz = a

    # b = np.zeros_like(struparaset.occ.value)
    # struparaset.occ = b

    from diffpy.srreal.pdfcalculator import PDFCalculator, DebyePDFCalculator
    from matplotlib.pyplot import plot, show

    pc1 = DebyePDFCalculator()
    pc1.rmax = 20
    pc1.qmin = 1.0
    r1, g1 = pc1(struparaset._getSrRealStructure())
    plot(r1, g1)

    # xyz = np.array(struparaset.xyz.value)
    # struparaset.xyz = xyz * 1.1
    # uij = np.array(struparaset.uij.value)
    # struparaset.uij = uij * 3.0
    struparaset.zoomscale = 1.1
    r2, g2 = pc1(struparaset._getSrRealStructure())
    plot(r2, g2)

    show()

