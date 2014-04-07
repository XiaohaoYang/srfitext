#!/usr/bin/env python

from diffpy.Structure import Structure
from diffpy.Structure.utils import _linkAtomAttribute, atomBareSymbol
from diffpy.pdfgetx.functs import loadData

from diffpy.Structure import loadStructure, Structure, Atom
from pyobjcryst.crystal import CreateCrystalFromCIF
from pyobjcryst.crystal import Crystal
from pyobjcryst.molecule import Molecule
from pyobjcryst.scatteringpower import ScatteringPowerAtom

from diffpy.srreal.srreal_ext import PeriodicStructureAdapter
from diffpy.srreal.srreal_ext import nosymmetry
from diffpy.srreal.srreal_ext import Atom as AdapterAtom

import numpy as np
import itertools
import re
import os

class StructureExt(object):

    _params = None
    _extparams = None

    def __init__(self, name='stru', filename=None, loadstype=None, periodic=None):
        self.name = name
        self.rawstru = None
        self.rawstype = None
        self.stru = None
        self.periodic = periodic

        self.n = None
        self.element = None
        self.occ = None
        self.xyz = None
        self.xyz_c = None
        self.uij_c = None
        self._uiso = None
        self.anisotropy = None

        self.lat = None

        self._params = {}
        self._extparams = {}

        if filename != None:
            self.loadStrufile(filename, loadstype, periodic)
        return

    def _getUiso(self):
        rv = np.sum(self.uij_c, axis=(1, 2)) / 3
        return rv

    def _setUiso(self, value):
        self.anisotropy = np.ones(self.n, dtype=bool)
        self.uij_c = value.reshape(value.shape[0], 1, 1) * np.identity(3).reshape(1, 3, 3)
        return

    uiso = property(_getUiso, _setUiso, "Uiso")

    def convertStru(self, stype='diffpy', mode='xyz', periodic=None):
        '''
        convert stru to stype
        '''
        if stype == 'diffpy':
            rv = self.convertDiffpyStru(mode)
        elif stype == 'periodic':
            rv = self.convertPeriodicStru(mode)
        elif stype == 'objcryst':
            if self.rawstype == 'objcryst':
                rv = self.addProp(self.rawstru)
            else:
                rv = self.convertObjcrystStru(mode)
        elif stype == 'struext':
            rv = self
        else:
            raise TypeError('stype error')
        return rv

    def loadStrufile(self, filename, stype='diffpy', periodic=None):
        '''
        read and parse a structure file
        '''
        ext = os.path.splitext(filename)[1]

        if periodic != None:
            self.periodic = periodic
        else:
            # detect periodicity using file type
            if ext in ['.cif']:
                periodic = True
            else:
                periodic = False
            self.periodic = periodic

        # read the file
        if stype == 'diffpy':
            self.rawstru = loadStructure(filename)
            self.rawstype = stype
            self.parseDiffpyStru(self.rawstru)
        elif stype == 'objcryst':
            if ext == '.cif':
                self.rawstru = CreateCrystalFromCIF(file(filename))
                self.rawstype = stype
            else:
                raise TypeError('Cannot read file!')
        else:
            raise TypeError('Cannot read file!')
        return

    def exportStru(self, filename, format='cif', stype='diffpy'):
        '''
        Save structure to file in the specified format
        
        :param filename: str, name of the file
        :param stype: str, the type of stru file to export
        
        :return: None

        Note: available structure formats can be obtained by:
            from Parsers import formats
        '''
        from diffpy.Structure.Parsers import getParser
        p = getParser(format)
        p.filename = filename
        stru = self.convertStru(stype)
        s = p.tostring(stru)
        f = open(filename, 'wb')
        f.write(s)
        f.close()
        return

    ### Tools ###

    def addProp(self, stru):
        '''
        add properties to the stru
        '''
        stru.title = self.name
        stru.name = self.name
        stru._params = self._params
        stru._extparams = self._extparams
        stru.periodic = self.periodic
        stru.parent = self
        return stru

    ###########################################################
    # parse functions
    ###########################################################

    def parseDiffpyStru(self, stru=None):
        '''
        parse stru and store the information to self.xxx
        '''
        stru = self.rawstru if stru == None else stru

        n = len(stru)
        ulist = np.concatenate([0.001 * np.eye(3, 3) if (np.sum(u) == 0) else u for u in stru.U]).reshape(n, 3, 3)
        stru.U = ulist

        self.element = stru.element
        self.occ = stru.occupancy
        self.xyz = stru.xyz
        self.xyz_c = stru.xyz_cartn
        self.uij_c = stru.U
        self.anisotropy = stru.anisotropy
        self.n = len(self.anisotropy)

        self.lat = stru.lattice.abcABG()
        return

    def parseObjcrystStru(self, stru=None):
        '''
        parse stru and store the information to self.xxx
        FIXME: not complete
        '''
        # raise TypeError('parse a objcryst object is not reliable')
        stru = self.rawstru if stru == None else stru

        n = stru.GetNbScatterer()
        self.n = n

        lat = stru.GetLatticePar()
        self.lat = [lat[0], lat[1], lat[2], np.degrees(lat[3]), np.degrees(lat[4]), np.degrees(lat[5])]
        self.element = []
        self.occ = []
        self.xyz = []
        self.uij_c = []
        self.anisotropy = []
        for i in range(n):
            atom = stru.GetScatterer(i)
            st = atom.GetScatteringPower()
            self.element.append(st.GetSymbol())
            self.occ.append(atom.Occupancy)
            self.xyz.append([atom.X, atom.Y, atom.Z])
            bij_c = np.array([[st.B11, st.B12, st.B13],
                              [st.B12, st.B22, st.B12],
                              [st.B13, st.B12, st.B33]])
            biso = st.Biso
            if bij_c.sum() < 1.0e-10:
                self.anisotropy.append(False)
                self.uij_c.append(biso / np.pi ** 2 / 8 * np.identity(3))
            else:
                self.anisotropy.append(True)
                self.uij_c.append(bij_c / np.pi ** 2 / 8)

        self.xyz = np.array(self.xyz)
        self.occ = np.array(self.occ)
        self.uij_c = np.array(self.uij_c)

        sstru = self.convertDiffpyStru('xyz')
        self.parseDiffpyStru(sstru)
        return

    ###########################################################
    # convert functions
    ###########################################################

    def convertPeriodicStru(self, mode='xyz'):
        '''
        conver the self.xxx to PeriodicStructureAdapter
        
        :param mode: 'xyz' or 'xyz_c',
            'xyz': pass fractional xyz and covert to Cartesian xyz
            'xyz_c': pass Cartesian xyz directly 
        '''
        rv = PeriodicStructureAdapter()
        if mode == 'xyz':
            rv.setLatPar(*self.lat)

        del rv[:]
        rv.reserve(self.n)
        aa = AdapterAtom()
        for ele, occ, aniso in itertools.izip(self.element, self.occ, self.anisotropy):
            aa.atomtype = ele
            aa.occupancy = occ
            aa.anisotropy = bool(aniso)
            rv.append(aa)

        if mode == 'xyz':
            for a, xyz, uij_c in itertools.izip(rv, self.xyz, self.uij_c):
                a.xyz_cartn = xyz
                a.uij_cartn = uij_c
                rv.toCartesian(a)
        elif mode == 'xyz_c':
            for a, xyz_c, uij_c in itertools.izip(rv, self.xyz_c, self.uij_c):
                a.xyz_cartn = xyz_c
                a.uij_cartn = uij_c
        # if np.allclose(np.array(self.lat), np.array([1.0, 1.0, 1.0, 90.0, 90.0, 90.0])):
        #    rv = nosymmetry(rv)
        return self.addProp(rv)

    def convertDiffpyStru(self, mode='xyz'):
        '''
        convert self.xxx to diffpy
        
        :param mode: 'xyz' or 'xyz_c',
            'xyz': pass fractional xyz
            'xyz_c': pass Cartesian xyz directly
        '''
        rv = Structure()
        if mode == 'xyz':
            rv.lattice.setLatPar(*self.lat)

        aa = Atom()
        for i in range(self.n):
            rv.append(aa, copy=True)

        rv.element = self.element
        rv.occupancy = self.occ
        rv.anisotropy = self.anisotropy
        rv.U = self.uij_c
        if mode == 'xyz':
            rv.xyz = self.xyz
        elif mode == 'xyz_c':
            rv.xyz_cartn = self.xyz_c
        rv.title = self.name
        return self.addProp(rv)

    def convertObjcrystStru(self, mode='xyz_c'):
        '''
        convert self.xxx to objcryst object
        
        only applied to non-periodic structure
        '''
        if self.periodic:
            # raise TypeError('Cannot convert to periodic structure')
            if self.rawstype == 'diffpy':
                cif = self.rawstru.write('temp.cif', 'cif')
                objcryst = CreateCrystalFromCIF(file('temp.cif'))
                rv = objcryst
                os.remove('temp.cif')
        else:
            c = Crystal(1, 1, 1, "P1")
            c.SetName(self.name)
            m = Molecule(c, self.name)
            c.AddScatterer(m)

            for i in range(self.n):
                ele = self.element[i]
                sp = ScatteringPowerAtom(self.element[i], ele)
                if self.anisotropy[i]:
                    uij = self.uij_c[i]
                    sp.B11 = uij[0, 0]
                    sp.B22 = uij[1, 1]
                    sp.B33 = uij[2, 2]
                    sp.B12 = uij[0, 1]
                    sp.B13 = uij[0, 2]
                    sp.B23 = uij[1, 2]
                else:
                    biso = np.sum(self.uij_c[i].diagonal()) / 3 * (8 * np.pi ** 2)
                    sp.SetBiso(biso)
                if mode == 'xyz':
                    x, y, z = map(float, self.xyz[i])
                else:
                    x, y, z = map(float, self.xyz_c[i])
                a = m.AddAtom(x, y, z, sp, "%s%i" % (ele, i + 1))
                a.Occupancy = self.occ[i]
            rv = m
        return self.addProp(rv)

    def superCell(self, mno, stru=None, replace=False):
        from diffpy.Structure.expansion import supercell
        if stru == None:
            stru = self.convertDiffpyStru('xyz')
            newstru = supercell(stru, mno)

        if replace:
            self.rawstru = newstru
            self.rawstype = 'diffpy'
            self.parseDiffpyStru(newstru)
        return self.addProp(newstru)

if __name__ == '__main__':
    stru = StructureExt()
    stru.loadStrufile('CdS.cif', 'diffpy', periodic=True)
    # stru.loadStrufile('fitresult/test1/m.cif', 'objcryst', False)
    # stru.loadStrufile('cd35se20.xyz', 'diffpy', periodic=False)
    stru.parseDiffpyStru()
    # stru.parseObjcrystStru()
    stru.superCell([10, 10, 10], replace=True)

    a = stru.convertPeriodicStru('xyz')
    # b = stru.convertDiffpyStru('xyz')
    # c = stru.convertObjcrystStru('xyz_c')
    # d = stru.superCell([5,5,5])
    # stru.exportStru('test.stru', 'pdb', 'diffpy')
    pass

    from diffpy.srreal.pdfcalculator import PDFCalculator, DebyePDFCalculator
    from diffpy.srreal.bondcalculator import BondCalculator
    from matplotlib.pyplot import plot, show

    # stru = loadStructure('CdS.cif')
    pc1 = PDFCalculator()
    pc1.rmax = 20
    r1, g1 = pc1(a)
    plot(r1, g1)

    # r2, g2 = pc1(b)
    # plot(r2, g2)

    # r3, g3 = pc1(c)
    # plot(r3, g3)
    show()
